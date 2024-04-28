import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
import einops
from ...modules.diffusionmodules.util import (
    conv_nd,
    timestep_embedding,
    zero_module,
)
from ...util import default, exists, instantiate_from_config

from sgm.modules.attention import BasicTransformerBlock, SpatialTransformer, SpatialTransformerCA, SpatialTransformer3DCA
from sgm.modules.diffusionmodules.openaimodel import (
    spatial_temporal_forward,
    TimestepEmbedSequential,
    UNetModel,
    UNetModel3D,
)


class ControlNet3D(UNetModel3D):
    """A locked copy branch of UNetModel3D that processes task-specific conditions.
    The model weights are initilized from the weights of the pretrained UNetModel3D.
    The additional input_hint_block is used to transform the input condition into the
    same dimension as the output of the vae-encoder
    """

    def __init__(
        self, hint_channels, control_scales, disable_temporal=False, *args, **kwargs
    ):
        kwargs["out_channels"] = kwargs["in_channels"]  # this is unused actually
        self.control_scales = control_scales
        # Note: disable_temporal means only conduct 2d operation on the center frame
        self.disable_temporal = disable_temporal
        super().__init__(*args, **kwargs)

        model_channels = kwargs["model_channels"]
        channel_mult = kwargs["channel_mult"]
        del self.output_blocks
        del self.out
        del self.out_temporal
        if hasattr(self, "id_predictor"):
            del self.id_predictor
            del self.id_predictor_temporal

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(2, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 16, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 32, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 96, 256, 3, padding=1),
            nn.SiLU(),
            zero_module(conv_nd(2, 256, model_channels, 3, padding=1)),
        )

        # this is for the transformation of hint
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])
        if disable_temporal:
            self.zero_convs_temporal = [None]
        else:
            self.zero_convs_temporal = nn.ModuleList(
                [self.make_zero_conv(model_channels, dims=1)]
            )

        input_block_chans = [model_channels]
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                ch = mult * model_channels
                self.zero_convs.append(self.make_zero_conv(ch))
                if disable_temporal:
                    self.zero_convs_temporal.append(None)
                else:
                    self.zero_convs_temporal.append(self.make_zero_conv(ch, dims=1))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.zero_convs.append(self.make_zero_conv(ch))
                if disable_temporal:
                    self.zero_convs_temporal.append(None)
                else:
                    self.zero_convs_temporal.append(self.make_zero_conv(ch, dims=1))

        self.middle_block_out = self.make_zero_conv(ch)
        if disable_temporal:
            self.middle_block_out_temporal = None
        else:
            self.middle_block_out_temporal = self.make_zero_conv(ch, dims=1)

        if disable_temporal:
            self.setup_disbale_temporal()

    def setup_disbale_temporal(self):
        from sgm.util import torch_dfs
        from sgm.modules.diffusionmodules.openaimodel import (
            ResBlock3D,
            Upsample3D,
            Downsample3D,
        )
        from sgm.modules.attention import SpatialTransformer3D

        self.input_blocks_temporal = None
        all_modules = torch_dfs(self)
        for module in all_modules:
            if isinstance(module, ResBlock3D):
                module.in_layers_temporal = None
                module.out_layers_temporal = None
                if hasattr(module, "skip_connection_temporal"):
                    module.skip_connection_temporal = None
                if hasattr(module, "alpha_temporal1"):
                    module.alpha_temporal1 = None
                if hasattr(module, "alpha_temporal2"):
                    module.alpha_temporal2 = None
            if isinstance(module, SpatialTransformer3D):
                del module.norm_temporal
                del module.proj_in_temporal
                del module.transformer_blocks_temporal
                del module.proj_out_temporal
                if hasattr(module, "alpha_temporal"):
                    del module.alpha_temporal
            if isinstance(module, Downsample3D) or isinstance(module, Upsample3D):
                if hasattr(module, "conv_temporal"):
                    module.conv_temporal = None
        return

    def make_zero_conv(self, channels, dims=2):
        return TimestepEmbedSequential(
            zero_module(conv_nd(dims, channels, channels, 1, padding=0))
        )

    def forward(self, x, hint, timesteps=None, context=None, y=None, **kwargs):
        if self.disable_temporal:
            x = x[:, :, x.shape[2] // 2, :, :].unsqueeze(2)
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        t_emb = t_emb.to(self.input_hint_block[0].weight.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        context = (
            context.to(self.input_hint_block[0].weight.dtype)
            if context is not None
            else None
        )
        guided_hint = self.input_hint_block(hint, emb, context)
        outs = []

        h = x
        for module, zero_conv, zero_conv_temporal in zip(
            self.input_blocks, self.zero_convs, self.zero_convs_temporal
        ):
            if guided_hint is not None:
                h = spatial_temporal_forward(
                    h, module, self.input_blocks_temporal, emb=emb, context=context
                )
                frame_length = h.shape[2]
                guided_hint = repeat(
                    guided_hint, "b c h w -> b c t h w", t=frame_length
                )
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(spatial_temporal_forward(h, zero_conv, zero_conv_temporal))
        h = self.middle_block(h, emb, context)
        outs.append(
            spatial_temporal_forward(
                h, self.middle_block_out, self.middle_block_out_temporal
            )
        )
        control_scales = [self.control_scales for _ in range(len(outs))]
        control = [
            c * scale for c, scale in zip(outs, control_scales)
        ]  # Adjusting the strength of control

        return control


# -----------------------------------------------------
# This is used for TV2V (text-video-to-video) generation
class ControlNet2D(UNetModel):
    def __init__(self, hint_channels, control_scales, no_add_x=False, set_input_hint_block_as_identity=False, *args, **kwargs):
        kwargs["out_channels"] = kwargs["in_channels"]  # this is unused actually
        super().__init__(*args, **kwargs)

        self.control_scales = control_scales
        model_channels = kwargs["model_channels"]
        channel_mult = kwargs["channel_mult"]
        del self.output_blocks
        del self.out
        if hasattr(self, "id_predictor"):
            del self.id_predictor

        self.set_input_hint_block_as_identity = set_input_hint_block_as_identity
        if set_input_hint_block_as_identity:
            self.input_hint_block = TimestepEmbedSequential(
            nn.Identity()
            )
            # though set input_hint_block as identity,
        else:
            self.input_hint_block = TimestepEmbedSequential(
                conv_nd(2, hint_channels, 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(2, 16, 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(2, 16, 32, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(2, 32, 32, 3, padding=1),
                nn.SiLU(),
                conv_nd(2, 32, 96, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(2, 96, 96, 3, padding=1),
                nn.SiLU(),
                conv_nd(2, 96, 256, 3, padding=1, stride=2),
                nn.SiLU(),
                zero_module(conv_nd(2, 256, model_channels, 3, padding=1))
            )

        # this is for the transformation of hint
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        input_block_chans = [model_channels]
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                ch = mult * model_channels
                self.zero_convs.append(self.make_zero_conv(ch))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.zero_convs.append(self.make_zero_conv(ch))

        self.middle_block_out = self.make_zero_conv(ch)
        self.no_add_x = no_add_x

    def make_zero_conv(self, channels, dims=2):
        return TimestepEmbedSequential(zero_module(conv_nd(dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps=None, context=None, y=None, **kwargs):
        assert (y is not None) == (self.num_classes is not None), \
            "must specify y if and only if the model is class-conditional"
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        # t_emb = t_emb.to(self.input_hint_block[0].weight.dtype)
        t_emb = t_emb.to(self.input_blocks[0][0].weight.dtype)
        emb = self.time_embed(t_emb)

        if x.dim() == 5:
            is_video = True
            n_frames = x.shape[2]
            x = einops.rearrange(x, 'b c t h w -> (b t) c h w')
            hint = einops.rearrange(hint, 'b c t h w -> (b t) c h w')
            emb = einops.repeat(emb, 'b d -> (b t) d', t=n_frames)
            context = einops.repeat(context, 'b n d -> (b t) n d', t=n_frames) if context is not None else None
        else:
            is_video = False

        if self.num_classes is not None:
            if is_video:
                raise NotImplementedError("class-conditional video generation is not supported yet")
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        hint = hint.to(self.input_blocks[0][0].weight.dtype)
        emb = emb.to(self.input_blocks[0][0].weight.dtype)
        context = context.to(self.input_blocks[0][0].weight.dtype) if context is not None else None
        # hint = hint.to(self.input_hint_block[0].weight.dtype)
        # emb = emb.to(self.input_hint_block[0].weight.dtype)
        # context = context.to(self.input_hint_block[0].weight.dtype) if context is not None else None
        guided_hint = self.input_hint_block(hint, emb, context)
        if self.set_input_hint_block_as_identity:
            guided_hint = self.input_blocks[0](guided_hint, emb, context)
        outs = []

        # h = x.type(self.dtype)
        h = x
        # if self.no_add_x:
        #     h = torch.zeros_like(x)
        # else:
        #     h = x
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                # h = module(h, emb, context)
                # h += guided_hint
                if self.no_add_x:
                    h = guided_hint
                else:
                    h = module(h, emb, context)
                    h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        # Adjusting the strength of control
        control_scales = [self.control_scales for _ in range(len(outs))]
        control = [c * scale for c, scale in zip(outs, control_scales)]  

        if is_video:
            control = [einops.rearrange(each, '(b t) c h w -> b c t h w', t=n_frames) for each in control]

        return control


class ControlledUNetModel3DTV2V(UNetModel3D):
    """A trainable copy branch of UNetModel3D that processes the video inputs.
    The model weights are initilized from the weights of the pretrained UNetModel3D.
    """

    def __init__(self, controlnet_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controlnet = instantiate_from_config(controlnet_config)
        
        controlnet_img_config = kwargs.get("controlnet_img_config", None)
        if controlnet_img_config is not None:
            self.controlnet_img = instantiate_from_config(controlnet_img_config)

        # reference-aware condition
        crossframe_type = kwargs.get("crossframe_type", None)
        if crossframe_type is not None:
            assert hasattr(self, 'controlnet_img'), "must have controlnet_img if crossframe_type is not None"
            assert crossframe_type == 'reference', "only support reference-aware condition"
            self.crossframe_type = crossframe_type
            # register hook in controlnet_img
            self.bank_attn = []
            for name, module in self.controlnet_img.named_modules():
                if isinstance(module, SpatialTransformer):
                    print('registering attention hook for', name)
                    module.register_forward_hook(self._get_attn_hook)

            # hack the attention function in unet
            def hacked_spatialtransformer_inner_forward(self, x, context=None):
                assert hasattr(self, 'm_control'), "must have m_control if crossframe_type is not None"
                anchor_frame = self.m_control

                # x = super().forward(x, context)
                # note: if no context is given, cross-attention defaults to self-attention
                if not isinstance(context, list):
                    context = [context]
                b, c, t, h, w = x.shape
                # spatial attention
                x = rearrange(x, "b c t h w -> (b t) c h w").contiguous()
                x_in = x
                x = self.norm(x)
                if not self.use_linear:
                    x = self.proj_in(x)
                x = rearrange(x, "bt c h w -> bt (h w) c").contiguous()
                if self.use_linear:
                    x = self.proj_in(x)

                for i, block in enumerate(self.transformer_blocks):
                    if i > 0 and len(context) == 1:
                        i = 0
                    context_i = (
                        repeat(context[i], "b l c -> (b t) l c", t=t).contiguous()
                        if context[i] is not None
                        else None
                    )
                    x = block(x, context=context_i)
                if self.use_linear:
                    x = self.proj_out(x)
                x = rearrange(x, "bt (h w) c -> bt c h w", h=h, w=w).contiguous()
                if not self.use_linear:
                    x = self.proj_out(x)
                x = x + x_in

                x = rearrange(x, "(b t) c h w -> (b h w) c t", t=t).contiguous()
                # temporal attention
                if hasattr(self, "norm_temporal"):  # temporal operation exist
                    x_in = x
                    x = self.norm_temporal(x)
                    if not self.use_linear:
                        x = self.proj_in_temporal(x)
                    x = rearrange(x, "bhw c t->bhw t c").contiguous()
                    if self.use_linear:
                        x = self.proj_in_temporal(x)
                    for i, block in enumerate(self.transformer_blocks_temporal):
                        if i > 0 and len(context) == 1:
                            i = 0  # use same context for each block
                        # if context[i] != None:
                        context_i = (
                            repeat(context[i], "b l c -> (b h w) l c", h=h, w=w).contiguous()
                            if context[i] is not None
                            else None
                        )
                        # x = block(x, context=context_i)
                        if self.disable_temporal_text_ca:
                            x = block(x, context=x)
                        else:
                            x = block(x, context=context_i)
                    if self.use_linear:
                        x = self.proj_out_temporal(x)
                    x = rearrange(x, "bhw t c -> bhw c t").contiguous()
                    if not self.use_linear:
                        x = self.proj_out_temporal(x)
                    if hasattr(self, "alpha_temporal"):
                        x = self.alpha_temporal * x_in + (1 - self.alpha_temporal) * x
                    else:
                        x = x_in + x

                x = rearrange(x, "(b h w) c t -> b c t h w", h=h, w=w).contiguous()

                # note: if no context is given, cross-attention defaults to self-attention
                if not isinstance(context, list):
                    context = [context]
                b, c, t, h, w = x.shape
                # cross-frame attention
                x = rearrange(x, "b c t h w -> (b t) c h w").contiguous()
                x_in = x
                x = self.norm_temporal_ca(x)
                if not self.use_linear:
                    x = self.proj_in_temporal_ca(x)
                x = rearrange(x, "bt c h w -> bt (h w) c").contiguous()
                if self.use_linear:
                    x = self.proj_in_temporal_ca(x)

                for i, block in enumerate(self.transformer_blocks_temporal_ca):
                    if i > 0 and len(context) == 1:
                        i = 0
                    # # center frame as anchor
                    x = rearrange(x, "(b t) hw c -> b t hw c", b=b).contiguous()
                    # attn_anchor_frame_idx = t // 2  # center frame
                    # anchor_frame = x[:, attn_anchor_frame_idx, :, :].contiguous()
                    # anchor_frame = repeat(anchor_frame, "b hw c -> b t hw c", t=t).contiguous()

                    # anchor_frame = repeat(anchor_frame, "b c h w -> b t (h w) c", t=t).contiguous()
                    anchor_frame = repeat(anchor_frame, "b c h w -> b t h w c", t=t).contiguous()
                    anchor_frame = rearrange(anchor_frame, "b t h w c -> b t (h w) c").contiguous()
                    anchor_frame = rearrange(anchor_frame, "b t hw c -> (b t) hw c").contiguous()
                    context_texture = anchor_frame
                    x = rearrange(x, "b t hw c -> (b t) hw c", b=b).contiguous()
                    x = block(x, context_texture)

                if self.use_linear:
                    x = self.proj_out_temporal_ca(x)
                x = rearrange(x, "bt (h w) c -> bt c h w", h=h, w=w).contiguous()
                if not self.use_linear:
                    x = self.proj_out_temporal_ca(x)
                # print(x.min(), x.max()) #! debug
                x = x + x_in

                x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t).contiguous()

                return x

            all_modules = torch_dfs(self)
            st_modules = [module for module in all_modules if isinstance(module, SpatialTransformer3DCA)]    # st = spatialtransformer

            # hard code, the first 7 st modules are used for reference aware cross-frame attention
            for i, module in enumerate(st_modules[:7]):
                if getattr(module, 'original_inner_forward', None) is None:
                    module.original_inner_forward = module.forward
                module.forward = hacked_spatialtransformer_inner_forward.__get__(module, SpatialTransformer3DCA)
                # module.attn1_type = spatial_transformer_attn1_type
    
    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        control=None,
        img_control=None, 
        only_mid_control=False,
        **kwargs
    ):
        # 1. If img_control is not None, img_control would be added on the center frame of the video.
        # 2. Note that control (lineart maps or something) would conduct on the whole video, 
        #    which controls the global motion or structure.
        #    But img_control would only conduct on the center frame, which controls the local texture.
        #    The texture introduced from img_control would spread to the whole video through the temporal blocks.
        # 3. Note that control is added in the decoder, while img_control is added in the encoder.

        if hasattr(self, 'crossframe_type') and self.crossframe_type == 'reference':
            all_modules = torch_dfs(self)
            st_modules = [module for module in all_modules if isinstance(module, SpatialTransformer3DCA)]    # st = spatialtransformer

            # control_attn = self.bank_attn
            # st_modules = st_modules[:len(control_attn)]
            # for (module, m_control) in zip(st_modules, control_attn):
            #     module.m_control = m_control
            st_modules = st_modules[:7]
            assert len(self.bank_attn) == 7, "hard code, the first 7 st modules are used for reference aware cross-frame attention" \
                "and the number in self.bank_attn is {} now".format(len(self.bank_attn))
            # for (module, m_control) in zip(st_modules, self.bank_attn):
            for module in st_modules:
                module.m_control = self.bank_attn.pop(0)
            assert len(self.bank_attn) == 0, "self.bank_attn should be empty now"

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        t_emb = t_emb.to(self.input_blocks_temporal[0].weight.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)
        context = (
            context.to(self.input_blocks_temporal[0].weight.dtype)
            if context is not None
            else None
        )
        h = x
        for layer, module in enumerate(self.input_blocks):
            if layer == 0:
                h = spatial_temporal_forward(
                    h, module, self.input_blocks_temporal, emb=emb, context=context
                )
            else:
                h = module(h, emb, context)
            if (not only_mid_control) and (img_control is not None):
                h[:,:,h.shape[2]//2,:,:] += img_control.pop(0)
            hs.append(h)
                
        h = self.middle_block(h, emb, context)
        if img_control is not None:
            h[:,:,h.shape[2]//2,:,:] += img_control.pop(0)
        if control is not None:
            h = h + control.pop()  # B C T H W

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = th.cat([h, hs.pop()], dim=1)
            else:
                h = th.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            assert False, "not supported anymore. what the f*** are you doing?"  # niubi
        else:
            return spatial_temporal_forward(h, self.out, self.out_temporal)
        
    def _get_attn_hook(self, module, input, output):
        self.bank_attn.append(output)


class ControlledUNetModel3DTV2VInterpolate(ControlledUNetModel3DTV2V):
    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        control=None,
        interpolate_control=None, 
        only_mid_control=False,
        **kwargs
    ):
        assert control is not None
        assert interpolate_control is not None
        interpolate_control_first, interpolate_control_last = interpolate_control

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        t_emb = t_emb.to(self.input_blocks_temporal[0].weight.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)
        context = (
            context.to(self.input_blocks_temporal[0].weight.dtype)
            if context is not None
            else None
        )
        h = x
        for layer, module in enumerate(self.input_blocks):
            if layer == 0:
                h = spatial_temporal_forward(
                    h, module, self.input_blocks_temporal, emb=emb, context=context
                )
            else:
                h = module(h, emb, context)
            if (not only_mid_control):
                h[:,:,0,:,:] += interpolate_control_first.pop(0)
                h[:,:,-1,:,:] += interpolate_control_last.pop(0)
            hs.append(h)
                
        h = self.middle_block(h, emb, context)
        h[:,:,0,:,:] += interpolate_control_first.pop(0)
        h[:,:,-1,:,:] += interpolate_control_last.pop(0)
        h = h + control.pop()  # B C T H W

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = th.cat([h, hs.pop()], dim=1)
            else:
                h = th.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            assert False, "not supported anymore. what the f*** are you doing?"  # niubi
        else:
            return spatial_temporal_forward(h, self.out, self.out_temporal)


class ControlledUNetModel2DRAIG(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        controlnet_img_config = kwargs.get("controlnet_img_config", None)
        if controlnet_img_config is not None:
            self.controlnet_img = instantiate_from_config(controlnet_img_config)

        # reference-aware condition
        enable_ref_attn = kwargs.get("enable_ref_attn", False)
        self.enable_ref_attn = enable_ref_attn
        if enable_ref_attn:
            assert hasattr(self, 'controlnet_img'), "must have controlnet_img if crossframe_type is not None"
            # register hook in controlnet_img
            self.bank_attn = []
            for name, module in self.controlnet_img.named_modules():
                if isinstance(module, SpatialTransformer):
                    print('registering attention hook for', name)
                    module.register_forward_hook(self._get_attn_hook)
    
    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        control=None,
        img_control=None, 
        only_mid_control=False,
        **kwargs
    ): 
        assert img_control == None, 'img_control should not shown here, features needed are hooked during the forward process'

        if self.enable_ref_attn:
            all_modules = torch_dfs(self)
            st_modules = [module for module in all_modules if isinstance(module, SpatialTransformerCA)]    # st = spatialtransformer
            
            # hard code, might be changed later
            mapping_dict = {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                4: 4,
                5: 5,
                6: 6,
                7: 5,
                8: 5,
                9: 4,
                10: 3,
                11: 3,
                12: 2,
                13: 1,
                14: 1,
                15: 0,
            }

            for idx, module in enumerate(st_modules):
                module.ref_control = self.bank_attn[mapping_dict[idx]]

            self.bank_attn = []

        assert (y is not None) == (self.num_classes is not None), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        t_emb = t_emb.to(self.output_blocks[0][0].in_layers[0].weight.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)
        context = (
            context.to(self.output_blocks[0][0].in_layers[0].weight.dtype)
            if context is not None
            else None
        )
        h = x
        for layer, module in enumerate(self.input_blocks):
            h = module(h, emb, context)
            hs.append(h)
                
        h = self.middle_block(h, emb, context)
        if control is not None:
            h = h + control.pop()  # B C T H W

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = th.cat([h, hs.pop()], dim=1)
            else:
                h = th.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            assert False, "not supported anymore. what the f*** are you doing?"
        else:
            return self.out(h)

    def _get_attn_hook(self, module, input, output):
        self.bank_attn.append(output)


# DFS Search for Torch.nn.Module, Written by Lvmin
def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result