import einops
import torch
import torch.nn as nn
from packaging import version

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"
OPENAIUNETWRAPPERRAIG = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapperRAIG"
OPENAIUNETWRAPPERCONTROLLDM3D = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapperControlLDM3D"
OPENAIUNETWRAPPERCONTROLLDM3DSSN = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapperControlLDM3DSSN"
OPENAIUNETWRAPPERCONTROLLDM3DTV2V = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapperControlLDM3DTV2V"
OPENAIUNETWRAPPERCONTROLLDM3DTV2V_INTERPOLATE = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapperControlLDM3DTV2VInterpolate"

class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        model_dtype = self.diffusion_model.time_embed[0].weight.dtype
        x = x.to(model_dtype)
        vector = c.get("vector", None)
        if vector is not None:
            vector = vector.to(model_dtype)
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=vector,
            **kwargs
        )


class OpenAIWrapperRAIG(OpenAIWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        model_dtype = self.diffusion_model.time_embed[0].weight.dtype
        x = x.to(model_dtype)
        vector = c.get("vector", None)
        if vector is not None:
            vector = vector.to(model_dtype)
        cond_feat = c.get("cond_feat", None)
        if cond_feat is not None:
            cond_feat = cond_feat.to(model_dtype)
            img_control = self.diffusion_model.controlnet_img(
                x=x,
                hint=cond_feat,
                timesteps=t,
                context=c.get(
                    "crossattn", None
                ),
                y=c.get("vector", None),
                **kwargs
            )
        else:
            img_control = None
        # actually, img_control is not used. features needed are hooked during the forward process
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=vector,
            **kwargs
        )


class OpenAIWrapperControlLDM3D(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        cond_feat = c["cond_feat"]

        model_dtype = self.diffusion_model.controlnet.input_hint_block[0].weight.dtype
        x = x.to(model_dtype)
        cond_feat = cond_feat.to(model_dtype)

        control = self.diffusion_model.controlnet(
            x=x,  # noisy control image, use or not used it depend on control_model style
            hint=cond_feat,  # control image B C H W
            timesteps=t,  # time step
            context=c.get(
                "crossattn", None
            ),  # text prompt, use or not used it depend on control_model style
            y=c.get("vector", None),
            **kwargs
        )

        out = self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            control=control,
            only_mid_control=False, 
            **kwargs
        )

        return out


class OpenAIWrapperControlLDM3DSSN(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        cond_feat = c["cond_feat"]

        model_dtype = self.diffusion_model.controlnet.input_hint_block[0].weight.dtype
        x = x.to(model_dtype)
        cond_feat = cond_feat.to(model_dtype)

        control, img_emb = self.diffusion_model.controlnet(
            x=x,  # noisy control image, use or not used it depend on control_model style
            hint=cond_feat,  # control image B C H W
            timesteps=t,  # time step
            context=c.get(
                "crossattn", None
            ),  # text prompt, use or not used it depend on control_model style
            y=c.get("vector", None),
            **kwargs
        )

        out = self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            control=control,
            img_emb=img_emb,
            only_mid_control=False, 
            **kwargs
        )

        return out


# -----------------------------------------------------
# This is used for TV2V (text-video-to-video) generation
class OpenAIWrapperControlLDM3DTV2V(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        control_hint = c["control_hint"]    # -1 ~ 1
        control_hint = (control_hint + 1) / 2.0  # 0 ~ 1
        control_hint = 1.0 - control_hint # this follow the official controlNet (refer control 1.1 in the gradio_lineart.py)

        model_dtype = self.diffusion_model.controlnet.input_hint_block[0].weight.dtype
        x = x.to(model_dtype)
        control_hint = control_hint.to(model_dtype)

        control = self.diffusion_model.controlnet(
            x=x,  
            hint=control_hint,  
            timesteps=t, 
            context=c.get(
                "crossattn", None
            ), 
            y=c.get("vector", None),
            **kwargs
        )
        cond_feat = c.get("cond_feat", None)
        if cond_feat is not None:
            cond_feat = cond_feat.to(model_dtype)
            img_control = self.diffusion_model.controlnet_img(
                x=x[:,:,x.shape[2]//2,:,:],
                hint=cond_feat,
                timesteps=t,
                context=c.get(
                    "crossattn", None
                ),
                y=c.get("vector", None),
                **kwargs
            )
        else:
            img_control = None
        # control = [each * 0.5 for each in control]
        # control = [each * 0. for each in control]    # !!!!!! this is for test, remove it later

        out = self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            control=control,
            img_control=img_control,
            only_mid_control=False, 
            **kwargs
        )

        return out


class OpenAIWrapperControlLDM3DTV2VInterpolate(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        control_hint = c["control_hint"]    # -1 ~ 1
        control_hint = (control_hint + 1) / 2.0  # 0 ~ 1
        control_hint = 1.0 - control_hint # this follow the official controlNet (refer control 1.1 in the gradio_lineart.py)

        model_dtype = self.diffusion_model.controlnet.input_hint_block[0].weight.dtype
        x = x.to(model_dtype)
        control_hint = control_hint.to(model_dtype)

        control = self.diffusion_model.controlnet(
            x=x,  
            hint=control_hint,  
            timesteps=t, 
            context=c.get(
                "crossattn", None
            ), 
            y=c.get("vector", None),
            **kwargs
        )
        assert 'interpolate_first_last' in c
        interpolate_first = c['interpolate_first_last'][:,:,0,:,:]
        interpolate_last = c['interpolate_first_last'][:,:,1,:,:]
        x_tmp = torch.cat((x[:,:,0,:,:], x[:,:,-1,:,:]), dim=0)
        interpolate_tmp = torch.cat((interpolate_first, interpolate_last), dim=0)
        t_tmp = torch.cat((t, t), dim=0)
        context_tmp = torch.cat((c['crossattn'], c['crossattn']), dim=0) if 'crossattn' in c else None
        y_tmp = torch.cat([c['vector'], c['vector']], dim=0) if 'vector' in c else None
        interpolate_control = self.diffusion_model.controlnet_img(
            x=x_tmp,
            hint=interpolate_tmp,
            timesteps=t_tmp,
            context=context_tmp,
            y=y_tmp,
            **kwargs
        )
        interpolate_control = [each.chunk(2) for each in interpolate_control]
        interpolate_control_first, interpolate_control_last = zip(*interpolate_control)
        interpolate_control_first = list(interpolate_control_first)
        interpolate_control_last = list(interpolate_control_last)
        out = self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            control=control,
            interpolate_control=(interpolate_control_first, interpolate_control_last),
            only_mid_control=False, 
            **kwargs
        )

        return out

