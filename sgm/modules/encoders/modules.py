from contextlib import nullcontext
from functools import partial
from typing import Dict, List, Optional, Tuple, Union
import torch.nn.functional as F

import random
import omegaconf
import kornia
import numpy as np
import open_clip
import torch
import torch.nn as nn
import einops
from einops import rearrange, repeat
from omegaconf import ListConfig
from torch.utils.checkpoint import checkpoint
from transformers import (
    ByT5Tokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)

from ...modules.autoencoding.regularizers import DiagonalGaussianRegularizer
from ...modules.diffusionmodules.model import Encoder
from ...modules.diffusionmodules.openaimodel import Timestep
from ...modules.diffusionmodules.util import extract_into_tensor, make_beta_schedule
from ...modules.distributions.distributions import DiagonalGaussianDistribution
from ...util import (
    autocast,
    count_params,
    default,
    disabled_train,
    expand_dims_like,
    instantiate_from_config,
)


class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key


class GeneralConditioner(nn.Module):
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}

    def __init__(self, emb_models: Union[List, ListConfig]):
        super().__init__()
        embedders = []
        for n, embconfig in enumerate(emb_models):
            embedder = instantiate_from_config(embconfig)
            assert isinstance(
                embedder, AbstractEmbModel
            ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
            embedder.is_trainable = embconfig.get("is_trainable", False)
            embedder.ucg_rate = embconfig.get("ucg_rate", 0.0)
            if not embedder.is_trainable:
                embedder.train = disabled_train
                for param in embedder.parameters():
                    param.requires_grad = False
                embedder.eval()
            print(
                f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
            )

            if "input_key" in embconfig:
                embedder.input_key = embconfig["input_key"]
            elif "input_keys" in embconfig:
                embedder.input_keys = embconfig["input_keys"]
            else:
                raise KeyError(
                    f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}"
                )

            embedder.legacy_ucg_val = embconfig.get("legacy_ucg_value", None)
            if embedder.legacy_ucg_val is not None:
                embedder.ucg_prng = np.random.RandomState()

            embedders.append(embedder)
        self.embedders = nn.ModuleList(embedders)

    def possibly_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict) -> Dict:
        assert embedder.legacy_ucg_val is not None
        p = embedder.ucg_rate
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                batch[embedder.input_key][i] = val
        return batch

    def forward(
        self, batch: Dict, force_zero_embeddings: Optional[List] = None
    ) -> Dict:
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []
        for embedder in self.embedders:
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                    if embedder.legacy_ucg_val is not None:
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    emb_out = embedder(batch[embedder.input_key])
                elif hasattr(embedder, "input_keys"):
                    emb_out = embedder(*[batch[k] for k in embedder.input_keys])
            assert isinstance(
                emb_out, (torch.Tensor, list, tuple)
            ), f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"
            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]
            for emb in emb_out:
                if hasattr(embedder, "input_key") and (embedder.input_key == "cond_img"):
                    out_key = "cond_feat"
                elif hasattr(embedder, "input_key") and (embedder.input_key == "interpolate_first"):
                    out_key = "interpolate_first"
                elif hasattr(embedder, "input_key") and (embedder.input_key == "interpolate_last"):
                    out_key = "interpolate_last"
                elif hasattr(embedder, "input_key") and (embedder.input_key == "interpolate_first_last"):
                    out_key = "interpolate_first_last"
                elif hasattr(embedder, 'input_key') and (embedder.input_key == 'control_hint'):
                    out_key = 'control_hint'
                else:
                    out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
                if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None: # zeros out embeddings with probability ucg_rate
                    emb = (
                        expand_dims_like(
                            torch.bernoulli(
                                (1.0 - embedder.ucg_rate)
                                * torch.ones(emb.shape[0], device=emb.device)
                            ),
                            emb,
                        )
                        * emb
                    )
                if (
                    hasattr(embedder, "input_key")
                    and embedder.input_key in force_zero_embeddings
                ):
                    emb = torch.zeros_like(emb)
                if out_key in output:
                    output[out_key] = torch.cat(
                        (output[out_key], emb), self.KEY2CATDIM[out_key]
                    )
                else:
                    output[out_key] = emb
        return output

    def get_unconditional_conditioning(
        self, batch_c, batch_uc=None, force_uc_zero_embeddings=None
    ):
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        c = self(batch_c)
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)

        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        return c, uc


class InceptionV3(nn.Module):
    """Wrapper around the https://github.com/mseitzer/pytorch-fid inception
    port with an additional squeeze at the end"""

    def __init__(self, normalize_input=False, **kwargs):
        super().__init__()
        from pytorch_fid import inception

        kwargs["resize_input"] = True
        self.model = inception.InceptionV3(normalize_input=normalize_input, **kwargs)

    def forward(self, inp):
        # inp = kornia.geometry.resize(inp, (299, 299),
        #                              interpolation='bicubic',
        #                              align_corners=False,
        #                              antialias=True)
        # inp = inp.clamp(min=-1, max=1)

        outp = self.model(inp)

        if len(outp) == 1:
            return outp[0].squeeze()

        return outp


class IdentityEncoder(AbstractEmbModel):
    def encode(self, x):
        return x

    def forward(self, x):
        return x


class ClassEmbedder(AbstractEmbModel):
    def __init__(self, embed_dim, n_classes=1000, add_sequence_dim=False):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.add_sequence_dim = add_sequence_dim

    def forward(self, c):
        c = self.embedding(c)
        if self.add_sequence_dim:
            c = c[:, None, :]
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = (
            self.n_classes - 1
        )  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc.long()}
        return uc


class ClassEmbedderForMultiCond(ClassEmbedder):
    def forward(self, batch, key=None, disable_dropout=False):
        out = batch
        key = default(key, self.key)
        islist = isinstance(batch[key], list)
        if islist:
            batch[key] = batch[key][0]
        c_out = super().forward(batch, key, disable_dropout)
        out[key] = [c_out] if islist else c_out
        return out


class FrozenT5Embedder(AbstractEmbModel):
    """Uses the T5 transformer encoder for text"""

    def __init__(
        self, version="google/t5-v1_1-xxl", device="cuda", max_length=77, freeze=True
    ):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    # @autocast
    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        with torch.autocast("cuda", enabled=False):
            outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenByT5Embedder(AbstractEmbModel):
    """
    Uses the ByT5 transformer encoder for text. Is character-aware.
    """

    def __init__(
        self, version="google/byt5-base", device="cuda", max_length=77, freeze=True
    ):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = ByT5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        with torch.autocast("cuda", enabled=False):
            outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEmbModel):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        layer_idx=None,
        always_return_pooled=False,
    ):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        self.return_pooled = always_return_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(
            input_ids=tokens, output_hidden_states=self.layer == "hidden"
        )
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        if self.return_pooled:
            return z, outputs.pooler_output
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder2(AbstractEmbModel):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    LAYERS = ["pooled", "last", "penultimate"]

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        always_return_pooled=False,
        legacy=True,
    ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),
            pretrained=version,
        )
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        self.return_pooled = always_return_pooled
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        self.legacy = legacy

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        if not self.return_pooled and self.legacy:
            return z
        if self.return_pooled:
            assert not self.legacy
            return z[self.layer], z["pooled"]
        return z[self.layer]

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        if self.legacy:
            x = x[self.layer]
            x = self.model.ln_final(x)
            return x
        else:
            # x is a dict and will stay a dict
            o = x["last"]
            o = self.model.ln_final(o)
            pooled = self.pool(o, text)
            x["pooled"] = pooled
            return x

    def pool(self, x, text):
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            @ self.model.text_projection
        )
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        outputs = {}
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - 1:
                outputs["penultimate"] = x.permute(1, 0, 2)  # LND -> NLD
            if (
                self.model.transformer.grad_checkpointing
                and not torch.jit.is_scripting()
            ):
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        outputs["last"] = x.permute(1, 0, 2)  # LND -> NLD
        return outputs

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEmbModel):
    LAYERS = [
        # "pooled",
        "last",
        "penultimate",
    ]

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        use_bf16=False,
    ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),
            pretrained=version,
            precision=torch.bfloat16 if use_bf16 else torch.float32,
        )
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if (
                self.model.transformer.grad_checkpointing
                and not torch.jit.is_scripting()
            ):
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPImageEmbedder(AbstractEmbModel):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        antialias=True,
        ucg_rate=0.0,
        unsqueeze_dim=False,
        repeat_to_max_len=False,
        num_image_crops=0,
        output_tokens=False,
    ):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),
            pretrained=version,
        )
        del model.transformer
        self.model = model
        self.max_crops = num_image_crops
        self.pad_to_max_len = self.max_crops > 0
        self.repeat_to_max_len = repeat_to_max_len and (not self.pad_to_max_len)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

        self.antialias = antialias

        self.register_buffer(
            "mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False
        )
        self.register_buffer(
            "std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False
        )
        self.ucg_rate = ucg_rate
        self.unsqueeze_dim = unsqueeze_dim
        self.stored_batch = None
        self.model.visual.output_tokens = output_tokens
        self.output_tokens = output_tokens

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )
        x = (x + 1.0) / 2.0
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, image, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        tokens = None
        if self.output_tokens:
            z, tokens = z[0], z[1]
        z = z.to(image.dtype)
        if self.ucg_rate > 0.0 and not no_dropout and not (self.max_crops > 0):
            z = (
                torch.bernoulli(
                    (1.0 - self.ucg_rate) * torch.ones(z.shape[0], device=z.device)
                )[:, None]
                * z
            )
            if tokens is not None:
                tokens = (
                    expand_dims_like(
                        torch.bernoulli(
                            (1.0 - self.ucg_rate)
                            * torch.ones(tokens.shape[0], device=tokens.device)
                        ),
                        tokens,
                    )
                    * tokens
                )
        if self.unsqueeze_dim:
            z = z[:, None, :]
        if self.output_tokens:
            assert not self.repeat_to_max_len
            assert not self.pad_to_max_len
            return tokens, z
        if self.repeat_to_max_len:
            if z.dim() == 2:
                z_ = z[:, None, :]
            else:
                z_ = z
            return repeat(z_, "b 1 d -> b n d", n=self.max_length), z
        elif self.pad_to_max_len:
            assert z.dim() == 3
            z_pad = torch.cat(
                (
                    z,
                    torch.zeros(
                        z.shape[0],
                        self.max_length - z.shape[1],
                        z.shape[2],
                        device=z.device,
                    ),
                ),
                1,
            )
            return z_pad, z_pad[:, 0, ...]
        return z

    def encode_with_vision_transformer(self, img):
        # if self.max_crops > 0:
        #    img = self.preprocess_by_cropping(img)
        if img.dim() == 5:
            assert self.max_crops == img.shape[1]
            img = rearrange(img, "b n c h w -> (b n) c h w")
        img = self.preprocess(img)
        if not self.output_tokens:
            assert not self.model.visual.output_tokens
            x = self.model.visual(img)
            tokens = None
        else:
            assert self.model.visual.output_tokens
            x, tokens = self.model.visual(img)
        if self.max_crops > 0:
            x = rearrange(x, "(b n) d -> b n d", n=self.max_crops)
            # drop out between 0 and all along the sequence axis
            x = (
                torch.bernoulli(
                    (1.0 - self.ucg_rate)
                    * torch.ones(x.shape[0], x.shape[1], 1, device=x.device)
                )
                * x
            )
            if tokens is not None:
                tokens = rearrange(tokens, "(b n) t d -> b t (n d)", n=self.max_crops)
                print(
                    f"You are running very experimental token-concat in {self.__class__.__name__}. "
                    f"Check what you are doing, and then remove this message."
                )
        if self.output_tokens:
            return x, tokens
        return x

    def encode(self, text):
        return self(text)


class FrozenCLIPT5Encoder(AbstractEmbModel):
    def __init__(
        self,
        clip_version="openai/clip-vit-large-patch14",
        t5_version="google/t5-v1_1-xl",
        device="cuda",
        clip_max_length=77,
        t5_max_length=77,
    ):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(
            clip_version, device, max_length=clip_max_length
        )
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        print(
            f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder) * 1.e-6:.2f} M parameters, "
            f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder) * 1.e-6:.2f} M params."
        )

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]


class SpatialRescaler(nn.Module):
    def __init__(
        self,
        n_stages=1,
        method="bilinear",
        multiplier=0.5,
        in_channels=3,
        out_channels=None,
        bias=False,
        wrap_video=False,
        kernel_size=1,
        remap_output=False,
    ):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in [
            "nearest",
            "linear",
            "bilinear",
            "trilinear",
            "bicubic",
            "area",
        ]
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None or remap_output
        if self.remap_output:
            print(
                f"Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing."
            )
            self.channel_mapper = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                padding=kernel_size // 2,
            )
        self.wrap_video = wrap_video

    def forward(self, x):
        if self.wrap_video and x.ndim == 5:
            B, C, T, H, W = x.shape
            x = rearrange(x, "b c t h w -> b t c h w")
            x = rearrange(x, "b t c h w -> (b t) c h w")

        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.wrap_video:
            x = rearrange(x, "(b t) c h w -> b t c h w", b=B, t=T, c=C)
            x = rearrange(x, "b t c h w -> b c t h w")
        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class LowScaleEncoder(nn.Module):
    def __init__(
        self,
        model_config,
        linear_start,
        linear_end,
        timesteps=1000,
        max_noise_level=250,
        output_size=64,
        scale_factor=1.0,
    ):
        super().__init__()
        self.max_noise_level = max_noise_level
        self.model = instantiate_from_config(model_config)
        self.augmentation_schedule = self.register_schedule(
            timesteps=timesteps, linear_start=linear_start, linear_end=linear_end
        )
        self.out_size = output_size
        self.scale_factor = scale_factor

    def register_schedule(
        self,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def forward(self, x):
        z = self.model.encode(x)
        if isinstance(z, DiagonalGaussianDistribution):
            z = z.sample()
        z = z * self.scale_factor
        noise_level = torch.randint(
            0, self.max_noise_level, (x.shape[0],), device=x.device
        ).long()
        z = self.q_sample(z, noise_level)
        if self.out_size is not None:
            z = torch.nn.functional.interpolate(z, size=self.out_size, mode="nearest")
        # z = z.repeat_interleave(2, -2).repeat_interleave(2, -1)
        return z, noise_level

    def decode(self, z):
        z = z / self.scale_factor
        return self.model.decode(z)


class ConcatTimestepEmbedderND(AbstractEmbModel):
    """embeds each dimension independently and concatenates them"""

    def __init__(self, outdim):
        super().__init__()
        self.timestep = Timestep(outdim)
        self.outdim = outdim

    def forward(self, x):
        if x.ndim == 1:
            x = x[:, None]
        assert len(x.shape) == 2
        b, dims = x.shape[0], x.shape[1]
        x = rearrange(x, "b d -> (b d)")
        emb = self.timestep(x)
        emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return emb


class GaussianEncoder(Encoder, AbstractEmbModel):
    def __init__(
        self, weight: float = 1.0, flatten_output: bool = True, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.posterior = DiagonalGaussianRegularizer()
        self.weight = weight
        self.flatten_output = flatten_output

    def forward(self, x) -> Tuple[Dict, torch.Tensor]:
        z = super().forward(x)
        z, log = self.posterior(z)
        log["loss"] = log["kl_loss"]
        log["weight"] = self.weight
        if self.flatten_output:
            z = rearrange(z, "b c h w -> b (h w ) c")
        return log, z


class VAEEmbedder(AbstractEmbModel):
    def __init__(self, down_blur_factor=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.down_blur_factor = down_blur_factor
        assert down_blur_factor >= 1, "down_blur_factor must be >= 1"

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # return x
        assert hasattr(self, "first_stage_model"), "first_stage_model not defined"
        assert hasattr(
            self, "disable_first_stage_autocast"
        ), "disable_first_stage_autocast not defined"
        assert hasattr(self, "scale_factor"), "scale_factor not defined"

        if self.down_blur_factor > 1:
            hx, wx = x.shape[-2:]
            # downsample
            x = torch.nn.functional.interpolate(
                x,
                scale_factor=1.0 / self.down_blur_factor,
                mode="bilinear",
                align_corners=False,
            )
            # upsample back
            x = torch.nn.functional.interpolate(
                x,
                size=(hx, wx),
                mode="bilinear",
                align_corners=False,
            )
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            z = self.first_stage_model.encode(x)
        z = self.scale_factor * z
        return z

    def encode(self, x):
        return self(x)


class CustomIdentityEncoder(AbstractEmbModel):
    def __init__(self, down_blur_factor=None, down_blur_probs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.down_blur_factor = down_blur_factor
        if down_blur_factor:
            assert down_blur_factor.__class__ == omegaconf.listconfig.ListConfig, "down_blur_factor must be a list"
            assert min(down_blur_factor) >= 1, "down_blur_factor must be >= 1"
            if down_blur_probs:
                assert down_blur_probs.__class__ == omegaconf.listconfig.ListConfig, "probs must be a list"
                assert len(down_blur_probs) == len(down_blur_factor), "probs must have the same length as down_blur_factor"
                assert sum(down_blur_probs) == 1, "probs must sum to 1"
                self.down_blur_probs = down_blur_probs
            else:
                self.down_blur_probs = [1.0/len(down_blur_factor) for _ in range(len(down_blur_factor))]

    def encode(self, x):
        return self(x)

    def forward(self, x):
        if self.down_blur_factor:
            factor = np.random.choice(self.down_blur_factor, p=self.down_blur_probs)

            hx, wx = x.shape[-2:]
            if x.dim() == 4:
                mode = "bilinear"
                size_down = int(hx / factor), int(wx / factor)
                size_ori = hx, wx
            elif x.dim() == 5:
                nframe = x.shape[2]
                mode = "trilinear"
                size_down = nframe, int(hx / factor), int(wx / factor)
                size_ori = nframe, hx, wx
            else:
                raise NotImplementedError("CustomIdentityEncoder only support 4D and 5D input")
            
            # downsample
            x = torch.nn.functional.interpolate(
                x,
                size=size_down,
                mode=mode,
                align_corners=False,
            )
            # upsample back
            x = torch.nn.functional.interpolate(
                x,
                size=size_ori,
                mode=mode,
                align_corners=False,
            )

            # if x.dim() == 4:
            #     hx, wx = x.shape[-2:]
            #     # downsample
            #     x = torch.nn.functional.interpolate(
            #         x,
            #         scale_factor=1.0 / factor,
            #         mode="bilinear",
            #         align_corners=False,
            #     )
            #     # upsample back
            #     x = torch.nn.functional.interpolate(
            #         x,
            #         size=(hx, wx),
            #         mode="bilinear",
            #         align_corners=False,
            #     )
            # elif x.dim() == 5:
            #     hx, wx = x.shape[-2:]
            #     nframe = x.shape[2]
            #     # downsample
            #     x = torch.nn.functional.interpolate(
            #         x,
            #         size=(nframe, int(hx / factor), int(wx / factor)),
            #         mode="trilinear",
            #         align_corners=False,
            #     )
            #     # upsample back
            #     x = torch.nn.functional.interpolate(
            #         x,
            #         size=(nframe, hx, wx),
            #         mode="trilinear",
            #         align_corners=False,
            #     )
            # else:
            #     raise NotImplementedError("CustomIdentityEncoder only support 4D and 5D input")

        return x


class CustomIdentityDownCondEncoder(CustomIdentityEncoder):
    def __init__(self, outdim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestep = Timestep(outdim)
        self.outdim = outdim
    
    def forward(self, x):
        if self.down_blur_factor:
            factor = np.random.choice(self.down_blur_factor, p=self.down_blur_probs)

            hx, wx = x.shape[-2:]
            if x.dim() == 4:
                mode = "bilinear"
                size_down = int(hx / factor), int(wx / factor)
                size_ori = hx, wx
            elif x.dim() == 5:
                nframe = x.shape[2]
                mode = "trilinear"
                size_down = nframe, int(hx / factor), int(wx / factor)
                size_ori = nframe, hx, wx
            else:
                raise NotImplementedError("CustomIdentityEncoder only support 4D and 5D input")
            
            # downsample
            x = torch.nn.functional.interpolate(
                x,
                size=size_down,
                mode=mode,
                align_corners=False,
            )
            # upsample back
            x = torch.nn.functional.interpolate(
                x,
                size=size_ori,
                mode=mode,
                align_corners=False,
            )

            factor = torch.tensor(factor).to(x.device).unsqueeze(0).float()
            factor = einops.repeat(factor, 'n -> b n', b=x.shape[0])
            assert len(factor.shape) == 2
            b, dims = factor.shape[0], factor.shape[1]
            factor = rearrange(factor, "b d -> (b d)")
            emb = self.timestep(factor)
            emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
            emb = emb[:,:,None,None,None]
            emb = emb.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
            x = torch.cat([x, emb], dim=1)
        
        return x

# -----------------------------------------------------
# This is used for TV2V (text-video-to-video) generation

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad

def lineart_standard(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    x = img.astype(np.float32)
    g = cv2.GaussianBlur(x, (0, 0), 6.0)
    intensity = np.min(g - x, axis=2).clip(0, 255)
    intensity /= max(16, np.median(intensity[intensity > 8]))
    intensity *= 127
    result = intensity.clip(0, 255).astype(np.uint8)
    return remove_pad(result), True

class LineartEncoder(AbstractEmbModel):
    def __init__(self, lineart_coarse=False, lineart_standard=False, *args, **kwargs):
    # def __init__(self, lineart_coarse=False, lineart_standard=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sgm.modules.diffusionmodules.util import LineartDetector
        self.lineart_coarse = lineart_coarse
        self.lineart_standard = lineart_standard
        self.lineart_detector = LineartDetector()
        # freeze the lineart detector
        self.lineart_detector.eval()
        for param in self.lineart_detector.parameters():
            param.requires_grad = False

    def forward(self, x):
        assert x.ndim == 5, "input must be 5D tensor"
        n_frames = x.shape[2]
        x = einops.rearrange(x, 'b c t h w -> (b t) c h w')
        # with torch.no_grad():
        #     x = self.lineart_detector(x, coarse=self.lineart_coarse)    # -1 ~ 1
        if self.lineart_standard:
            b,c,h,w = x.shape
            x_bef = x
            x = einops.rearrange(x, 'b c h w -> b h w c')
            x = (x.cpu().numpy()+1)/2*255.
            x = x.astype(np.uint8)
            xs = [e for e in x]
            for i in range(len(xs)):
                xs[i], _ = lineart_standard(xs[i])
            x = np.stack(xs)
            x = torch.from_numpy(x).cuda()
            x = x.float()/255*2-1
            x = -x
            x = x.unsqueeze(3)
            x = einops.rearrange(x, 'b h w c -> b c h w')
            x = torch.nn.functional.interpolate(
                x,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            # import torchvision
            # torchvision.utils.save_image(x_bef, 'debug_lineart_standard_bef.png', normalize=True)
            # torchvision.utils.save_image(x, 'debug_lineart_standard.png', normalize=True)
        else:
            with torch.no_grad():
                x = self.lineart_detector(x, coarse=self.lineart_coarse)    # -1 ~ 1
        x = einops.rearrange(x, '(b t) c h w -> b c t h w', t=n_frames)
        out_data = einops.repeat(x, 'b c t h w -> b (3 c) t h w')
        return out_data
    
    def encode(self, x):
        return self(x)

import sys
sys.path.append('./src/controlnet11')
# from src.controlnet11.annotator.zoe import ZoeDetector
import os
import cv2
import numpy as np
import torch

from einops import rearrange
from src.controlnet11.annotator.zoe.zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
from src.controlnet11.annotator.zoe.zoedepth.utils.config import get_config
from src.controlnet11.annotator.util import annotator_ckpts_path

class DepthZoeEncoder(AbstractEmbModel):   # TODO: Support more depth encoder
    def __init__(self):
        super().__init__()
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt"
        modelpath = os.path.join(annotator_ckpts_path, "ZoeD_M12_N.pt")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        conf = get_config("zoedepth", "infer")
        model = ZoeDepth.build_from_config(conf)
        model.load_state_dict(torch.load(modelpath)['model'])
        # model = model.cuda()
        # model.device = 'cuda'
        model.eval()
        self.model = model

        for param in self.parameters():
            param.requires_grad = False
        self.model.float()      # this model must use float32, or nan will occur, don't know why

    def forward(self, input_image):
        dtype_ = input_image.dtype
        if self.model.state_dict()['conditional_log_binomial.mlp.2.weight'].dtype != torch.float32:
            print('converting depth model to torch.float32')
            self.model.float()
        # assert input_image.ndim == 3
        assert input_image.ndim == 5, "input must be 5D tensor" # range -1 ~ 1
        n_frames = input_image.shape[2]
        input_image = einops.rearrange(input_image, 'b c t h w -> (b t) c h w')
        input_image = (input_image + 1) / 2 # 0 ~ 1

        image_depth = input_image.float()
        with torch.no_grad():
            depth = self.model.infer(image_depth)
            depth = einops.rearrange(depth, '(b t) c h w -> b c t h w', t=n_frames)
            # TODO: not sure whether conduct on THW or HW
            # calculate the 2nd percentile (vmin) along the CTHW dimension
            percentile_2 = int(0.02 * depth[0].numel())
            vmin = torch.kthvalue(depth.view(depth.shape[0], -1), percentile_2, dim=1).values
            # Calculate the 85th percentile (vmax) along the CTHW dimension
            percentile_85 = int(0.85 * depth[0].numel())
            vmax = torch.kthvalue(depth.view(depth.shape[0], -1), percentile_85, dim=1).values
            
            depth -= vmin[:,None,None,None,None]
            depth /= (vmax - vmin)[:,None,None,None,None]
            depth = torch.clamp(depth, 0, 1)
            depth = depth * 2 - 1   # -1 ~ 1
            depth = einops.repeat(depth, 'b c t h w -> b (3 c) t h w')

            depth = depth.to(dtype_)
            return depth

    def encode(self, x):
        return self(x)


from src.controlnet11.annotator.midas.api import MiDaSInference
class DepthMidasEncoder(AbstractEmbModel):
    def __init__(self):
        super().__init__()
        self.model = MiDaSInference(model_type="dpt_hybrid").cuda()
        for param in self.parameters():
            param.requires_grad = False

    def __call__(self, input_image):
        dtype_ = input_image.dtype
        if self.model.state_dict()['model.pretrained.model.cls_token'].dtype != torch.float32:
            print('converting depthmidas model to torch.float32')
            self.model.float()
        assert input_image.ndim == 5, "input must be 5D tensor" # range -1 ~ 1
        # assert input_image.ndim == 3
        # image_depth = input_image
        n_frames = input_image.shape[2]
        input_image = einops.rearrange(input_image, 'b c t h w -> (b t) c h w')
        # input_image = (input_image + 1) / 2 # 0 ~ 1
        image_depth = input_image.float()
        with torch.no_grad():
            # image_depth = torch.from_numpy(image_depth).float().cuda()
            # image_depth = image_depth / 127.5 - 1.0
            # image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            # depth = self.model(image_depth)[0]
            depth = self.model(image_depth)
            depth = depth.unsqueeze(1)
            # import pdb; pdb.set_trace()
            # import torchvision
            # torchvision.utils.save_image(depth, 'debug.png', normalize=True, nrow=12)

            depth -= torch.min(depth)
            depth /= torch.max(depth)
            # depth = depth.cpu().numpy()
            # depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)

            # return depth_image.to(dtype_)
            depth = torch.clamp(depth, 0, 1)
            depth = depth * 2 - 1   # -1 ~ 1
            depth = - depth
            depth = einops.rearrange(depth, '(b t) c h w -> b c t h w', t=n_frames)
            depth = einops.repeat(depth, 'b c t h w -> b (3 c) t h w')

            depth = depth.to(dtype_)
            return depth
        
    def encode(self, x):
        return self(x)

# Pidinet
# https://github.com/hellozhuo/pidinet

import os
import torch
import numpy as np
from einops import rearrange
from src.controlnet11.annotator.pidinet.model import pidinet
from src.controlnet11.annotator.util import annotator_ckpts_path, safe_step


# class PidiNetDetector(AbstractEmbModel):
class SoftEdgeEncoder(AbstractEmbModel):    # TODO: currently, PidiNet is used to generate soft edge, support more softedge encoder
    def __init__(self):
        super().__init__()
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth"
        modelpath = os.path.join(annotator_ckpts_path, "table5_pidinet.pth")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        self.netNetwork = pidinet()
        self.netNetwork.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(modelpath)['state_dict'].items()})
        # self.netNetwork = self.netNetwork.cuda()
        self.netNetwork.eval()
        for param in self.parameters():
            param.requires_grad = False

    def __call__(self, input_image, safe=False):
        dtype_ = input_image.dtype
        if self.netNetwork.state_dict()['classifier.weight'].dtype != torch.float32:
            print('converting softedge model to torch.float32')
            self.netNetwork.float()
        assert input_image.ndim == 5, "input must be 5D tensor" # range -1 ~ 1
        n_frames = input_image.shape[2]
        input_image = einops.rearrange(input_image, 'b c t h w -> (b t) c h w')
        input_image = (input_image + 1) / 2 # 0 ~ 1
        input_image = input_image[:,[2,1,0],:,:]
        input_image = input_image.float()
        with torch.no_grad():
            edge = self.netNetwork(input_image)[-1]
            if safe:
                edge = safe_step(edge)
            edge = torch.clamp(edge, 0, 1)
            edge = 1 - edge # 
            edge = edge * 2 - 1   # -1 ~ 1
            edge = einops.rearrange(edge, '(b t) c h w -> b c t h w', t=n_frames)
            edge = einops.repeat(edge, 'b c t h w -> b (3 c) t h w')

            edge = edge.to(dtype_)
            return edge

    def encode(self, x):
        return self(x)


# Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal Estimation
# https://github.com/baegwangbin/surface_normal_uncertainty

import os
import types
import torch
import numpy as np

from einops import rearrange
from src.controlnet11.annotator.normalbae.models.NNET import NNET
from src.controlnet11.annotator.normalbae.utils import utils
from src.controlnet11.annotator.util import annotator_ckpts_path
import torchvision.transforms as transforms


# class NormalBaeDetector:
class NormalBaeEncoder(AbstractEmbModel):
    def __init__(self):
        super().__init__()
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/scannet.pt"
        modelpath = os.path.join(annotator_ckpts_path, "scannet.pt")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        args = types.SimpleNamespace()
        args.mode = 'client'
        args.architecture = 'BN'
        args.pretrained = 'scannet'
        args.sampling_ratio = 0.4
        args.importance_ratio = 0.7
        model = NNET(args)
        model = utils.load_checkpoint(modelpath, model)
        model = model.cuda()
        model.eval()
        self.model = model
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        for param in self.parameters():
            param.requires_grad = False

    def __call__(self, input_image):
        # assert input_image.ndim == 3
        # image_normal = input_image
        dtype_ = input_image.dtype
        # TODO: 
        if self.model.state_dict()['decoder.out_conv_res1.6.bias'].dtype != torch.float32:
            print('converting normalbae model to torch.float32')
            self.model.float()
        # assert input_image.ndim == 3
        assert input_image.ndim == 5, "input must be 5D tensor" # range -1 ~ 1
        n_frames = input_image.shape[2]
        input_image = einops.rearrange(input_image, 'b c t h w -> (b t) c h w')
        input_image = (input_image + 1) / 2 # 0 ~ 1
        # image_normal = input_image.float()
        image_normal = input_image.float()
        with torch.no_grad():
            # image_normal = torch.from_numpy(image_normal).float().cuda()
            # image_normal = image_normal / 255.0
            # image_normal = rearrange(image_normal, 'h w c -> 1 c h w')
            # TODO
            image_normal = self.norm(image_normal)
            normal = self.model(image_normal)

            normal = normal[0][-1][:, :3]
            # import torchvision
            # torchvision.utils.save_image(input_image, 'debug_img.png', nrow=12)
            # torchvision.utils.save_image(normal, 'debug_normal.png', normalize=True, nrow=12)
            # d = torch.sum(normal ** 2.0, dim=1, keepdim=True) ** 0.5
            # d = torch.maximum(d, torch.ones_like(d) * 1e-5)
            # normal /= d

            normal = einops.rearrange(normal, '(b t) c h w -> b c t h w', t=n_frames)
            normal = - normal   # todo : not elegant
            normal = torch.clamp(normal, -1, 1)
            # normal = ((normal + 1) * 0.5).clip(0, 1)

            # normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()
            # normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

            # return normal_image
            normal = normal.to(dtype_)
            return normal

    def encode(self, x):
        return self(x)

# Scribble
class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        for i in range(1, layer_number):
            self.convs.append(torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.projection = torch.nn.Conv2d(in_channels=output_channel, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def __call__(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


class ScribbleHEDEncoder(AbstractEmbModel):
    def __init__(self, lineart_coarse=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sgm.modules.diffusionmodules.util import LineartDetector
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth"
        modelpath = os.path.join(annotator_ckpts_path, "ControlNetHED.pth")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        self.netNetwork = ControlNetHED_Apache2().float().cuda().eval()
        self.netNetwork.load_state_dict(torch.load(modelpath))
        for param in self.netNetwork.parameters():
            param.requires_grad = False

    def forward(self, x, safe=False):
        dtype_ = x.dtype
        if self.netNetwork.state_dict()['block1.convs.0.weight'].dtype != torch.float32:
            print('converting softedge model to torch.float32')
            self.netNetwork.float()
        x = x.float()
        assert x.ndim == 5, "input must be 5D tensor" # range -1 ~ 1
        B, C, n_frames, Hh, Ww  = x.shape

        x = einops.rearrange(x, 'b c t h w -> (b t) c h w')
        x = (x + 1) / 2 # 0 ~ 1
        with torch.no_grad():
            edges = self.netNetwork(x)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [cv2.resize(e, (Ww, Hh), interpolation=cv2.INTER_LINEAR) for e in edges]
            import pdb; pdb.set_trace()
            raise NotImplementedError
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))


            edges = [e.detach().float() for e in edges]
            edges = [torch.nn.functional.interpolate(e, size=(Hh, Ww), mode='bilinear', align_corners=False) for e in edges]
            edges = torch.cat(edges, dim=1)

            # TODO: keep on developing, seems bug here
            edge = 1 / (1 + torch.exp(-torch.mean(edges, axis=1).to(torch.float64)))
            if safe:
                edge = safe_step(edge)
            edge = edge.unsqueeze(1)
            import pdb; pdb.set_trace()

            edge = 1 - edge # 
            edge = edge * 2 - 1   # -1 ~ 1
            edge = einops.rearrange(edge, '(b t) c h w -> b c t h w', t=n_frames)
            edge = einops.repeat(edge, 'b c t h w -> b (3 c) t h w')
        return edges.to(dtype_)
    
    def encode(self, x):
        return self(x)


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z

import cv2

class ScribblePidiNetEncoder(AbstractEmbModel):
    def __init__(self):
        super().__init__()
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth"
        modelpath = os.path.join(annotator_ckpts_path, "table5_pidinet.pth")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        self.netNetwork = pidinet()
        self.netNetwork.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(modelpath)['state_dict'].items()})
        self.netNetwork = self.netNetwork.cuda()
        self.netNetwork.eval()
        for param in self.netNetwork.parameters():
            param.requires_grad = False

    def __call__(self, input_image, safe=False):
        dtype_ = input_image.dtype
        if self.netNetwork.state_dict()['block1_1.conv2.weight'].dtype != torch.float32:
            print('converting softedge model to torch.float32')
            self.netNetwork.float()
        input_image = input_image.float()
        assert input_image.ndim == 5, "input must be 5D tensor" # range -1 ~ 1
        B, C, n_frames, H, W  = input_image.shape

        input_image = einops.rearrange(input_image, 'b c t h w -> (b t) c h w')
        input_image = (input_image + 1) / 2 # 0 ~ 1
        # input_image = input_image[:, :, ::-1].copy()
        input_image = input_image[:,[2,1,0],:,:].clone().float()
        with torch.no_grad():
            # image_pidi = torch.from_numpy(input_image).float().cuda()
            # image_pidi = image_pidi / 255.0
            # image_pidi = rearrange(image_pidi, 'h w c -> 1 c h w')
            # edge = self.netNetwork(image_pidi)[-1]
            edge = self.netNetwork(input_image)[-1]

            edge = torch.clamp(edge * 255., 0, 255)
            edge = edge.cpu().squeeze(1).numpy().astype(np.uint8) 
            edge_ = []
            for e in edge:
                e = nms(e, 127, 3.0)
                e = cv2.GaussianBlur(e, (0, 0), 3.0)
                e[e > 4] = 255
                e[e < 255] = 0
                edge_.append(e)
            edge = np.stack(edge_, axis=0)
            edge = torch.from_numpy(edge).float().cuda().unsqueeze(1)
            edge = edge / 255.

            if safe:
                edge = safe_step(edge)

            edge = 1 - edge # 
            edge = edge * 2 - 1   # -1 ~ 1
            edge = einops.rearrange(edge, '(b t) c h w -> b c t h w', t=n_frames)
            edge = einops.repeat(edge, 'b c t h w -> b (3 c) t h w')
        return edge.to(dtype_)
    
    def encode(self, x):
        return self(x)


# openpose 
import src.controlnet11.annotator.util
import src.controlnet11.annotator.openpose 
from src.controlnet11.annotator.openpose.body import Body
from src.controlnet11.annotator.openpose.hand import Hand
from src.controlnet11.annotator.openpose.face import Face
from scipy.ndimage.filters import gaussian_filter
import math
from math import exp

body_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth"
hand_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth"
face_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth"


def draw_pose(pose, H, W, draw_body=True, draw_hand=True, draw_face=True):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if draw_body:
        # canvas = util.draw_bodypose(canvas, candidate, subset)
        canvas = src.controlnet11.annotator.openpose.util.draw_bodypose(canvas, candidate, subset)

    if draw_hand:
        # canvas = util.draw_handpose(canvas, hands)
        canvas = src.controlnet11.annotator.openpose.util.draw_handpose(canvas, hands)

    if draw_face:
        # canvas = util.draw_facepose(canvas, faces)
        canvas = src.controlnet11.annotator.openpose.util.draw_facepose(canvas, faces)

    return canvas


def smart_resize_torch(x, s):
    Ht, Wt = s
    if len(x.shape) == 2:
        Ho, Wo = x.shape
        Co = 1
        x = x.unsqueeze(0).unsqueeze(0)  # Convert to (1, 1, Ho, Wo)
    elif len(x.shape) == 3:
        Ho, Wo, Co = x.shape
        x = x.permute(2, 0, 1).unsqueeze(0)  # Convert to (1, Co, Ho, Wo)
    elif len(x.shape) == 4:
        x = x.permute(0, 3, 1, 2)  # Convert to (B, Co, Ho, Wo)
    else:
        raise ValueError("Unsupported shape for x")

    k = float(Ht + Wt) / float(Ho + Wo)

    mode = 'area' if k < 1 else 'bicubic'
    
    if mode == 'bicubic':
        resized_x = F.interpolate(x, size=(Ht, Wt), mode=mode, align_corners=True)
    else:
        resized_x = F.interpolate(x, size=(Ht, Wt), mode=mode)

    if Co == 1:
        return resized_x.squeeze(0).squeeze(0)
    elif Co == 3:
        return resized_x.squeeze(0).permute(1, 2, 0)
    else:
        return torch.stack([smart_resize_torch(resized_x[0, i], s) for i in range(Co)], dim=2)


def smart_resize_k_torch(x, fx, fy):
    """
    Resize the input tensor `x` using the scaling factors `fx` and `fy`.

    Args:
        x (torch.Tensor): The input tensor to be resized.
        fx (float): The scaling factor for the width dimension.
        fy (float): The scaling factor for the height dimension.

    Returns:
        torch.Tensor: The resized tensor.

    Raises:
        ValueError: If the shape of `x` is not supported.

    """

    if len(x.shape) == 2:
        Ho, Wo = x.shape
        Co = 1
        x = x.unsqueeze(0).unsqueeze(0)  # Convert to (1, 1, Ho, Wo)
    elif len(x.shape) == 3:
        Ho, Wo, Co = x.shape
        x = x.permute(2, 0, 1).unsqueeze(0)  # Convert to (1, Co, Ho, Wo)
    elif len(x.shape) == 4:
        B, Ho, Wo, Co = x.shape
        x = x.permute(0, 3, 1, 2)  # Convert to (B, Co, Ho, Wo)
    else:
        raise ValueError("Unsupported shape for x")

    Ht, Wt = int(Ho * fy), int(Wo * fx)
    k = float(Ht + Wt) / float(Ho + Wo)

    mode = 'area' if k < 1 else 'bicubic'
    
    if mode == 'bicubic':
        resized_x = F.interpolate(x, size=(Ht, Wt), mode=mode, align_corners=True)
    else:
        resized_x = F.interpolate(x, size=(Ht, Wt), mode=mode)

    if Co == 1:
        return resized_x.squeeze(0).squeeze(0)
    elif Co == 3:
        return resized_x.squeeze(0).permute(1, 2, 0)
    else:
        return torch.stack([smart_resize_k_torch(resized_x[0, i], fx, fy) for i in range(Co)], dim=2)


def padRightDownCorner_torch(img, stride, padValue):
    h, w = img.shape[0], img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img.clone()

    if pad[0] > 0:
        pad_up = torch.zeros((pad[0], w, img.shape[2]), dtype=img.dtype, device=img.device) + padValue
        img_padded = torch.cat((pad_up, img_padded), dim=0)

    if pad[1] > 0:
        pad_left = torch.zeros((h + pad[0], pad[1], img.shape[2]), dtype=img.dtype, device=img.device) + padValue
        img_padded = torch.cat((pad_left, img_padded), dim=1)

    if pad[2] > 0:
        pad_down = torch.zeros((pad[2], w + pad[1], img.shape[2]), dtype=img.dtype, device=img.device) + padValue
        img_padded = torch.cat((img_padded, pad_down), dim=0)

    if pad[3] > 0:
        pad_right = torch.zeros((h + pad[0] + pad[2], pad[3], img.shape[2]), dtype=img.dtype, device=img.device) + padValue
        img_padded = torch.cat((img_padded, pad_right), dim=1)

    return img_padded, pad


def gaussian_kernel(size: int, sigma: float):
    kernel = torch.tensor([exp(-(x - size // 2)**2 / float(2 * sigma**2)) for x in range(size)])
    return kernel / kernel.sum()


def apply_gaussian_filter(input, sigma):
    kernel = gaussian_kernel(3 * int(sigma), sigma)
    kernel = kernel.view(1, 1, -1, 1)
    padding = kernel.shape[2] // 2
    
    if len(input.shape) == 2:
        input = input.unsqueeze(0).unsqueeze(0)
    elif len(input.shape) == 3:
        input = input.unsqueeze(0)
    kernel = kernel.to(input.device)
    input = F.conv2d(input, kernel, padding=(padding, padding), stride=(1, 1))
    
    return input.squeeze(0).squeeze(0)


class OpenposeEncoder(AbstractEmbModel):
    def __init__(self):
        super().__init__()
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"   # move to here, not influence other modules

        body_modelpath = os.path.join(annotator_ckpts_path, "body_pose_model.pth")
        hand_modelpath = os.path.join(annotator_ckpts_path, "hand_pose_model.pth")
        face_modelpath = os.path.join(annotator_ckpts_path, "facenet.pth")

        if not os.path.exists(body_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(body_model_path, model_dir=annotator_ckpts_path)

        if not os.path.exists(hand_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(hand_model_path, model_dir=annotator_ckpts_path)

        if not os.path.exists(face_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(face_model_path, model_dir=annotator_ckpts_path)

        self.body_estimation = HackedBody(body_modelpath)
        # TODO 
        self.hand_estimation = Hand(hand_modelpath)
        self.face_estimation = Face(face_modelpath)


        for param in self.body_estimation.model.parameters():
            param.requires_grad = False
        for param in self.hand_estimation.model.parameters():
            param.requires_grad = False
        for param in self.face_estimation.model.parameters():
            param.requires_grad = False

    def __call__(self, oriImg, hand_and_face=False, return_is_index=False):
        dtype_ = oriImg.dtype
        device = oriImg.device
        if self.body_estimation.model.state_dict()['model0.conv1_1.weight'].dtype != torch.float32:
            print('converting body_estimation model to torch.float32')
            self.body_estimation.model.float()
        if self.hand_estimation.model.state_dict()['model1_0.conv1_1.weight'].dtype != torch.float32:
            print('converting hand_estimation model to torch.float32')
            self.hand_estimation.model.float()
        if self.face_estimation.model.state_dict()['conv1_1.weight'].dtype != torch.float32:
            print('converting face_estimation model to torch.float32')
            self.hand_estimation.model.float()
        oriImg = oriImg.float()
        assert oriImg.ndim == 5, "input must be 5D tensor" # range -1 ~ 1
        _, _, n_frames, _, _  = oriImg.shape
        oriImg = einops.rearrange(oriImg, 'b c t h w -> (b t) h w c')
        oriImg = (oriImg + 1) / 2 # 0 ~ 1
        oriImg = oriImg * 255.  # 0 ~ 255
        oriImg = torch.clamp(oriImg, 0, 255)
        # oriImg = oriImg.cpu().numpy().astype(np.uint8)
        
        # oriImg = oriImg[:, :, ::-1].copy()
        oriImgs = torch.flip(oriImg, dims=[3]).clone()
        del oriImg
        B, H, W, C = oriImgs.shape
        poses = []
        with torch.no_grad():
            # TODO: optimize speed
            # TODO: the operation on device change is not elegant.
            for i in range(B):
                oriImg = oriImgs[i]
                with torch.autocast("cuda", enabled=False):
                    candidate, subset = self.body_estimation(oriImg, device)

                hands = []
                faces = []
                if hand_and_face:
                    assert False, "not implemented"
                    # Hand
                    # hands_list = util.handDetect(candidate, subset, oriImg)
                    hands_list = src.controlnet11.annotator.openpose.util.handDetect(candidate, subset, oriImg)
                    for x, y, w, is_left in hands_list:
                        with torch.autocast("cuda", enabled=False):
                            peaks = self.hand_estimation(oriImg[y:y+w, x:x+w, :]).astype(np.float32)
                        if peaks.ndim == 2 and peaks.shape[1] == 2:
                            peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                            peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                            hands.append(peaks.tolist())
                    # Face
                    # faces_list = util.faceDetect(candidate, subset, oriImg)
                    faces_list = src.controlnet11.annotator.openpose.util.faceDetect(candidate, subset, oriImg)
                    for x, y, w in faces_list:
                        with torch.autocast("cuda", enabled=False):
                            heatmaps = self.face_estimation(oriImg[y:y+w, x:x+w, :])
                            peaks = self.face_estimation.compute_peaks_from_heatmaps(heatmaps).astype(np.float32)
                        if peaks.ndim == 2 and peaks.shape[1] == 2:
                            peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                            peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                            faces.append(peaks.tolist())
                if candidate.ndim == 2 and candidate.shape[1] == 4:
                    candidate = candidate[:, :2]
                    candidate[:, 0] /= float(W)
                    candidate[:, 1] /= float(H)
                bodies = dict(candidate=candidate.tolist(), subset=subset.tolist())
                pose = dict(bodies=bodies, hands=hands, faces=faces)
                
                if return_is_index:
                    poses.append(pose)
                else:
                    poses.append(draw_pose(pose, H, W))

        poses = np.stack(poses, axis=0)
        poses = torch.from_numpy(poses).float().to(device)
        poses = poses / 255.    # 0 ~ 1
        poses = 1 - poses  
        poses = poses * 2 - 1   # -1 ~ 1
        # import pdb; pdb.set_trace()
        # import torchvision
        # oriImgs = torch.from_numpy(oriImgs).float().cuda()
        # oriImgs = oriImgs / 255.
        # oriImgs = einops.rearrange(oriImgs, 'bt h w c -> bt c h w')
        # poses_vis = einops.rearrange(poses, 'bt h w c -> bt c h w')
        # torchvision.utils.save_image(oriImgs, 'debug_oriImgs.png', normalize=True, nrow=12)
        # torchvision.utils.save_image(poses_vis, 'debug_poses.png', normalize=True, nrow=12)

        poses = einops.rearrange(poses, '(b t) h w c -> b c t h w', t=n_frames)
        return poses.to(dtype_)

    def encode(self, x):
        return self(x)


class HackedBody(object):
    def __init__(self, model_path):
        from src.controlnet11.annotator.openpose.model import bodypose_model
        # self.model = bodypose_model().cuda()
        self.model = bodypose_model()
        # if torch.cuda.is_available():
        #     self.model = self.model.cuda()
        #     print('cuda')
        model_dict = src.controlnet11.annotator.openpose.util.transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, oriImg, device):
        # import time

        # scale_search = [0.5, 1.0, 1.5, 2.0]
        scale_search = [0.5]
        boxsize = 368
        stride = 8
        padValue = 128
        thre1 = 0.1
        thre2 = 0.05
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        # heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        heatmap_avg = torch.zeros((oriImg.shape[0], oriImg.shape[1], 19)).to(device)
        # paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
        paf_avg = torch.zeros((oriImg.shape[0], oriImg.shape[1], 38)).to(device)

        # t_ = time.time()

        # oriImg = torch.from_numpy(oriImg).float().to(device)
        for m in range(len(multiplier)):
            scale = multiplier[m]
            # time_head = time.time()
            # imageToTest = src.controlnet11.annotator.openpose.util.smart_resize_k(oriImg, fx=scale, fy=scale)
            # imageToTest_padded, pad = src.controlnet11.annotator.openpose.util.padRightDownCorner(imageToTest, stride, padValue)
            # im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            # im = np.ascontiguousarray(im)
            imageToTest = smart_resize_k_torch(oriImg, fx=scale, fy=scale)
            imageToTest_padded, pad = padRightDownCorner_torch(imageToTest, stride, padValue)
            im = imageToTest_padded.unsqueeze(0).permute(0, 3, 1, 2) / 256 - 0.5
            
            # data = torch.from_numpy(im).float()
            data = im
            # if torch.cuda.is_available():
            #     data = data.cuda()
            # print('time_head:{}'.format(time.time() - time_head), end='\t')
            # data = data.permute([2, 0, 1]).unsqueeze(0).float()
            with torch.no_grad():
                # t_model = time.time()
                device = data.device
                self.model = self.model.to(device)
                Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(data)
                # print('time_model_forward:{}'.format(time.time() - t_model), end='\t')
            # time_post = time.time()
            # Mconv7_stage6_L1 = Mconv7_stage6_L1.cpu().numpy()
            # Mconv7_stage6_L2 = Mconv7_stage6_L2.cpu().numpy()

            # extract outputs, resize, and remove padding
            # heatmap = np.transpose(np.squeeze(Mconv7_stage6_L2), (1, 2, 0))  # output 1 is heatmaps
            # heatmap = src.controlnet11.annotator.openpose.util.smart_resize_k(heatmap, fx=stride, fy=stride)
            # heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            # heatmap = src.controlnet11.annotator.openpose.util.smart_resize(heatmap, (oriImg.shape[0], oriImg.shape[1]))
            heatmap = torch.nn.functional.interpolate(Mconv7_stage6_L2, scale_factor=stride, mode='bilinear', align_corners=False)
            heatmap = heatmap[:,:,:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3]]
            heatmap = torch.nn.functional.interpolate(heatmap, size=(oriImg.shape[0], oriImg.shape[1]), mode='bilinear', align_corners=False)
            # heatmap = heatmap.squeeze(0).permute(1, 2, 0).cpu().numpy()
            heatmap = heatmap.squeeze(0).permute(1, 2, 0)

            # paf = np.transpose(np.squeeze(Mconv7_stage6_L1), (1, 2, 0))  # output 0 is PAFs
            # paf = src.controlnet11.annotator.openpose.util.smart_resize_k(paf, fx=stride, fy=stride)
            # paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            # paf = src.controlnet11.annotator.openpose.util.smart_resize(paf, (oriImg.shape[0], oriImg.shape[1]))
            paf = torch.nn.functional.interpolate(Mconv7_stage6_L1, scale_factor=stride, mode='bilinear', align_corners=False)
            paf = paf[:,:,:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3]]
            paf = torch.nn.functional.interpolate(paf, size=(oriImg.shape[0], oriImg.shape[1]), mode='bilinear', align_corners=False)
            # paf = paf.squeeze(0).permute(1, 2, 0).cpu().numpy()
            paf = paf.squeeze(0).permute(1, 2, 0)

            heatmap_avg += heatmap_avg + heatmap / len(multiplier)
            paf_avg += + paf / len(multiplier)
            # print('time_post:{}'.format(time.time() - time_post), end='\t')

        # print('time1:{}'.format(time.time() - t_), end='\t')
        # t_ = time.time()

        all_peaks = []
        peak_counter = 0
        
        # heatmap_avg = torch.from_numpy(heatmap_avg).float().cuda()
        heatmap_avg = heatmap_avg.float()

        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = apply_gaussian_filter(map_ori, sigma=3) 

            map_left = torch.zeros_like(one_heatmap)
            map_left[1:, :] = one_heatmap[:-1, :]
            map_right = torch.zeros_like(one_heatmap)
            map_right[:-1, :] = one_heatmap[1:, :]
            map_up = torch.zeros_like(one_heatmap)
            map_up[:, 1:] = one_heatmap[:, :-1]
            map_down = torch.zeros_like(one_heatmap)
            map_down[:, :-1] = one_heatmap[:, 1:]

            peaks_binary = (one_heatmap >= map_left) & (one_heatmap >= map_right) & \
                        (one_heatmap >= map_up) & (one_heatmap >= map_down) & (one_heatmap > thre1)

            peaks = torch.nonzero(peaks_binary, as_tuple=False)
            peaks[:,0] = torch.clamp(peaks[:,0], 0, heatmap_avg.shape[0]-1)
            peaks[:,1] = torch.clamp(peaks[:,1], 0, heatmap_avg.shape[1]-1)
            peaks_with_score = [(x[1].item(), x[0].item(), map_ori[x[0], x[1]].item()) for x in peaks]
            peak_id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)


        # print('time2:{}'.format(time.time() - t_), end='\t')
        # t_ = time.time()

        # find connection in the specified sequence, center 29 is in the position 15
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                   [1, 16], [16, 18], [3, 17], [6, 18]]
        # the middle joints heatmap correpondence
        mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
                  [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
                  [55, 56], [37, 38], [45, 46]]

        connection_all = []
        special_k = []
        mid_num = 10

        # for k in range(len(mapIdx)):
        #     score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        #     candA = all_peaks[limbSeq[k][0] - 1]
        #     candB = all_peaks[limbSeq[k][1] - 1]
        #     nA = len(candA)
        #     nB = len(candB)
        #     indexA, indexB = limbSeq[k]
        #     if (nA != 0 and nB != 0):
        #         connection_candidate = []
        #         for i in range(nA):
        #             for j in range(nB):
        #                 vec = np.subtract(candB[j][:2], candA[i][:2])
        #                 norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
        #                 norm = max(0.001, norm)
        #                 vec = np.divide(vec, norm)

        #                 startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
        #                                     np.linspace(candA[i][1], candB[j][1], num=mid_num)))

        #                 vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
        #                                   for I in range(len(startend))])
        #                 vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
        #                                   for I in range(len(startend))])

        #                 score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
        #                 score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
        #                     0.5 * oriImg.shape[0] / norm - 1, 0)
        #                 criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
        #                 criterion2 = score_with_dist_prior > 0
        #                 if criterion1 and criterion2:
        #                     connection_candidate.append(
        #                         [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

        #         connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
        #         connection = np.zeros((0, 5))
        #         for c in range(len(connection_candidate)):
        #             i, j, s = connection_candidate[c][0:3]
        #             if (i not in connection[:, 3] and j not in connection[:, 4]):
        #                 connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
        #                 if (len(connection) >= min(nA, nB)):
        #                     break

        #         connection_all.append(connection)
        #     else:
        #         special_k.append(k)
        #         connection_all.append([])

        for k in range(len(mapIdx)):
            score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0] - 1]
            candB = all_peaks[limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if nA != 0 and nB != 0:
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = torch.tensor(candB[j][:2]) - torch.tensor(candA[i][:2])
                        norm = torch.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        norm = max(0.001, norm)
                        vec = vec / norm

                        startend = [torch.linspace(candA[i][0], candB[j][0], steps=mid_num),
                                    torch.linspace(candA[i][1], candB[j][1], steps=mid_num)]

                        vec_x = torch.tensor([score_mid[int(torch.round(startend[1][I])), int(torch.round(startend[0][I])), 0] for I in range(len(startend[0]))])
                        vec_y = torch.tensor([score_mid[int(torch.round(startend[1][I])), int(torch.round(startend[0][I])), 1] for I in range(len(startend[0]))])

                        score_midpts = vec_x * vec[0] + vec_y * vec[1]
                        score_with_dist_prior = torch.sum(score_midpts) / len(score_midpts) + min(
                            0.5 * oriImg.shape[0] / norm - 1, 0)
                        criterion1 = len(torch.nonzero(score_midpts > thre2)) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0

                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = torch.zeros((0, 5))

                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = torch.cat([connection, torch.tensor([[candA[i][3], candB[j][3], s, i, j]])], dim=0)
                        if len(connection) >= min(nA, nB):
                            break

                # connection_all.append(connection)
                connection_all.append(connection.cpu().numpy())
            else:
                special_k.append(k)
                connection_all.append([])


        # print('time3:{}'.format(time.time() - t_), end='\t')
        # t_ = time.time()

        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        # print('time4:{}'.format(time.time() - t_))
        # t_ = time.time()

        # delete some rows of subset which has few parts occur
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

        # subset: n*20 array, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts
        # candidate: x, y, score, id
        return candidate, subset


# Inpainting
class InpaintingEncoder(AbstractEmbModel):
    def __init__(self, mask_ratio_min=0.3, mask_ratio_max=0.5):
        super().__init__()
        assert 0 <= mask_ratio_min < mask_ratio_max <= 0.5
        self.mask_ratio_max = mask_ratio_max
        self.mask_ratio_min = mask_ratio_min

    def __call__(self, x):
        with torch.no_grad():
            with torch.autocast("cuda", enabled=False):
                B,C,T,H,W = x.shape
                mask_ratio_max = self.mask_ratio_max
                mask_ratio_min = self.mask_ratio_min
                mask_ratio_range = mask_ratio_max - mask_ratio_min
                assert mask_ratio_range > 0

                mask_h_start = ((torch.rand(1) * mask_ratio_range + self.mask_ratio_min) * H).int()
                mask_h_end = ((1 - (torch.rand(1) * mask_ratio_range + self.mask_ratio_min)) * H).int()
                mask_w_start = ((torch.rand(1) * mask_ratio_range + self.mask_ratio_min) * W).int()
                mask_w_end = ((1 - (torch.rand(1) * mask_ratio_range + self.mask_ratio_min)) * W).int()

                # mask = torch.ones((B, C, T, H, W)).float().to(x.device)
                # # mask[:, :, :, mask_h_start:mask_h_end, mask_w_start:mask_w_end] = 1
                # mask[:, :, :, mask_h_start:mask_h_end, mask_w_start:mask_w_end] = 0
                # x = (x + 1) / 2     # 0 ~ 1
                # x = x * mask        # 0 ~ 1
                # x = x * 2 - 1       # -1 ~ 1
                # x = -x
                mask = torch.zeros((B, C, T, H, W)).float().to(x.device)
                mask[:, :, :, mask_h_start:mask_h_end, mask_w_start:mask_w_end] = 1
                x = (x + 1) / 2     # 0 ~ 1
                x[mask == 1] = -1   # 0 ~ 1

                x = -x

        return x

    def encode(self, x):
        return self(x)


# Outpainting
class OutpaintingEncoder(AbstractEmbModel):
    def __init__(self, mask_ratio_min=0.0, mask_ratio_max=0.4):
        super().__init__()
        assert 0 <= mask_ratio_min < mask_ratio_max <= 0.5
        self.mask_ratio_max = mask_ratio_max
        self.mask_ratio_min = mask_ratio_min

    def __call__(self, x):
        with torch.no_grad():
            with torch.autocast("cuda", enabled=False):
                B,C,T,H,W = x.shape
                mask_ratio_max = self.mask_ratio_max
                mask_ratio_min = self.mask_ratio_min
                mask_ratio_range = mask_ratio_max - mask_ratio_min
                assert mask_ratio_range > 0

                mask_h_start = ((torch.rand(1) * mask_ratio_range + self.mask_ratio_min) * H).int()
                mask_h_end = ((1 - (torch.rand(1) * mask_ratio_range + self.mask_ratio_min)) * H).int()
                mask_w_start = ((torch.rand(1) * mask_ratio_range + self.mask_ratio_min) * W).int()
                mask_w_end = ((1 - (torch.rand(1) * mask_ratio_range + self.mask_ratio_min)) * W).int()

                mask = torch.zeros((B, C, T, H, W)).float().to(x.device)
                mask[:, :, :, mask_h_start:mask_h_end, mask_w_start:mask_w_end] = 1
                x = (x + 1) / 2     # 0 ~ 1
                x = x * mask        # 0 ~ 1
                x = x * 2 - 1       # -1 ~ 1
                x = -x

        return x

    def encode(self, x):
        return self(x)