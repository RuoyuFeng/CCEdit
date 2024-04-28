from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union

import einops
import pytorch_lightning as pl
import torch
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR

from sgm.modules.encoders.modules import VAEEmbedder
from sgm.modules.encoders.modules import (
    LineartEncoder,
    DepthZoeEncoder,
    DepthMidasEncoder,
    SoftEdgeEncoder,
    NormalBaeEncoder,
    ScribbleHEDEncoder,
    ScribblePidiNetEncoder,
    OpenposeEncoder,
    OutpaintingEncoder,
    InpaintingEncoder,
)

import os
import numpy as np
import torch.nn as nn

from ..modules import UNCONDITIONAL_CONFIG
from ..modules.diffusionmodules.wrappers import (
    OPENAIUNETWRAPPER,
    OPENAIUNETWRAPPERRAIG,
    OPENAIUNETWRAPPERCONTROLLDM3D,
    OPENAIUNETWRAPPERCONTROLLDM3DSSN,
    OPENAIUNETWRAPPERCONTROLLDM3DTV2V,
    OPENAIUNETWRAPPERCONTROLLDM3DTV2V_INTERPOLATE,
)
from ..modules.ema import LitEma
from ..util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
)

class DiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
    ):
        super().__init__()
        self.log_keys = log_keys
        self.input_key = input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        model = instantiate_from_config(network_config)
        wrapper_type = (
            self.wrapper_type if hasattr(self, "wrapper_type") else OPENAIUNETWRAPPER
        )
        self.model = get_obj_from_str(default(network_wrapper, wrapper_type))(
            model, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        print(f"Loading checkpoint from {path} ... ")
        if path.endswith("ckpt"):
            # sd = torch.load(path, map_location="cpu")["state_dict"]
            if "deepspeed" in path:
                sd = torch.load(path, map_location="cpu")
                sd = {k.replace("_forward_module.", ""): v for k, v in sd.items()}
            else:
                sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            out = self.first_stage_model.decode(z)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            z = self.first_stage_model.encode(x)
        z = self.scale_factor * z
        return z

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        x = self.encode_first_stage(x)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)    

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        # xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                        # xc = log_txt_as_img((image_w * 2, image_h), x, size=image_h // 15)
                        xc = log_txt_as_img(
                            (image_w * 2, image_h), x, size=image_h // 25
                        )
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        log["inputs"] = x
        z = self.encode_first_stage(x)
        log["reconstructions"] = self.decode_first_stage(z)
        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)
            log["samples"] = samples
        return log


class VideoDiffusionEngine(DiffusionEngine):
    def __init__(
        self,
        freeze_model="none",
        wrapper_type="OPENAIUNETWRAPPERCONTROLLDM3D",
        *args,
        **kwargs,
    ):
        self.wrapper_type = eval(wrapper_type)
        super().__init__(*args, **kwargs)
        self.freeze_model = freeze_model

        self.setup_vaeembedder()

    def setup_vaeembedder(self):
        for embedder in self.conditioner.embedders:
            if isinstance(embedder, VAEEmbedder):
                embedder.first_stage_model = (
                    self.first_stage_model
                )  # TODO: should we add .clone()
                embedder.disable_first_stage_autocast = (
                    self.disable_first_stage_autocast
                )
                embedder.scale_factor = self.scale_factor
                embedder.freeze()

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # video tensors should be scaled to -1 ... 1 and in bcthw format
        out_data = batch[self.input_key]
        return out_data

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        x = self.encode_first_stage(x)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[-2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        # xc = log_txt_as_img((image_w, image_h), x, size=image_h // 20)
                        xc = log_txt_as_img(
                            (image_w, image_h), x, size=image_h // 10, split_loc=15
                        )
                        # xc = log_txt_as_img((image_w * 2, image_h), x, size=image_h // 15, split_loc=20)
                        # xc = log_txt_as_img(
                        #     (image_w * 3, image_h), x, size=image_h // 5, split_loc=15
                        # )
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        # TODO: refactor this
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        sampling_kwargs = {
            key: batch[key] for key in self.loss_fn.batch2model_keys.intersection(batch)
        }

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        log["inputs"] = x
        log["inputs-video"] = x
        log["cond_img"] = batch["cond_img"]
        z = self.encode_first_stage(x)
        log["reconstructions"] = self.decode_first_stage(z)
        log["reconstructions-video"] = self.decode_first_stage(z)
        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)
            log["samples"] = samples
            log["samples-video"] = samples

        # concat the inputs and outputs for visualization
        log["inputs_samples"] = torch.cat([log["inputs"], log["samples"]], dim=3)
        del log["inputs"]
        del log["samples"]
        # log['inputs_samples-video'] = torch.cat([log['inputs-video'], log['samples-video']], dim=3)
        # del log['inputs-video']
        # del log['samples-video']
        return log

    def configure_optimizers(self):
        lr = self.learning_rate

        if self.freeze_model == "none":
            params = list(self.model.diffusion_model.parameters())
            for name, param in self.model.diffusion_model.named_parameters():
                print(f"Setting {name} to trainable")
                param.requires_grad = True  # TODO: why this?
        elif self.freeze_model == "spatial":
            params = []
            if hasattr(self.model.diffusion_model, "controlnet"):
                params += list(self.model.diffusion_model.controlnet.parameters())
            for name, param in self.model.diffusion_model.named_parameters():
                if "controlnet" not in name:
                    if "temporal" in name:
                        params.append(param)
                    else:
                        param.requires_grad = False
        elif self.freeze_model == "spatial_openlora":
            params = []
            if hasattr(self.model.diffusion_model, "controlnet"):
                params += list(self.model.diffusion_model.controlnet.parameters())
            for name, param in self.model.diffusion_model.named_parameters():
                if "controlnet" not in name:
                    if "temporal" in name or "lora" in name:
                        params.append(param)
                    else:
                        param.requires_grad = False
        else:
            raise NotImplementedError

        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

# -----------------------------------------------------
class VideoDiffusionEngineTV2V(VideoDiffusionEngine):
    def __init__(self, *args, **kwargs):
        # kwargs['wrapper_type'] = OPENAIUNETWRAPPERCONTROLLDM3DTV2V
        kwargs["wrapper_type"] = kwargs.get(
            "wrapper_type", "OPENAIUNETWRAPPERCONTROLLDM3DTV2V"
        )
        super().__init__(*args, **kwargs)

        # freeze the controlnet (load pre-trained weights, no need to train)
        self.model.diffusion_model.controlnet.eval()
        for name, param in self.model.diffusion_model.controlnet.named_parameters():
            param.requires_grad = False

        if hasattr(self.model.diffusion_model, "controlnet_img"):
            print('Setting controlnet_img to trainable ... ')
            # open the controlnet_img
            for (
                name,
                param,
            ) in self.model.diffusion_model.controlnet_img.named_parameters():
                param.requires_grad = True

    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        print(f"Loading checkpoint from {path} ... ")
        if path.endswith("ckpt"):
            if "deepspeed" in path:
                sd = torch.load(path, map_location="cpu")
                sd = {k.replace("_forward_module.", ""): v for k, v in sd.items()}
            else:
                sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)

        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        negative_prompt = "ugly, low quality"
        batch_uc = {
            "txt": [negative_prompt for i in range(x.shape[0])],
            "control_hint": batch[
                "control_hint"
            ].clone(),  # to use the pretrained weights, we must use the same control_hint in the batch_uc
        }
        if "cond_img" in batch.keys():  # for TVI2V;
            # TODO: in fact, we can delete this, just use empty tensor as cond_img for batch_uc
            batch_uc["cond_img"] = batch["cond_img"].clone()  
            # batch_uc['cond_img'] = torch.zeros_like(batch['cond_img'])
        batch["txt"] = ["masterpiece, best quality, " + each for each in batch["txt"]]
        c, uc = self.conditioner.get_unconditional_conditioning(
            batch_c=batch,
            batch_uc=batch_uc,
        )

        sampling_kwargs = {
            key: batch[key] for key in self.loss_fn.batch2model_keys.intersection(batch)
        }

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        log["inputs"] = x
        log["inputs-video"] = x
        if "cond_img" in batch.keys():
            log["cond_img"] = batch["cond_img"]
        z = self.encode_first_stage(x)
        # log["reconstructions"] = self.decode_first_stage(z)
        # log["reconstructions-video"] = self.decode_first_stage(z)
        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)
            log["samples"] = samples
            log["samples-video"] = samples

        for embedder in self.conditioner.embedders:
            if (
                isinstance(embedder, LineartEncoder)
                or isinstance(embedder, DepthZoeEncoder)
                or isinstance(embedder, DepthMidasEncoder)
                or isinstance(embedder, SoftEdgeEncoder)
                or isinstance(embedder, NormalBaeEncoder)
                or isinstance(embedder, ScribbleHEDEncoder)
                or isinstance(embedder, ScribblePidiNetEncoder)
                or isinstance(embedder, OpenposeEncoder)
                or isinstance(embedder, OutpaintingEncoder)
                or isinstance(embedder, InpaintingEncoder)
            ):
                # log['control_hint'] = embedder.encode(batch['control_hint'])
                # log['control_hint-video'] = embedder.encode(batch['control_hint'])
                log["control_hint"] = -embedder.encode(batch["control_hint"])
                log["control_hint-video"] = -embedder.encode(batch["control_hint"])
                break

        # concat the inputs and outputs for visualization
        log["inputs_samples_hint"] = torch.cat(
            [log["inputs"], log["samples"], log["control_hint"]], dim=3
        )
        del log["inputs"]
        del log["samples"]
        del log["control_hint"]

        log["inputs_samples_hint-video"] = torch.cat(
            [log["inputs-video"], log["samples-video"], log["control_hint-video"]],
            dim=3,
        )
        del log["inputs-video"]
        del log["samples-video"]
        del log["control_hint-video"]
        return log

    def configure_optimizers(self):
        lr = self.learning_rate

        if self.freeze_model == "none":
            params = list(self.model.diffusion_model.parameters())
            for name, param in self.model.diffusion_model.named_parameters():
                print(f"Setting {name} to trainable")
                param.requires_grad = True
        elif self.freeze_model == "spatial":
            params = []
            if hasattr(self.model.diffusion_model, "controlnet"):
                params += list(self.model.diffusion_model.controlnet.parameters())
            if hasattr(self.model.diffusion_model, "controlnet_img"):
                params += list(self.model.diffusion_model.controlnet_img.parameters())
            for name, param in self.model.diffusion_model.named_parameters():
                if "controlnet" not in name:
                    if "temporal" in name:
                        params.append(param)
                    else:
                        param.requires_grad = False
        else:
            raise NotImplementedError

        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt


class VideoDiffusionEngineTV2VInterpolate(VideoDiffusionEngineTV2V):
    def __init__(self, *args, **kwargs):
        kwargs["wrapper_type"] = "OPENAIUNETWRAPPERCONTROLLDM3DTV2V_INTERPOLATE"
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        # negative_prompt = "ugly, low quality"
        negative_prompt = ''
        batch_uc = {
            "txt": [negative_prompt for i in range(x.shape[0])],
            "control_hint": batch["control_hint"].clone(),
            "interpolate_first_last": batch["interpolate_first_last"].clone(),
        }
        # TODO: specify this in the config file
        batch["txt"] = ['' for each in batch["txt"]]  # disbale text prompt
        # batch["txt"] = ["masterpiece, best quality, " + each for each in batch["txt"]]
        # batch['txt'] = ['masterpiece, best quality' for each in batch['txt']]  # disbale text prompt

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch_c=batch,
            batch_uc=batch_uc,
        )

        sampling_kwargs = {
            key: batch[key] for key in self.loss_fn.batch2model_keys.intersection(batch)
        }

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        log["inputs"] = x
        log["inputs-video"] = x
        # log['interpolate_first_last'] = torch.cat([batch['interpolate_first'], batch['interpolate_last']], dim=2)
        from sgm.modules.encoders.modules import CustomIdentityEncoder, CustomIdentityDownCondEncoder

        for embedder in self.conditioner.embedders:
            if isinstance(embedder, CustomIdentityEncoder) or isinstance(embedder, CustomIdentityDownCondEncoder):
                log["interpolate_first_last"] = embedder.encode(batch["interpolate_first_last"])[:,:3,:,...]    # in case of more than 3
                break
        z = self.encode_first_stage(x)
        # log["reconstructions"] = self.decode_first_stage(z)
        # log["reconstructions-video"] = self.decode_first_stage(z)
        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)
            log["samples"] = samples
            log["samples-video"] = samples

        for embedder in self.conditioner.embedders:
            if (
                isinstance(embedder, LineartEncoder)
                or isinstance(embedder, DepthZoeEncoder)
                or isinstance(embedder, DepthMidasEncoder)
                or isinstance(embedder, SoftEdgeEncoder)
                or isinstance(embedder, NormalBaeEncoder)
                or isinstance(embedder, ScribbleHEDEncoder)
                or isinstance(embedder, ScribblePidiNetEncoder)
                or isinstance(embedder, OpenposeEncoder)
                or isinstance(embedder, OutpaintingEncoder)
                or isinstance(embedder, InpaintingEncoder)
            ):
                log["control_hint"] = -embedder.encode(batch["control_hint"])
                log["control_hint-video"] = -embedder.encode(batch["control_hint"])
                break

        # concat the inputs and outputs for visualization
        log["inputs_samples_hint"] = torch.cat(
            [log["inputs"], log["samples"], log["control_hint"]], dim=3
        )
        del log["inputs"]
        del log["samples"]
        del log["control_hint"]

        log["inputs_samples_hint-video"] = torch.cat(
            [log["inputs-video"], log["samples-video"], log["control_hint-video"]],
            dim=3,
        )
        del log["inputs-video"]
        del log["samples-video"]
        del log["control_hint-video"]
        return log


if __name__ == "__main__":
    import logging

    import yaml

    open("output.log", "w").close()

    logging.basicConfig(
        level=logging.DEBUG,
        filename="output.log",
        datefmt="%Y/%m/%d %H:%M:%S",
        format="%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    # logger.info('This is a log info')
    # logger.debug('Debugging')
    # logger.warning('Warning exists')
    # logger.info('Finish')

    BS = 2
    frame_length = 17
    # size = [BS, frame_length, 3, 320, 320]
    size = [BS, 3, 320, 320]
    batch = {
        "jpg": torch.randn(size).cuda(),
        "txt": BS * ["text"],
        "original_size_as_tuple": torch.tensor([320, 320]).repeat(BS, 1).cuda(),
        "crop_coords_top_left": torch.tensor([0, 0]).repeat(BS, 1).cuda(),
        "target_size_as_tuple": torch.tensor([320, 320]).repeat(BS, 1).cuda(),
    }

    model_config = yaml.load(
        open("configs/example_training/sd_xl_base-test.yaml"), Loader=yaml.Loader
    )["model"]

    learning_rate = model_config.pop("base_learning_rate")
    model = DiffusionEngine(**model_config["params"]).cuda()
    model.learning_rate = learning_rate
    logger.info(model)

    opt = model.configure_optimizers()

    while True:
        # out = model.shared_step(batch)
        loss = model.training_step(batch, 1)
        print(f"loss: {loss}")
        loss.backward()
        opt[0][0].step()
        opt[0][0].zero_grad()
