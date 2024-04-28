"""
adopted from
https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
and
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
and
https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py

thanks!
"""

import math
import os
import numpy as np

import torch
import torch.nn as nn
from einops import repeat, rearrange

from torch.utils.checkpoint import checkpoint as cp

import deepspeed

def make_beta_schedule(
    schedule,
    n_timestep,
    linear_start=1e-4,
    linear_end=2e-2,
):
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )
    return betas.numpy()


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def mixed_checkpoint(func, inputs: dict, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass. This differs from the original checkpoint function
    borrowed from https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py in that
    it also works with non-tensor inputs
    :param func: the function to evaluate.
    :param inputs: the argument dictionary to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        tensor_keys = [key for key in inputs if isinstance(inputs[key], torch.Tensor)]
        tensor_inputs = [
            inputs[key] for key in inputs if isinstance(inputs[key], torch.Tensor)
        ]
        non_tensor_keys = [
            key for key in inputs if not isinstance(inputs[key], torch.Tensor)
        ]
        non_tensor_inputs = [
            inputs[key] for key in inputs if not isinstance(inputs[key], torch.Tensor)
        ]
        args = tuple(tensor_inputs) + tuple(non_tensor_inputs) + tuple(params)
        return MixedCheckpointFunction.apply(
            func,
            len(tensor_inputs),
            len(non_tensor_inputs),
            tensor_keys,
            non_tensor_keys,
            *args,
        )
    else:
        return func(**inputs)


class MixedCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        run_function,
        length_tensors,
        length_non_tensors,
        tensor_keys,
        non_tensor_keys,
        *args,
    ):
        ctx.end_tensors = length_tensors
        ctx.end_non_tensors = length_tensors + length_non_tensors
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        assert (
            len(tensor_keys) == length_tensors
            and len(non_tensor_keys) == length_non_tensors
        )

        ctx.input_tensors = {
            key: val for (key, val) in zip(tensor_keys, list(args[: ctx.end_tensors]))
        }
        ctx.input_non_tensors = {
            key: val
            for (key, val) in zip(
                non_tensor_keys, list(args[ctx.end_tensors : ctx.end_non_tensors])
            )
        }
        ctx.run_function = run_function
        ctx.input_params = list(args[ctx.end_non_tensors :])

        with torch.no_grad():
            output_tensors = ctx.run_function(
                **ctx.input_tensors, **ctx.input_non_tensors
            )
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        # additional_args = {key: ctx.input_tensors[key] for key in ctx.input_tensors if not isinstance(ctx.input_tensors[key],torch.Tensor)}
        ctx.input_tensors = {
            key: ctx.input_tensors[key].detach().requires_grad_(True)
            for key in ctx.input_tensors
        }

        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = {
                key: ctx.input_tensors[key].view_as(ctx.input_tensors[key])
                for key in ctx.input_tensors
            }
            # shallow_copies.update(additional_args)
            output_tensors = ctx.run_function(**shallow_copies, **ctx.input_non_tensors)
        input_grads = torch.autograd.grad(
            output_tensors,
            list(ctx.input_tensors.values()) + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (
            (None, None, None, None, None)
            + input_grads[: ctx.end_tensors]
            + (None,) * (ctx.end_non_tensors - ctx.end_tensors)
            + input_grads[ctx.end_tensors :]
        )


def checkpoint_new(func, input, flag=False):
    """
    Custom checkpoint function
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        return cp(func, *input)
    else:
        return func(*input)


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


# def checkpoint_new(func, input, flag):
#     """
#     Custom checkpoint function.
#     Evaluate a function without caching intermediate activations, allowing for
#     reduced memory at the expense of extra compute in the backward pass.
#     :param func: the function to evaluate.
#     :param input: the argument sequence to pass to `func`.
#     :param flag: if False, disable gradient checkpointing.
#     """
#     if flag:
#         return deepspeed.checkpointing.checkpoint(func, *input)
#     else:
#         return func(*input)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        # ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors if x is not None]
        # Ensure all tensors have requires_grad set to True
        ctx.input_params = [p.requires_grad_(True) for p in ctx.input_params]
        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return nn.GroupNorm(32, channels)


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.to(torch.float32)).to(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

# ---------------------------------------------------
# This is used for the generation of lineart maps
# From https://github.com/carolineec/informative-drawings
annotator_ckpts_path = os.path.join(os.path.dirname(__file__), 'ckpts')

norm_layer = nn.InstanceNorm2d

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features)
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out


class LineartDetector(nn.Module):   
    # hacked from controlnet1.1, find differences from the official repo
    def __init__(self):
        super(LineartDetector, self).__init__()
        self.model = self.load_model('sk_model.pth')
        self.model_coarse = self.load_model('sk_model2.pth')

    def load_model(self, name):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/" + name
        modelpath = os.path.join(annotator_ckpts_path, name)
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        model = Generator(3, 1, 3)
        model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
        model.eval()
        # model = model.cuda()
        return model

    def forward(self, input_image, coarse): 
        model = self.model_coarse if coarse else self.model
        # if numpy
        if isinstance(input_image, np.ndarray):
            assert input_image.ndim == 3
            image = input_image
            with torch.no_grad():
                image = torch.from_numpy(image).float().cuda()
                image = image / 255.0
                image = rearrange(image, 'h w c -> 1 c h w')
                line = model(image)[0][0]

                line = line.cpu().numpy()
                line = (line * 255.0).clip(0, 255).astype(np.uint8)

                return line
        # or tensor
        elif isinstance(input_image, torch.Tensor):
            assert input_image.ndim == 4
            image = input_image
            with torch.no_grad():
                image = (image + 1) / 2.0   # 0 ~ 1
                line = model(image)
                line = line * 2.0 - 1.0
                line = line.clip(-1, 1)
                return line             # b c h w
        else:
            raise ValueError('input_image should be numpy or tensor')