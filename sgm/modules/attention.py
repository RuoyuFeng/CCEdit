import math
from inspect import isfunction
from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from packaging import version
from torch import nn

import loralib as lora

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = {}
    print(
        f"No SDP backend available, likely because you are running in pytorch versions < 2.0. In fact, "
        f"you are using PyTorch {torch.__version__}. You might want to consider upgrading."
    )

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    print("no module 'xformers'. Processing without...")

from .diffusionmodules.util import checkpoint

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
    from flash_attn.bert_padding import unpad_input

    use_flash_attention = True
except ImportError:
    try:
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_func as flash_attn_unpadded_func,
        )
        from flash_attn.bert_padding import unpad_input

        use_flash_attention = True
    except ImportError:
        flash_attn_unpadded_func = None
        unpad_input = None
        use_flash_attention = False
        print("Not use flash Attention")


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

def get_lora_params(kwargs):
    lora_names = ["q", "k", "v", "o"]
    lora_params = dict()
    for lora_name in lora_names:
        lora_use = lora_name + "_use_lora"
        lora_r = lora_name + "_lora_r"
        lora_alpha = lora_name + "_lora_alpha"
        lora_params[lora_use] = kwargs.get(lora_use, False)
        lora_params[lora_r] = kwargs.get(lora_r, 4)
        lora_params[lora_alpha] = kwargs.get(lora_alpha, 1)
    return lora_params    


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = torch.einsum("bij,bjk->bik", q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = self.proj_out(h_)

        return x + h_


class FlashCrossAttention(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs
    ):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads with a dimension of {dim_head}."
        )
        self.dropout = dropout
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.heads = heads
        self.dim_head = dim_head

        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5

        lora_params = get_lora_params(kwargs)

        if lora_params["q_use_lora"]:
            self.to_q = lora.Linear(
                query_dim, inner_dim, r=lora_params["q_lora_r"], bias=False
            )
        else:
            self.to_q = nn.Linear(query_dim, inner_dim, bias=False)

        if lora_params["k_use_lora"]:
            self.to_k = lora.Linear(
                context_dim, inner_dim, r=lora_params["k_lora_r"], bias=False
            )
        else:
            self.to_k = nn.Linear(context_dim, inner_dim, bias=False)

        if lora_params["v_use_lora"]:
            self.to_v = lora.Linear(
                context_dim, inner_dim, r=lora_params["v_lora_r"], bias=False
            )
        else:
            self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        if lora_params["o_use_lora"]:
            self.to_out = nn.Sequential(
                lora.Linear(inner_dim, query_dim, r=lora_params["o_lora_r"]),
                nn.Dropout(dropout),
            )
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
            )

    def get_input(self, x, seqlen, batch_size, nheads, mask=None):
        assert mask is None, "not implemented for mask with flash attention"
        lengths = torch.ones([batch_size, 1], dtype=torch.int, device="cuda") * seqlen
        attention_mask_bool = (
            repeat(torch.arange(seqlen, device="cuda"), "s -> b s", b=batch_size)
            < lengths
        )
        attention_mask = torch.zeros(
            batch_size, seqlen, device="cuda", dtype=torch.float16
        )
        attention_mask[~attention_mask_bool] = -10000.0
        attention_mask = rearrange(attention_mask, "b s -> b 1 1 s")
        x_unpad, indices, cu_seqlens_x, max_seqlen_in_batch_x = unpad_input(
            x, attention_mask_bool
        )
        x_unpad = rearrange(x_unpad, "nnz (h d) -> nnz h d", h=nheads)
        return x_unpad.to(torch.float16), cu_seqlens_x, max_seqlen_in_batch_x

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

        b, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]
        seqlen_v = v.shape[1]
        q, cu_seqlens_q, max_seqlen_in_batch_q = self.get_input(q, seqlen_q, b, h)
        k, cu_seqlens_k, max_seqlen_in_batch_k = self.get_input(k, seqlen_k, b, h)
        v, cu_seqlens_v, max_seqlen_in_batch_v = self.get_input(v, seqlen_v, b, h)

        if self.training:
            dropout_p = self.dropout
        else:
            dropout_p = 0

        out = flash_attn_unpadded_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_in_batch_q,
            max_seqlen_in_batch_k,
            dropout_p,
        )

        out = rearrange(out, "(b n) h d -> b n (h d)", b=b, h=h)
        out = out.to(context.dtype)
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        out = self.to_out(out)
        return out


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
        **kwargs,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

        self.backend = backend

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        h = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        q = self.to_q(x)
        context = default(context, x)
        context = context.to(self.to_k.weight.dtype)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # old
        """
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        """
        # new
        with sdp_kernel(**BACKEND_MAP[self.backend]):
            # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )  # scale is dim_head ** -0.5 per default
            # if self.to_q.weight.dtype == torch.float16:
            #     q, k, v = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
            # elif self.to_q.weight.dtype == torch.bfloat16:
            #     q, k, v = (
            #         q.to(torch.bfloat16),
            #         k.to(torch.bfloat16),
            #         v.to(torch.bfloat16),
            #     )
            # out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).to(
            #     self.to_q.weight.dtype
            # )

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs
    ):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads with a dimension of {dim_head}."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        lora_params = get_lora_params(kwargs)

        if lora_params["q_use_lora"]:
            self.to_q = lora.Linear(
                query_dim, inner_dim, r=lora_params["q_lora_r"], bias=False
            )
        else:
            self.to_q = nn.Linear(query_dim, inner_dim, bias=False)

        if lora_params["k_use_lora"]:
            self.to_k = lora.Linear(
                context_dim, inner_dim, r=lora_params["k_lora_r"], bias=False
            )
        else:
            self.to_k = nn.Linear(context_dim, inner_dim, bias=False)

        if lora_params["v_use_lora"]:
            self.to_v = lora.Linear(
                context_dim, inner_dim, r=lora_params["v_lora_r"], bias=False
            )
        else:
            self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        if lora_params["o_use_lora"]:
            self.to_out = nn.Sequential(
                lora.Linear(inner_dim, query_dim, r=lora_params["o_lora_r"]),
                nn.Dropout(dropout),
            )
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
            )

        self.attention_op: Optional[Any] = None

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)
        q = self.to_q(x)
        context = default(context, x)
        context = context.to(self.to_k.weight.dtype)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        # out = xformers.ops.memory_efficient_attention(
        #     q, k, v, attn_bias=None, op=self.attention_op
        # )

        with torch.autocast(enabled=False, device_type="cuda"):
            if self.to_q.weight.dtype == torch.float16:
                q, k, v = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
            elif self.to_q.weight.dtype == torch.bfloat16:
                q, k, v = (
                    q.to(torch.bfloat16),
                    k.to(torch.bfloat16),
                    v.to(torch.bfloat16),
                )
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False).to(
                self.to_q.weight.dtype
            )

        # TODO: Use this directly in the attention operation, as a bias
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "flash": FlashCrossAttention,  # flash attention
        "softmax-xformers": MemoryEfficientCrossAttention,  # ampere
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        flash_attention=False,
        attn_mode="softmax",
        sdp_backend=None,
        **kwargs,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if use_flash_attention and flash_attention:
            attn_mode = "flash"
        else:
            if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
                print(
                    f"Attention mode '{attn_mode}' is not available. Falling back to native attention. "
                    f"This is not a problem in Pytorch >= 2.0. FYI, you are running with PyTorch version {torch.__version__}"
                )
                attn_mode = "softmax"
            elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
                print(
                    "We do not support vanilla attention anymore, as it is too expensive. Sorry."
                )
                if not XFORMERS_IS_AVAILABLE:
                    assert (
                        False
                    ), "Please install xformers via e.g. 'pip install xformers==0.0.16'"
                else:
                    print("Falling back to xformers efficient attention.")
                    attn_mode = "softmax-xformers"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            backend=sdp_backend,
            **kwargs,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,
            **kwargs,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        if self.checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        kwargs = {"x": x}

        if context is not None:
            kwargs.update({"context": context})

        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})

        if n_times_crossframe_attn_in_self:
            kwargs.update(
                {"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self}
            )

        # return mixed_checkpoint(self._forward, kwargs, self.parameters(), self.checkpoint)
        return checkpoint(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    def _forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                if not self.disable_self_attn
                else 0,
            )
            + x
        )
        x = (
            self.attn2(
                self.norm2(x), context=context, additional_tokens=additional_tokens
            )
            + x
        )
        x = self.ff(self.norm3(x)) + x
        return x


class BasicTransformerSingleLayerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        # on the A100s not quite as fast as the above version
        "softmax-xformers": MemoryEfficientCrossAttention
        # (todo might depend on head_dim, check, falls back to semi-optimized kernels for dim!=[16,32,64,128])
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        attn_mode="softmax-xformers",
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context) + x
        x = self.ff(self.norm2(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        attn_type="softmax-xformers",
        use_checkpoint=True,
        sdp_backend=None,
        **kwargs,
    ):
        super().__init__()
        print(
            f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads"
        )
        from omegaconf import ListConfig

        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                print(
                    f"WARNING: {self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        disable_text_ca = kwargs.get("disable_text_ca", False)
        self.disable_text_ca = disable_text_ca
        if disable_text_ca:
            self.transformer_blocks = nn.ModuleList(
                [
                    # BasicTransformerBlock(  # temporal transformer does not use flash attention
                    BasicTransformerSingleLayerBlock(  # temporal transformer does not use flash attention
                        inner_dim,
                        n_heads,
                        d_head,
                        dropout=dropout,
                        # context_dim=context_dim[d],
                        context_dim=None,
                        attn_mode="softmax",
                        checkpoint=use_checkpoint,
                    )
                    for d in range(depth)
                ]
            )
        else:
            self.transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        inner_dim,
                        n_heads,
                        d_head,
                        dropout=dropout,
                        context_dim=context_dim[d],
                        disable_self_attn=disable_self_attn,
                        attn_mode=attn_type,
                        checkpoint=use_checkpoint,
                        sdp_backend=sdp_backend,
                    )
                    for d in range(depth)
                ]
            )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            # self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            if self.disable_text_ca:
                x = block(x, context=x)
            else:
                x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class SpatialTransformerCA(SpatialTransformer):
    """
    This is hacked from SpatialTransformer.
    Conduct additional cross-attention with k,v from the reference feature.
    Thus, the attention order is text cross-attention -> sptial self-attention -> reference cross-attention.
    Note that if the reference feature is not given, this module is equivalent to SpatialTransformer.


    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        attn_type="softmax-xformers",
        use_checkpoint=True,
        sdp_backend=None,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth,
            dropout,
            context_dim,
            disable_self_attn,
            use_linear,
            attn_type,
            use_checkpoint,
            sdp_backend,
            **kwargs,
        )
        inner_dim = n_heads * d_head

        # temporal crossattention part
        self.norm_ca = Normalize(in_channels)
        if not use_linear:
            self.proj_in_ca = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in_ca = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks_ca = nn.ModuleList(
            [
                BasicTransformerSingleLayerBlock( 
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=None,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out_ca = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out_ca = zero_module(nn.Linear(inner_dim, in_channels))

    def forward(self, x, context=None):
        x = super().forward(x, context)

        assert hasattr(self, 'ref_control'), "must have ref_control"
        ref_control = self.ref_control

        b, c, h, w = x.shape
        # cross-frame attention
        # x = rearrange(x, "b c t h w -> (b t) c h w").contiguous()
        x_in = x
        x = self.norm_ca(x)
        if not self.use_linear:
            x = self.proj_in_ca(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in_ca(x)

        for i, block in enumerate(self.transformer_blocks_ca):
            ref_control = rearrange(ref_control, "b c h w -> b (h w) c").contiguous()
            context_texture = ref_control
            x = block(x, context_texture)

        if self.use_linear:
            x = self.proj_out_ca(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out_ca(x)
        x = x + x_in

        return x


class SpatialTransformer3D(nn.Module):
    """
    This is hacked from the 2D version above.

    Transformer block for video-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        attn_type="softmax-xformers",
        use_checkpoint=True,
        sdp_backend=None,
        **kwargs,
    ):
        super().__init__()
        print(
            f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads"
        )
        from omegaconf import ListConfig

        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                print(
                    f"WARNING: {self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                    flash_attention=True,
                    **kwargs,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

        # temporal part
        self.norm_temporal = Normalize(in_channels)
        if not use_linear:
            self.proj_in_temporal = zero_module(
                nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_in_temporal = zero_module(nn.Linear(in_channels, inner_dim))
        disable_temporal_text_ca = kwargs.get("disable_temporal_text_ca", False)
        self.disable_temporal_text_ca = disable_temporal_text_ca
        if disable_temporal_text_ca:
            self.transformer_blocks_temporal = nn.ModuleList(
                [
                    # BasicTransformerBlock(  # temporal transformer does not use flash attention
                    BasicTransformerSingleLayerBlock(  # temporal transformer does not use flash attention
                        inner_dim,
                        n_heads,
                        d_head,
                        dropout=dropout,
                        # context_dim=context_dim[d],
                        context_dim=None,
                        attn_mode="softmax",
                        checkpoint=use_checkpoint,
                    )
                    for d in range(depth)
                ]
            )
        else:
            self.transformer_blocks_temporal = nn.ModuleList(
                [
                    BasicTransformerBlock(  # temporal transformer does not use flash attention
                        inner_dim,
                        n_heads,
                        d_head,
                        dropout=dropout,
                        context_dim=context_dim[d],
                        disable_self_attn=disable_self_attn,
                        attn_mode="softmax",
                        checkpoint=use_checkpoint,
                        sdp_backend=sdp_backend,
                    )
                    for d in range(depth)
                ]
            )
        if not use_linear:
            self.proj_out_temporal = zero_module(
                nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out_temporal = zero_module(nn.Linear(inner_dim, in_channels))

        use_learnable_alpha = kwargs.get("use_learnable_alpha", False)
        if use_learnable_alpha:
            self.alpha_temporal = nn.Parameter(
                torch.ones(1)
            )  # x = alpha * spatial + (1-alpha) * temporal

    def forward(self, x, context=None):
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
            # x = x_in    # ! DEBUG ONLY

        x = rearrange(x, "(b h w) c t -> b c t h w", h=h, w=w).contiguous()
        return x


class SpatialTransformer3DCA(SpatialTransformer3D):
    """
    # -> SpatialTransformer3DCrossAttention
    # Replace the second temporal attention in SpatialTransformer3D with cross-attention
    # Original attention order:
    # 1. spatial self-attention
    # 2. cross-attention with text condition
    # 3. temporal self-attention (1d)
    # 4. cross-attention with text condition
    # Attention order: 
    # 1. spatial self-attention
    # 2. cross-attention with text condition
    # 3. temporal self-attention (1d)
    # 4. cross-attention with text condition (maybe not necessary, but... nevermind)
    # 5. cross-attention with anchor frame (usually center frame, or reference image from outside)

    This is hacked from the 2D version above.

    Transformer block for video-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        attn_type="softmax-xformers",
        use_checkpoint=True,
        sdp_backend=None,
        **kwargs,
    ):
        # super().__init__(**kwargs)
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            context_dim=context_dim,
            disable_self_attn=disable_self_attn,
            use_linear=use_linear,
            attn_type=attn_type,
            use_checkpoint=use_checkpoint,
            sdp_backend=sdp_backend,
            **kwargs,
        )

        inner_dim = n_heads * d_head

        # temporal crossattention part
        self.norm_temporal_ca = Normalize(in_channels)
        if not use_linear:
            self.proj_in_temporal_ca = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in_temporal_ca = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks_temporal_ca = nn.ModuleList(
            [
                BasicTransformerSingleLayerBlock( 
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=None,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out_temporal_ca = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out_temporal_ca = zero_module(nn.Linear(inner_dim, in_channels))

        self.ST3DCA_ca_type = kwargs.get("ST3DCA_ca_type", "center")
        assert self.ST3DCA_ca_type in ["center", 'self', 'center_self']

    def forward(self, x, context=None):
        x = super().forward(x, context)

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
            attn_anchor_frame_idx = t // 2  # center frame
            anchor_frame = x[:, attn_anchor_frame_idx, :, :].contiguous()
            anchor_frame = repeat(anchor_frame, "b hw c -> b t hw c", t=t).contiguous()
            anchor_frame = rearrange(anchor_frame, "b t hw c -> (b t) hw c").contiguous()
            context_texture = anchor_frame
            x = rearrange(x, "b t hw c -> (b t) hw c", b=b).contiguous()
            if self.ST3DCA_ca_type == 'center':
                x = block(x, context=context_texture)
            elif self.ST3DCA_ca_type == 'self':
                x = block(x, context=x)
            elif self.ST3DCA_ca_type == 'center_self':
                context_texture = torch.cat([context_texture, x], dim=1)
                x = block(x, context=context_texture)
            else:
                raise NotImplementedError
            # x = block(x, context_texture)

        if self.use_linear:
            x = self.proj_out_temporal_ca(x)
        x = rearrange(x, "bt (h w) c -> bt c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out_temporal_ca(x)
        x = x + x_in

        x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t).contiguous()

        return x


def benchmark_attn():
    # Lets define a helpful benchmarking function:
    # https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import torch.nn.functional as F
    import torch.utils.benchmark as benchmark

    def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
        t0 = benchmark.Timer(
            stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
        )
        return t0.blocked_autorange().mean * 1e6

    # Lets define the hyper-parameters of our input
    batch_size = 32
    max_sequence_len = 1024
    num_heads = 32
    embed_dimension = 32

    dtype = torch.float16

    query = torch.rand(
        batch_size,
        num_heads,
        max_sequence_len,
        embed_dimension,
        device=device,
        dtype=dtype,
    )
    key = torch.rand(
        batch_size,
        num_heads,
        max_sequence_len,
        embed_dimension,
        device=device,
        dtype=dtype,
    )
    value = torch.rand(
        batch_size,
        num_heads,
        max_sequence_len,
        embed_dimension,
        device=device,
        dtype=dtype,
    )

    print(f"q/k/v shape:", query.shape, key.shape, value.shape)

    # Lets explore the speed of each of the 3 implementations
    from torch.backends.cuda import SDPBackend, sdp_kernel

    # Helpful arguments mapper
    backend_map = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
    }

    from torch.profiler import ProfilerActivity, profile, record_function

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    print(
        f"The default implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds"
    )
    with profile(
        activities=activities, record_shapes=False, profile_memory=True
    ) as prof:
        with record_function("Default detailed stats"):
            for _ in range(25):
                o = F.scaled_dot_product_attention(query, key, value)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print(
        f"The math implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds"
    )
    with sdp_kernel(**backend_map[SDPBackend.MATH]):
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("Math implmentation stats"):
                for _ in range(25):
                    o = F.scaled_dot_product_attention(query, key, value)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]):
        try:
            print(
                f"The flash attention implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds"
            )
        except RuntimeError:
            print("FlashAttention is not supported. See warnings for reasons.")
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("FlashAttention stats"):
                for _ in range(25):
                    o = F.scaled_dot_product_attention(query, key, value)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
        try:
            print(
                f"The memory efficient implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds"
            )
        except RuntimeError:
            print("EfficientAttention is not supported. See warnings for reasons.")
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("EfficientAttention stats"):
                for _ in range(25):
                    o = F.scaled_dot_product_attention(query, key, value)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def run_model(model, x, context):
    return model(x, context)


def benchmark_transformer_blocks():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import torch.utils.benchmark as benchmark

    def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
        t0 = benchmark.Timer(
            stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
        )
        return t0.blocked_autorange().mean * 1e6

    checkpoint = True
    compile = False

    batch_size = 32
    h, w = 64, 64
    context_len = 1024
    embed_dimension = 1024
    context_dim = 1024
    d_head = 64

    transformer_depth = 4

    n_heads = embed_dimension // d_head

    dtype = torch.float16

    model_native = SpatialTransformer(
        embed_dimension,
        n_heads,
        d_head,
        context_dim=context_dim,
        use_linear=True,
        use_checkpoint=checkpoint,
        attn_type="softmax",
        depth=transformer_depth,
        sdp_backend=SDPBackend.FLASH_ATTENTION,
    ).to(device)
    model_efficient_attn = SpatialTransformer(
        embed_dimension,
        n_heads,
        d_head,
        context_dim=context_dim,
        use_linear=True,
        depth=transformer_depth,
        use_checkpoint=checkpoint,
        attn_type="softmax-xformers",
    ).to(device)
    if not checkpoint and compile:
        print("compiling models")
        model_native = torch.compile(model_native)
        model_efficient_attn = torch.compile(model_efficient_attn)

    x = torch.rand(batch_size, embed_dimension, h, w, device=device, dtype=dtype)
    c = torch.rand(batch_size, context_len, context_dim, device=device, dtype=dtype)

    from torch.profiler import ProfilerActivity, profile, record_function

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with torch.autocast("cuda"):
        print(
            f"The native model runs in {benchmark_torch_function_in_microseconds(model_native.forward, x, c):.3f} microseconds"
        )
        print(
            f"The efficientattn model runs in {benchmark_torch_function_in_microseconds(model_efficient_attn.forward, x, c):.3f} microseconds"
        )

        print(75 * "+")
        print("NATIVE")
        print(75 * "+")
        torch.cuda.reset_peak_memory_stats()
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("NativeAttention stats"):
                for _ in range(25):
                    model_native(x, c)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(torch.cuda.max_memory_allocated() * 1e-9, "GB used by native block")

        print(75 * "+")
        print("Xformers")
        print(75 * "+")
        torch.cuda.reset_peak_memory_stats()
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("xformers stats"):
                for _ in range(25):
                    model_efficient_attn(x, c)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(torch.cuda.max_memory_allocated() * 1e-9, "GB used by xformers block")


def test01():
    # conv1x1 vs linear
    from ..util import count_params

    conv = nn.Conv2d(3, 32, kernel_size=1).cuda()
    print(count_params(conv))
    linear = torch.nn.Linear(3, 32).cuda()
    print(count_params(linear))

    print(conv.weight.shape)

    # use same initialization
    linear.weight = torch.nn.Parameter(conv.weight.squeeze(-1).squeeze(-1))
    linear.bias = torch.nn.Parameter(conv.bias)

    print(linear.weight.shape)

    x = torch.randn(11, 3, 64, 64).cuda()

    xr = rearrange(x, "b c h w -> b (h w) c").contiguous()
    print(xr.shape)
    out_linear = linear(xr)
    print(out_linear.mean(), out_linear.shape)

    out_conv = conv(x)
    print(out_conv.mean(), out_conv.shape)
    print("done with test01.\n")


def test02():
    # try cosine flash attention
    import time

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("testing cosine flash attention...")
    DIM = 1024
    SEQLEN = 4096
    BS = 16

    print(" softmax (vanilla) first...")
    model = BasicTransformerBlock(
        dim=DIM,
        n_heads=16,
        d_head=64,
        dropout=0.0,
        context_dim=None,
        attn_mode="softmax",
    ).cuda()
    try:
        x = torch.randn(BS, SEQLEN, DIM).cuda()
        tic = time.time()
        y = model(x)
        toc = time.time()
        print(y.shape, toc - tic)
    except RuntimeError as e:
        # likely oom
        print(str(e))

    print("\n now softmax-xformer ...")
    model = BasicTransformerBlock(
        dim=DIM,
        n_heads=16,
        d_head=64,
        dropout=0.0,
        context_dim=None,
        attn_mode="softmax-xformers",
    ).cuda()
    x = torch.randn(BS, SEQLEN, DIM).cuda()
    tic = time.time()
    y = model(x)
    toc = time.time()
    print(y.shape, toc - tic)
    print("done with test02.\n")


if __name__ == "__main__":
    test01()
    test02()

    benchmark_attn()
    # benchmark_transformer_blocks()

    print("done.")
