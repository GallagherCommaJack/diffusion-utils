import math
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange  # type: ignore
from einops.layers.torch import Rearrange  # type: ignore

from diffusion_utils.norm import *
from diffusion_utils.pos_emb import *
from diffusion_utils.utils import *
from diffusion_utils.ff import *


class ScaledCosineAttention(nn.Module):
    def __init__(self, n_head, d_head, scale=None, split_head=True):
        super().__init__()
        scale = default(scale, math.log(d_head ** -0.5))
        self.softmax_scale = nn.Parameter(torch.full([n_head, 1, 1], scale))
        self.split_head = (
            partial(
                rearrange,
                pattern="b s (h d) -> b h s d",
                h=n_head,
            )
            if split_head
            else lambda x: x
        )
        self.unsplit_head = partial(rearrange, pattern="b h s d -> b s (h d)")

    def forward(self, q, k, v, v_kq=None, mask=None):
        if exists(v_kq):
            q, k, v, v_kq = map(self.split_head, (q, k, v, v_kq))
        else:
            q, k, v = map(self.split_head, (q, k, v))

        q, k = map(partial(F.normalize, dim=-1), (q, k))
        sim = einsum("bhid,bhjd->bhij", q, k) * clamp_exp(self.softmax_scale)
        if exists(mask):
            sim.masked_fill_(mask, max_neg_value(sim))
        qkv = einsum("bhij,bhjd->bhid", sim.softmax(dim=-1), v)
        qkv = self.unsplit_head(qkv)
        if exists(v_kq):
            vkq = einsum("bhij,bhid->bhjd", sim.softmax(dim=-1), v_kq)
            vkq = self.unsplit_head(vkq)
            return qkv, vkq
        else:
            return qkv


class AttnType(Protocol):
    def __init__(self, dim, d_out=None):
        pass


class ChannelAttention2d(nn.Module):
    def __init__(self, dim, d_out=None):
        heads, dim_head = factor_int(dim)
        d_out = d_out if d_out else dim
        self.attn = ScaledCosineAttention(heads, dim_head)
        self.proj_in = nn.Sequential(
            nn.Conv2d(dim, dim * 3, 3, padding=1),
            Rearrange("b c x y -> b c (x y)"),
        )
        self.proj_out = nn.Conv2d(dim, d_out, 3, bias=False)

    def forward(self, y, **kwargs):
        b, c, h, w = y.shape
        q, k, v = self.proj_in(y).chunk(3, dim=1)
        attn = rearrange(self.attn(q, k, v), "b c (h w) -> b c h w", h=h, w=w)
        return self.proj_out(attn)


class SelfAttention2d(nn.Module):
    def __init__(self, dim, d_out=None):
        d_out = d_out if d_out else dim
        heads, dim_head = factor_int(d_out)
        self.attn = ScaledCosineAttention(heads, dim_head)
        self.proj_in = nn.Sequential(
            nn.Conv2d(dim, d_out * 3, 1, padding=1),
            Rearrange("b c x y -> b (x y) c"),
        )
        self.proj_out = nn.Conv2d(d_out, d_out, 1, bias=False)

    def forward(self, y, **kwargs):
        b, c, h, w = y.shape
        q, k, v = self.proj_in(y).chunk(3, dim=-1)
        attn = rearrange(self.attn(q, k, v), "b (h w) c -> b c h w", h=h, w=w)
        return self.proj_out(attn)


class CrossAttn2d(nn.Module):
    def __init__(self, dim, d_out, d_cond=None, d_cross=None):
        super().__init__()
        d_cond = d_cond if d_cond else dim
        d_cross = d_cross if d_cross else d_cond
        n_head, d_head = factor_int(d_out)
        assert d_cond % n_head == 0
        # only a scale instead of an nn.LayerNorm bc we should prenormalize
        # only a scale instead of a ScaleShift bc we then project it
        self.affine_scales = nn.Parameter(torch.ones([1, d_cross]))
        self.q_proj = nn.Sequential(
            nn.Conv2d(dim, d_out, 1),
            Rearrange("b c h w -> b (h w) c", n_h=n_head),
        )
        self.kv_proj = nn.Linear(d_cross, d_out * 2)
        self.attn = ScaledCosineAttention(n_head, d_head)
        self.out_proj = nn.Conv2d(d_out, d_out, 1)

    def forward(self, x, global_cond, cond, cond_mask, **kwargs):
        n, c, h, w = x.shape
        q = self.q_proj(x)
        k, v = self.kv_proj(self.affine_scales * cond).chunk(2, dim=1)
        attn = self.attn(q, k, v, mask=cond_mask[:, None, None, :])
        attn = rearrange(attn, "b (h w) c -> b c h w", h=h, w=w)
        return self.out_proj(attn)


class Layer(SandwichNorm):
    def __init__(
        self,
        dim: int,
        inner: AttnType = SelfAttention2d,
        norm_fn: NormFnType = LayerNorm,
        scale_shift: bool = False,
        init_weight: float = 1e-3,
    ):
        super().__init__(dim, inner, norm_fn, scale_shift, init_weight)
