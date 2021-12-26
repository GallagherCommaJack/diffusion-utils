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

    def forward(self, q, k, v, v_kq=None):
        if exists(v_kq):
            q, k, v, v_kq = map(self.split_head, (q, k, v, v_kq))
        else:
            q, k, v = map(self.split_head, (q, k, v))

        q, k = map(partial(F.normalize, dim=-1), (q, k))
        sim = einsum("bhid,bhjd->bhij", q, k) * clamp_exp(self.softmax_scale)
        qkv = einsum("bhij,bhjd->bhid", sim.softmax(dim=-1), v)
        qkv = self.unsplit_head(qkv)
        if exists(v_kq):
            vkq = einsum("bhij,bhid->bhjd", sim.softmax(dim=-1), v_kq)
            vkq = self.unsplit_head(vkq)
            return qkv, vkq
        else:
            return qkv


class DuplexAttn(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        d_cond=512,
        norm_fn=LayerNorm,
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        self.attn = ScaledCosineAttention(heads, dim_head)

        self.to_qv = nn.Conv2d(dim, inner_dim * 2, 1)
        self.to_kv = nn.Linear(d_cond, inner_dim * 2)
        self.to_out = SequentialKwargs(
            DropKwargs(
                nn.Conv2d(
                    inner_dim,
                    dim * 2,
                    3,
                    padding=1,
                    bias=False,
                )
            ),
            norm_scales_and_shifts(
                norm_fn,
                dim,
                init_weight=1e-3,
                cond_weight=1e-3,
            ),
        )

        self.to_cond_update = nn.Sequential(
            nn.Linear(inner_dim, d_cond, bias=False),
            nn.LayerNorm(d_cond),
        )
        with torch.no_grad():
            self.to_cond_update[1].weight.fill_(1e-3)

        self.cond_norm = nn.LayerNorm(d_cond)

    def forward(self, x, cond, time_emb=None, **kwargs):
        b, _, h, w = x.shape
        y = x
        cond = self.cond_norm(cond)

        if exists(time_emb):
            time_emb = rearrange(time_emb, "b c -> b c () ()")
            y = y + time_emb

        q, v_q = rearrange(
            self.to_qv(y),
            "b (n c) x y -> n b h (x y) c",
            n=2,
        )
        v_q = rearrange(v_q, "b x y d -> b (x y) d")

        k, v_k = rearrange(
            self.to_kv(cond),
            "b s (n h c) -> n b s c",
            n=2,
        )

        qkv, vkq = self.attn(q, k, v_q, v_k)

        qkv = rearrange(qkv, "b (h w) c -> b c h w", b=b, h=h, w=w)
        scales, shifts = self.to_out(qkv, **kwargs).chunk(2, dim=1)
        out = shifts.addcmul(x, scales)

        cond = cond + self.to_cond_update(vkq)

        return out, cond


class ChannelAttention(nn.Module):
    def __init__(self, dim, heads=8, norm_fn=LayerNorm):
        super().__init__()
        self.heads = heads
        inner_dim = dim
        assert inner_dim % heads == 0
        dim_head = inner_dim // heads
        self.attn = ScaledCosineAttention(heads, dim_head)
        self.proj_in = nn.Sequential(
            nn.Conv2d(dim, inner_dim * 3, 3, padding=1),
            Rearrange("b c x y -> b c (x y)"),
        )
        self.proj_out = SequentialKwargs(
            DropKwargs(
                DepthwiseSeparableConv2d(
                    inner_dim,
                    dim * 2,
                    bias=False,
                )
            ),
            norm_scales_and_shifts(norm_fn, dim, init_weight=1e-3),
        )

    def forward(self, x, time_emb=None, **kwargs):
        b, c, h, w = x.shape
        y = x
        if exists(time_emb):
            y = y + rearrange(time_emb, "b c -> b c () ()")
        q, k, v = self.proj_in(y).chunk(3, dim=1)
        attn = rearrange(self.attn(q, k, v), "b c (h w) -> b c h w", h=h, w=w)
        scales, shifts = self.proj_out(attn, **kwargs).chunk(2, dim=1)
        return shifts.addcmul(x, scales)


class SelfAttention2d(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        skip=False,
        norm_fn=LayerNorm,
    ):
        super().__init__()
        self.attn = ScaledCosineAttention(heads, dim_head, split_head=False)
        inner_dim = dim_head * heads

        self.proj_in = nn.Sequential(
            nn.Conv2d(dim, inner_dim * 3, 1),
            Rearrange("b (h c) x y -> b h x y c", h=heads),
        )
        if skip:
            self.proj_skip = nn.Sequential(
                nn.Conv2d(dim, inner_dim * 2, 1),
                Rearrange("b (h c) x y -> b h x y c", h=heads),
            )
        else:
            self.proj_skip = None

        self.proj_out = SequentialKwargs(
            DropKwargs(
                DepthwiseSeparableConv2d(
                    inner_dim,
                    dim * 2,
                    bias=False,
                )
            ),
            norm_scales_and_shifts(norm_fn, dim, init_weight=1e-3),
        )

        self.heads = heads

    def forward(self, x, time_emb=None, pos_emb=None, skip=None, **kwargs):
        b, c, h, w = x.shape
        y = x
        if exists(time_emb):
            time_emb = rearrange(time_emb, "b c -> b c () ()")
            y = y + time_emb

        q, k, v = self.proj_in(y).chunk(3, dim=-1)

        use_skip = exists(skip) and exists(self.proj_skip)
        if use_skip:
            k_skip, v_skip = self.proj_skip(skip).chunk(2, dim=-1)
            k = torch.cat([k, k_skip], dim=0)
            v = torch.cat([v, v_skip], dim=0)

        if exists(pos_emb):
            q, k = apply_rotary_emb(q, k, pos_emb)

        q = rearrange(q, "b h x y c -> b h (x y) c")
        r = 2 if use_skip else 1
        k, v = map(
            partial(
                rearrange,
                pattern="(r b) h x y c -> b h (r x y) c",
                r=r,
            ),
            (k, v),
        )

        update = self.attn(q, k, v)
        update = rearrange(update, "b (h w) c -> b c h w", h=h, w=w)
        scales, shifts = self.proj_out(update, **kwargs).chunk(2, dim=1)

        return shifts.addcmul(x, scales)


class LocalAttn2d(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        window_size=16,
        skip=False,
        norm_fn=LayerNorm,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.window_size = window_size
        inner_dim = dim_head * heads

        self.proj_in = nn.Sequential(
            nn.Conv2d(dim, inner_dim * 3, 1),
            Rearrange("b (h c) x y -> b h x y c", h=heads),
        )
        if skip:
            self.proj_skip = nn.Sequential(
                nn.Conv2d(dim, inner_dim * 2, 1),
                Rearrange("b (h c) x y -> b h x y c", h=heads),
            )
        self.attn = ScaledCosineAttention(heads, dim_head, split_head=False)
        self.proj_out = SequentialKwargs(
            DropKwargs(
                DepthwiseSeparableConv2d(
                    inner_dim,
                    dim * 2,
                    bias=False,
                )
            ),
            norm_scales_and_shifts(norm_fn, dim, init_weight=1e-3),
        )

    def forward(self, x, skip=None, time_emb=None, pos_emb=None, **kwargs):
        h, w, b = self.heads, self.window_size, x.shape[0]
        y = x

        if exists(time_emb):
            time_emb = rearrange(time_emb, "b c -> b c () ()")
            y = y + time_emb

        q, k, v = self.proj_in(y).chunk(3, dim=-1)

        if exists(skip):
            k_skip, v_skip = self.proj_skip(skip).chunk(2, dim=-1)
            k = torch.cat([k, k_skip], dim=0)
            v = torch.cat([v, v_skip], dim=0)

        if exists(pos_emb):
            q, k = apply_rotary_emb(q, k, pos_emb)

        # chunk into patches
        q, k, v = map(
            lambda t: rearrange(
                t, "b h (x w1) (y w2) c -> (b x y) h (w1 w2) c", w1=w, w2=w
            ),
            (q, k, v),
        )

        # cat skip to sequences
        if exists(skip):
            k, v = map(
                lambda t: rearrange(t, "(r b) h n d -> b h (r n) d", r=2), (k, v)
            )

        update = self.attn(q, k, v)
        update = rearrange(
            update,
            "(b x y) (w1 w2) c -> b c (x w1) (y w2)",
            b=b,
            y=y.shape[-1] // w,
            w1=w,
            w2=w,
        )
        scales, shifts = self.proj_out(update, **kwargs).chunk(2, dim=1)
        return shifts.addcmul(y, scales)
