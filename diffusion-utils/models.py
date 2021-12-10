import math
from math import log, pi, sqrt
from functools import partial

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from utils import *
from attn import *
from ff import *
from norm import *
from pos_emb import *


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        time_emb_dim: Optional[int] = None,
        rotary_emb: bool = True,
        cross_attn: bool = False,
        local_self_attn: bool = True,
        d_cond: int = 512,
        skip: bool = False,
        norm_fn: NormFnType = LayerNorm,
    ):
        super().__init__()

        self.attn_time_emb = None
        self.ff_time_emb = None
        d_hidden = int(dim * ff_mult)
        if exists(time_emb_dim):
            self.attn_time_emb = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, dim),
            )
            self.ff_time_emb = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, d_hidden),
            )

        self.pos_emb = AxialRotaryEmbedding(dim_head) if rotary_emb else None

        self.attns = nn.ModuleList([])
        self.ffs = nn.ModuleList([])
        if cross_attn:
            self.cross = nn.ModuleList([])
        else:
            self.cross = None

        def mk_attn():
            if local_self_attn:
                attn = ChannelAttention(
                    dim,
                    heads=heads,
                    norm_fn=norm_fn,
                )
            else:
                attn = SelfAttention2d(
                    dim,
                    dim_head=dim_head,
                    heads=heads,
                    skip=skip,
                    norm_fn=norm_fn,
                )
            return pre_norm(dim, attn, norm_fn=norm_fn)

        for _ in range(depth):
            self.attns.append(mk_attn())
            self.ffs.append(
                pre_norm(
                    dim,
                    FeedForward(dim, mult=ff_mult, norm_fn=norm_fn),
                    norm_fn=norm_fn,
                ))
            if cross_attn:
                self.cross.append(
                    pre_norm(
                        dim,
                        DuplexAttn(
                            dim,
                            dim_head=dim_head,
                            heads=heads,
                            d_cond=d_cond,
                            norm_fn=norm_fn,
                        ),
                        norm_fn=norm_fn,
                    ))

    def forward(
        self,
        x: Tensor,
        time: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
        skip: Optional[Tensor] = None,
        *args,
        **kwargs,
    ):
        attn_time_emb = None
        ff_time_emb = None
        if exists(time):
            assert exists(self.attn_time_emb) and exists(
                self.ff_time_emb
            ), 'time_emb_dim must be given on init if you are conditioning based on time'
            attn_time_emb = self.attn_time_emb(time)
            ff_time_emb = self.ff_time_emb(time)

        pos_emb = None
        if exists(self.pos_emb):
            pos_emb = self.pos_emb(x)

        if exists(cond) and exists(self.cross):
            for attn, ff, cross in zip(self.attns, self.ffs, self.cross):
                x = attn(
                    x,
                    skip=skip,
                    time_emb=attn_time_emb,
                    pos_emb=pos_emb,
                    **kwargs,
                )
                x = ff(x, time_emb=ff_time_emb, **kwargs)
                x, cond = cross(x, cond=cond, time_emb=attn_time_emb, **kwargs)
        else:
            for attn, ff in zip(self.attns, self.ffs):
                x = attn(
                    x,
                    skip=skip,
                    time_emb=attn_time_emb,
                    pos_emb=pos_emb,
                    **kwargs,
                )
                x = ff(x, time_emb=ff_time_emb, **kwargs)

        return x
