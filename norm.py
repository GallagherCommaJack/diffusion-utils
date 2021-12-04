import torch
from torch import nn
import torch.nn.functional as F

from utils import mk_full, SequentialKwargs, exists


class ClassConditionalLayerNorm2d(nn.Module):
    def __init__(
        self,
        dim,
        n_classes,
        groups=1,
        cond_weight=1e-1,
        init_weight=1,
        init_bias=0,
        **kwargs,
    ):
        super().__init__()
        self.emb = nn.Embedding(n_classes, dim * 2)
        with torch.no_grad():
            self.emb.weight.mul_(cond_weight)
            self.emb.weight.add_(
                torch.cat([
                    mk_full(dim, init_weight),
                    mk_full(dim, init_bias),
                ]))
        self.groups = groups

    def forward(self, x, cond_class, **kwargs):
        scales, shifts = self.emb(cond_class)[:, :, None, None].chunk(2, dim=1)
        return shifts.addcmul(F.group_norm(x, self.groups), scales)


class ConditionalLayerNorm2d(nn.Module):
    def __init__(
        self,
        dim,
        d_cond,
        groups=1,
        cond_weight=1e-1,
        init_weight=1,
        init_bias=0,
        **kwargs,
    ):
        super().__init__()
        self.proj = nn.Linear(d_cond, dim * 2)
        self.groups = groups
        with torch.no_grad():
            self.proj.weight *= cond_weight
            self.proj.bias.copy_(
                torch.cat([
                    mk_full(dim, init_weight),
                    mk_full(dim, init_bias),
                ]))

    def forward(self, x, cond_emb, **kwargs):
        scales, shifts = self.proj(cond_emb)[:, :, None, None].chunk(2, dim=1)
        return torch.addcmul(shifts, F.group_norm(x, self.groups), scales)


class LayerNorm(nn.Module):
    def __init__(
        self,
        dim,
        groups=1,
        init_weight=1,
        init_bias=0,
        **kwargs,
    ):
        super().__init__()
        self.groups = groups
        self.weight = nn.Parameter(mk_full(dim, init_weight))
        self.bias = nn.Parameter(mk_full(dim, init_bias))

    def forward(self, x, **kwargs):
        return F.group_norm(x, self.groups, self.weight, self.bias)


def pre_norm(dim, fn, wrapper=None, norm_fn=LayerNorm):
    norm = norm_fn(dim)
    out = SequentialKwargs(norm, fn)
    if exists(wrapper):
        out = wrapper(fn, offload_to_cpu=True)
    return out


def norm_scales_and_shifts(norm_fn, dim, **kwargs):
    return norm_fn(
        dim * 2,
        groups=2,
        init_bias=torch.cat([
            torch.ones(dim),
            torch.zeros(dim),
        ]),
        **kwargs,
    )
