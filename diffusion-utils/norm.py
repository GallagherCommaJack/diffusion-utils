from typing import Callable, Optional, Union
from typing_extensions import Protocol

import torch
from torch import nn
import torch.nn.functional as F
from torch.types import Number

from utils import mk_full, SequentialKwargs, exists

param_type = Union[torch.Tensor, float]


class NormFnType(Protocol):
    def __call__(
        self,
        dim: int,
        groups: int = 1,
        init_weight: param_type = 1.0,
        init_bias: param_type = 0.0,
        **kwargs,
    ):
        pass


class ClassConditionalLayerNorm2d(nn.Module):
    def __init__(
        self,
        n_classes: int,
        dim: int,
        groups: int = 1,
        init_weight: param_type = 1.0,
        init_bias: param_type = 0.0,
        cond_weight: param_type = 1e-1,
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

    def forward(self, x, classes, **kwargs):
        scales, shifts = self.emb(classes)[:, :, None, None].chunk(2, dim=1)
        return shifts.addcmul(F.group_norm(x, self.groups), scales)


class ConditionalLayerNorm2d(nn.Module):
    def __init__(
        self,
        d_cond: int,
        dim: int,
        groups: int = 1,
        init_weight: param_type = 1.0,
        init_bias: param_type = 0.0,
        cond_weight: param_type = 1e-1,
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

    def forward(self, x, global_cond, **kwargs):
        scales, shifts = self.proj(global_cond)[:, :, None, None].chunk(2,
                                                                        dim=1)
        return torch.addcmul(shifts, F.group_norm(x, self.groups), scales)


class LayerNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        groups: int = 1,
        init_weight: param_type = 1.0,
        init_bias: param_type = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.groups = groups
        self.weight = nn.Parameter(mk_full(dim, init_weight))
        self.bias = nn.Parameter(mk_full(dim, init_bias))

    def forward(self, x, **kwargs):
        return F.group_norm(x, self.groups, self.weight, self.bias)


def pre_norm(
    dim,
    fn: nn.Module,
    wrapper: Optional[Callable[[nn.Module], nn.Module]] = None,
    norm_fn: NormFnType = LayerNorm,
) -> nn.Module:
    norm = norm_fn(dim)
    out = SequentialKwargs(norm, fn)
    if exists(wrapper):
        out = wrapper(
            out
        )  # type: ignore # mypy thinks wrapper isn't callable and out is not a module!
    return out


def norm_scales_and_shifts(
    norm_fn: NormFnType,
    dim: int,
    **kwargs,
) -> nn.Module:
    assert 'groups' not in kwargs
    assert 'dim' not in kwargs
    assert 'init_bias' not in kwargs
    return norm_fn(
        dim * 2,
        groups=2,
        init_bias=torch.cat([
            torch.ones(dim),
            torch.zeros(dim),
        ]),
        **kwargs,
    )
