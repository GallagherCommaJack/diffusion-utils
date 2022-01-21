from typing import Callable, Optional, Union, Dict, Any
from typing_extensions import Protocol

import torch
from torch import nn
import torch.nn.functional as F
from torch.types import Number

from diffusion_utils.utils import mk_full, SequentialKwargs, exists

param_type = Union[torch.Tensor, float]


class ScaleShift(nn.Module):
    def __init__(
        self,
        dim: int,
        init_weight: param_type = 1.0,
        init_bias: param_type = 0.0,
    ):
        super().__init__()
        scales = mk_full(dim, init_weight)
        shifts = mk_full(dim, init_bias)
        self.scales_and_shifts = torch.stack([scales, shifts])

    def forward(self, x):
        scales, shifts = self.scales_and_shifts
        return shifts.addcmul(scales, x)


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
                torch.cat(
                    [
                        mk_full(dim, init_weight),
                        mk_full(dim, init_bias),
                    ]
                )
            )
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
                torch.cat(
                    [
                        mk_full(dim, init_weight),
                        mk_full(dim, init_bias),
                    ]
                )
            )

    def forward(self, x, global_cond, **kwargs):
        scales, shifts = self.proj(global_cond)[:, :, None, None].chunk(2, dim=1)
        return torch.addcmul(shifts, F.group_norm(x, self.groups), scales)


class LayerNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        groups: int = 1,
        init_weight: param_type = 1.0,
        init_bias: param_type = 0.0,
        use_affine: bool = True,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.groups = groups
        self.weight = nn.Parameter(mk_full(dim, init_weight)) if use_affine else None
        self.bias = (
            nn.Parameter(mk_full(dim, init_bias)) if use_affine and use_bias else None
        )

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
    assert "groups" not in kwargs
    assert "dim" not in kwargs
    assert "init_bias" not in kwargs
    return norm_fn(
        dim * 2,
        groups=2,
        init_bias=torch.cat(
            [
                torch.ones(dim),
                torch.zeros(dim),
            ]
        ),
        **kwargs,
    )


class SandwichNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        inner,
        norm_fn: NormFnType,
        scale_shift: bool = False,
        init_weight: float = 1e-3,
        inner_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()
        d_out = dim * 2 if scale_shift else dim
        self.scale_shift = scale_shift
        self.inner = inner(dim, d_out, **inner_kwargs)
        if scale_shift:
            self.norm_in = LayerNorm(dim, use_affine=False)
            self.norm_out = norm_scales_and_shifts(
                norm_fn, dim, init_weight=init_weight
            )
        else:
            self.norm_in = norm_fn(dim)
            self.norm_out = norm_fn(dim, init_weight=init_weight)

    def forward(self, x, **kwargs):
        if self.scale_shift:
            x = self.norm_in(x)
            y = x
        else:
            y = self.norm_in(x, **kwargs)

        y = self.inner(y, **kwargs)
        y = self.norm_out(y, **kwargs)

        if self.scale_shift:
            scales, shifts = y.chunk(2, dim=1)
            return shifts.addcmul(x, scales)
        else:
            return x + y
