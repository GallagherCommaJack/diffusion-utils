import math
from typing import Optional, Union
import torch
from torch import nn
from torch._C import T
from torch.types import Number


class DropKwargs(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, *args, **kwargs):
        return self.inner(*args)


class SequentialKwargs(nn.Module):
    def __init__(self, *modules: nn.Module):
        super().__init__()
        self.inner = nn.ModuleList(modules)

    def forward(self, x, **kwargs):
        out = x
        for module in self.inner:
            out = module(out, **kwargs)
        return out


def exists(val: Optional[T]) -> bool:
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, depth: int = 1):
    return val if isinstance(val, tuple) else (val, ) * depth


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >=
                          0), None, None


clamp_with_grad = ClampWithGrad.apply


def clamp_exp(
        t: torch.Tensor,
        low: float = math.log(1e-2),
        high: float = math.log(100),
):
    return clamp_with_grad(t, low, high).exp()


def mk_full(d: int, init: Union[torch.Tensor, Number]):
    if isinstance(init, torch.Tensor):
        return init
    else:
        return torch.full([d], init)
