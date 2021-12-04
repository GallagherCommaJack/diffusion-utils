
import math
import torch
from torch import nn

class DropKwargs(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, *args, **kwargs):
        return self.inner(*args)


class SequentialKwargs(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.inner = nn.ModuleList(modules)

    def forward(self, x, **kwargs):
        out = x
        for module in self.inner:
            out = module(out, **kwargs)
        return out


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, depth=1):
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


def clamp_exp(t: torch.Tensor, low=math.log(1e-2), high=math.log(100)):
    return clamp_with_grad(t, low, high).exp()

def mk_full(d, init):
    if isinstance(init, torch.Tensor):
        return init
    else:
        return torch.full([d], init)
