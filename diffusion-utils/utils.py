import math
from typing import MutableSequence, Optional, TypeVar, Union
import torch
from torch import nn
from torch.types import Number

T = TypeVar('T')

def exists(val: Optional[T]) -> bool:
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, depth: int = 1):
    return val if isinstance(val, tuple) else (val, ) * depth


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


class PushBack(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, xtup: MutableSequence[torch.Tensor]):
        x = self.inner(*xtup)
        xtup.append(x)
        xtup[0] = x
        return xtup


class PopBack(nn.Module):
    def __init__(self, inner: nn.Module, key: str):
        super().__init__()
        self.inner = inner
        self.key = key

    def forward(self, xtup: MutableSequence[torch.Tensor]):
        kwargs = {self.key: xtup.pop()}
        x = self.inner(*xtup, **kwargs)
        xtup[0] = x
        return xtup


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


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].lerp_(param, 1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


def get_ddpm_schedule(t):
    """Returns log SNRs for the noise schedule from the DDPM paper."""
    return -torch.expm1(1e-4 + 10 * t**2).log()


def get_alphas_sigmas(log_snrs):
    """Returns the scaling factors for the clean image and for the noise, given
    the log SNR for a timestep."""
    alphas_squared = log_snrs.sigmoid()
    return alphas_squared.sqrt(), (1 - alphas_squared).sqrt()


def calculate_stats(e):
    e_mean = e.mean()
    e_variance = (e - e_mean).pow(2).mean()
    e_variance_stable = max(e_variance, 1e-5)
    e_skewness = (e - e_mean).pow(3).mean() / e_variance_stable**1.5
    e_kurtosis = (e - e_mean).pow(4).mean() / e_variance_stable**2
    return e_mean, e_variance, e_skewness, e_kurtosis