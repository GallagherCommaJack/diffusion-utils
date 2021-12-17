import math
from typing import MutableSequence, Optional, TypeVar, Union
import torch
from torch import nn
from torch import Tensor
from torch.types import Number

T = TypeVar("T")


def exists(val: Optional[T]) -> bool:
    return val is not None


def default(val: Optional[T], d: T) -> T:
    return d if val is None else val


def cast_tuple(val, depth: int = 1):
    return val if isinstance(val, tuple) else (val,) * depth


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


TensorSeq = MutableSequence[Tensor]


class PushBack(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(
        self,
        xtup: TensorSeq,
    ) -> TensorSeq:
        x = self.inner(*xtup)
        xtup.append(x)
        xtup[0] = x
        return xtup


class PopBack(nn.Module):
    def __init__(self, inner: nn.Module, key: str):
        super().__init__()
        self.inner = inner
        self.key = key

    def forward(self, xtup: TensorSeq) -> TensorSeq:
        kwargs = {self.key: xtup.pop()}
        x = self.inner(*xtup, **kwargs)
        xtup[0] = x
        return xtup


class ApplyMods(nn.Module):
    def __init__(self, *mods):
        super().__init__()
        self.inner = nn.ModuleList(mods)

    def forward(self, tup: TensorSeq) -> TensorSeq:
        for i, mod in enumerate(self.inner):
            tup[i] = mod(tup[i])
        return tup


class ApplyMod(nn.Module):
    def __init__(self, inner: nn.Module, ix: int = 0):
        super().__init__()
        self.inner = inner
        self.ix = ix

    def forward(self, tup: TensorSeq) -> TensorSeq:
        tup[self.ix] = self.inner(tup[self.ix])
        return tup


class RetIndex(nn.Module):
    def __init__(self, ix: int = 0):
        super().__init__()
        self.ix = ix

    def forward(self, tup: TensorSeq) -> Tensor:
        return tup[self.ix]


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        (input,) = ctx.saved_tensors
        return (
            grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0),
            None,
            None,
        )


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
    return -torch.expm1(1e-4 + 10 * t ** 2).log()


def get_alphas_sigmas(log_snrs):
    """Returns the scaling factors for the clean image and for the noise, given
    the log SNR for a timestep."""
    alphas_squared = log_snrs.sigmoid()
    return alphas_squared.sqrt(), (1 - alphas_squared).sqrt()


def calculate_stats(e):
    e_mean = e.mean()
    e_variance = (e - e_mean).pow(2).mean()
    e_variance_stable = max(e_variance, 1e-5)
    e_skewness = (e - e_mean).pow(3).mean() / e_variance_stable ** 1.5
    e_kurtosis = (e - e_mean).pow(4).mean() / e_variance_stable ** 2
    return e_mean, e_variance, e_skewness, e_kurtosis


def measure_perf(f):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    f()
    # Run some things here

    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)

    return elapsed_time_ms


def calc_delta(t_in, t_out):
    return math.pi / 2 * (t_in - t_out)


def diffusion_step(z, v, t_in, t_out):
    delta = calc_delta(t_in, t_out)
    z = torch.cos(delta) * z - torch.sin(delta) * v
    return z


def calc_v_with_distillation_errors(net, z, t_in, t_out, *args, **kwargs):
    v = net(z, t_in, *args, **kwargs)
    with torch.no_grad():
        delta = calc_delta(t_in, t_out)
        t_mid = (t_in + t_out) / 2
        z_1 = diffusion_step(z, v, t_in, t_mid)
        v_2 = net(z_1, t_mid, *args, **kwargs)
        z_2 = diffusion_step(z_1 < v_2, t_mid, t_out)
        targets = z / torch.tan(delta) - z_2 / torch.sin(delta)
    e = v.sub(targets).pow(2).mean(dim=[1, 2, 3])
    return v, e
