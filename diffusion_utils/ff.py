from functools import partial
from typing import Callable
from typing_extensions import Protocol

from torch import nn
from einops import reduce

from diffusion_utils.utils import *
from diffusion_utils.norm import *
from diffusion_utils.pos_emb import *

activation_type = Callable[[], nn.Module]
default_activation: activation_type = partial(nn.LeakyReLU, inplace=True)


class DepthwiseSeparableConv2d(nn.Sequential):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 3,
        bias: bool = True,
        act: activation_type = default_activation,
    ):
        super().__init__(
            nn.Conv2d(c_in, c_out, 1, bias=False),
            act(),
            nn.Conv2d(
                c_out,
                c_out,
                k,
                groups=c_out,
                bias=bias,
                padding="same",
            ),
        )


class DepthwiseRematConvFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        k1,
        k2,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.save_for_backward(input, k1, k2)
        with torch.no_grad():
            weight = torch.einsum("oi,hw->oihw", k1, k2)
            output = F.conv2d(
                input,
                weight,
                bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, k1, k2 = ctx.saved_tensors
        stride, padding, dilation, groups = (
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.groups,
        )
        grad_input = grad_k1 = grad_k2 = grad_bias = None
        with torch.enable_grad():
            weight = torch.einsum("oi,hw->oihw", k1, k2)
        if ctx.needs_input_grad[0]:
            grad_input = F.grad.conv2d_input(
                input.shape,
                weight,
                grad_output,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            grad_weight = F.grad.conv2d_weight(
                input,
                weight.shape,
                grad_output,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            if ctx.needs_input_grad[1] and ctx.needs_input_grad[2]:
                grad_k1, grad_k2 = torch.autograd.grad(weight, (k1, k2), grad_weight)
            elif ctx.needs_input_grad[1]:
                grad_k1 = torch.autograd.grad(weight, k1, grad_weight)
            elif ctx.needs_input_grad[2]:
                grad_k2 = torch.autograd.grad(weight, k2, grad_weight)
        if ctx.needs_input_grad[3]:
            grad_bias = reduce(grad_output, "b c h w -> c", "sum")
        return grad_input, grad_k1, grad_k2, grad_bias, None, None, None, None


depthwise_remat_conv = DepthwiseRematConvFn.apply


def conv_layer(
    c_in: int,
    c_out: int,
    k: int = 3,
    bias: bool = True,
    use_depthwise: bool = False,
    depthwise_act: activation_type = default_activation,
):
    if use_depthwise:
        return DepthwiseSeparableConv2d(c_in, c_out, k, bias, depthwise_act)
    else:
        return nn.Conv2d(c_in, c_out, k, bias=bias, padding="same")


class ConvFFT(nn.Module):
    def __init__(
        self,
        dim: int,
        mult: int = 4,
        use_depthwise: bool = True,
        pos_features: int = 16,
        norm_fn: NormFnType = LayerNorm,
        depthwise_act: activation_type = default_activation,
    ):
        super().__init__()
        hidden_dim = int(dim * mult)
        conv = partial(
            conv_layer,
            use_depthwise=use_depthwise,
            depthwise_act=depthwise_act,
        )
        self.in_norm = norm_fn(dim)
        self.map_in = conv(dim, hidden_dim)
        self.pos_emb = FourierPosEmb(pos_features)
        self.fft_map = conv(
            hidden_dim * 2 + pos_features,
            hidden_dim * 2,
            use_depthwise=use_depthwise,
        )
        self.map_out = conv(hidden_dim, dim)
        self.out_norm = norm_fn(dim, init_weight=1e-3)

    def forward(self, x, **kwargs):
        y = self.map_in(self.in_norm(x))
        yf = torch.fft.rfft2(y, norm="ortho")
        ym, ya = yf.abs(), yf.angle()
        pe = self.pos_emb(ym)
        yf = self.fft_map(torch.cat([ym, ya, pe], dim=1))
        yf = torch.polar(*yf.chunk(2, dim=1))
        y = torch.fft.irfft2(yf, norm="ortho")
        y = self.map_out(y)
        return x + self.out_norm(y, **kwargs)


class FF(nn.Module):
    def __init__(
        self,
        dim: int,
        mult: int = 4,
        norm_fn: NormFnType = LayerNorm,
        depthwise_act: activation_type = default_activation,
        act: activation_type = None,
        use_depthwise: bool = False,
    ):
        super().__init__()
        if act is None:
            act = depthwise_act
        conv = partial(
            conv_layer, use_depthwise=use_depthwise, depthwise_act=depthwise_act
        )
        inner = [
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            conv(dim, dim * mult),
            act(),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * mult, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim),
        ]
        with torch.no_grad():
            inner[-1].weight.fill_(1e-1)
        self.inner = nn.Sequential(*inner)

    def forward(self, x):
        return x + self.inner(x)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        mult: int = 4,
        norm_fn: NormFnType = LayerNorm,
        depthwise_act: activation_type = default_activation,
        act: activation_type = None,
        use_depthwise: bool = True,
    ):
        super().__init__()
        hidden_dim = int(dim * mult)
        if act is None:
            act = depthwise_act

        conv = partial(
            conv_layer,
            use_depthwise=use_depthwise,
            depthwise_act=depthwise_act,
        )

        self.map_in = conv(dim, hidden_dim)
        self.map_out = nn.Sequential(
            conv(hidden_dim, hidden_dim),
            act(),
            conv(hidden_dim, dim, bias=False),
        )
        self.out_norm = norm_fn(dim, init_weight=1e-3)

    def forward(self, x, **kwargs):
        y = self.map_in(x)
        y = self.map_out(y)
        y = self.out_norm(y, **kwargs)
        return y + x


def downsampler(
    c_in: int,
    c_out: int,
    factor: int = 2,
    norm_fn: NormFnType = LayerNorm,
):
    to_out = DropKwargs(
        nn.Sequential(
            nn.Conv2d(c_in, c_out // factor ** 2, 3, padding=1),
            nn.PixelUnshuffle(factor),
        )
    )
    return pre_norm(c_in, to_out, norm_fn=norm_fn)


def upsampler(
    c_in: int,
    c_out: int,
    factor: int = 2,
    norm_fn: NormFnType = LayerNorm,
):
    to_out = DropKwargs(
        nn.Sequential(
            nn.Conv2d(c_in, c_out * factor ** 2, 3, padding=1),
            nn.PixelShuffle(factor),
        )
    )
    return pre_norm(c_in, to_out, norm_fn=norm_fn)
