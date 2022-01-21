from functools import partial
from typing import Callable
from typing_extensions import Protocol

from torch import nn
from einops import reduce

from diffusion_utils.utils import compute_channel_change_mat, partial
from diffusion_utils.norm import *
from diffusion_utils.pos_emb import *

activation_type = Callable[[], nn.Module]
default_activation: activation_type = partial(nn.LeakyReLU, inplace=True)


def sn_layer(*args, **kwargs):
    conv = nn.Conv2d(*args, **kwargs)
    nn.init.kaiming_normal_(conv.weight, nonlinearity="linear")
    if exists(conv.bias):
        nn.init.constant_(conv.bias, 0.0)
    return conv


def sn_block(dim: int, depth: int = 2, k: int = 3):
    return nn.Sequential(
        *[
            nn.Sequential(sn_layer(dim, dim, k, padding="same"), nn.SELU())
            for _ in range(depth)
        ]
    )


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


class FFBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_out: int,
        norm_fn: NormFnType,
        ff_mult: int = 1,
        act: Callable[[], nn.Module] = nn.GELU,
    ):
        super().__init__()
        d_hidden = int(dim * ff_mult)
        self.conv_in = conv_layer(dim, d_hidden)
        self.act_1 = act()
        self.norm_mid = norm_fn(d_hidden)
        self.act_2 = act()
        self.conv_out = conv_layer(d_hidden, d_out)
        self.act_3 = act()

    def forward(self, x, **kwargs):
        y = self.conv_in(x)
        y = self.act_1(y)
        y = self.norm_mid(y, **kwargs)
        y = self.act_2(y)
        y = self.conv_out(y)
        y = self.act_3(y)
        return y


class Layer(SandwichNorm):
    def __init__(
        self,
        dim: int,
        norm_fn: NormFnType,
        scale_shift: bool = False,
        init_weight: float = 1e-3,
        ff_mult: int = 1,
    ):
        super().__init__(
            dim,
            FFBlock,
            norm_fn,
            scale_shift,
            init_weight,
            dict(ff_mult=ff_mult, norm_fn=norm_fn),
        )


class Downsample2d(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        dkern = torch.tensor([1, 2, 1]) / 4
        dkern = dkern[:, None] @ dkern[None, :]
        cmat = compute_channel_change_mat(c_out / c_in)
        weight = torch.einsum("hw,oi->oihw", dkern, cmat)
        self.register_buffer("weight", weight)

    def forward(self, x):
        n, c, h, w = x.shape
        o, i, _, _ = self.weight.shape
        groups = c // i
        x = rearrange(x, "b (g c) h w -> (b g) c h w", g=groups)
        x = F.conv2d(x, self.weight, padding=1, stride=2)
        x = rearrange(x, "(b g) c h w -> b (g c) h w", g=groups)
        return x


class Upsample2d(nn.Module):
    def __init__(self, c_in, c_out):
        # todo: figure out how to do this w/a fused kernel
        super().__init__()
        cmat = compute_channel_change_mat(c_out / c_in)
        self.register_buffer("weight", cmat[:, :, None, None])
        self.upsampler = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        n, c, h, w = x.shape
        o, i, _, _ = self.weight.shape
        groups = c // i
        x = rearrange(x, "b (g c) h w -> (b g) c h w", g=groups)
        x = F.conv2d(x, self.weight)
        x = rearrange(x, "(b g) c h w -> b (g c) h w", g=groups)
        return self.upsampler(x)
