from functools import partial
from typing import Callable
from typing_extensions import Protocol

from torch import nn

from utils import *
from norm import *
from pos_emb import *

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
                padding='same',
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
        return nn.Conv2d(c_in, c_out, k, bias=bias, padding='same')


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
        yf = torch.fft.rfft2(y, norm='ortho')
        ym, ya = yf.abs(), yf.angle()
        pe = self.pos_emb(ym)
        yf = self.fft_map(torch.cat([ym, ya, pe], dim=1))
        yf = torch.polar(*yf.chunk(2, dim=1))
        y = torch.fft.irfft2(yf, norm='ortho')
        y = self.map_out(y)
        return x + self.out_norm(y, **kwargs)


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

    def forward(self, x, time_emb=None, **kwargs):
        y = self.map_in(x)
        if exists(time_emb):
            time_emb = rearrange(time_emb, 'b c -> b c () ()')
            y = y + time_emb
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
            nn.Conv2d(c_in, c_out // factor**2, 3, padding=1),
            nn.PixelUnshuffle(factor),
        ))
    return pre_norm(c_in, to_out, norm_fn=norm_fn)


def upsampler(
    c_in: int,
    c_out: int,
    factor: int = 2,
    norm_fn: NormFnType = LayerNorm,
):
    to_out = DropKwargs(
        nn.Sequential(
            nn.Conv2d(c_in, c_out * factor**2, 3, padding=1),
            nn.PixelShuffle(factor),
        ))
    return pre_norm(c_in, to_out, norm_fn=norm_fn)
