from functools import partial

from torch import nn

from utils import *
from norm import *
from pos_emb import *


class DepthwiseSeparableConv2d(nn.Sequential):
    def __init__(
            self,
            c_in,
            c_out,
            bias=True,
            act=partial(nn.LeakyReLU, inplace=True),
    ):
        super().__init__(
            nn.Conv2d(c_in, c_out, 1, bias=False),
            act(),
            nn.Conv2d(c_out, c_out, 3, groups=c_out, bias=bias, padding=1),
        )


class FeedForward(nn.Module):
    def __init__(
            self,
            dim,
            mult=4,
            norm_fn=LayerNorm,
            act=partial(nn.LeakyReLU, inplace=True),
            use_depthwise=True,
    ):
        super().__init__()
        hidden_dim = int(dim * mult)
        if use_depthwise:
            conv = partial(DepthwiseSeparableConv2d, act=act)
        else:
            conv = partial(nn.Conv2d, k=3, padding=1)

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


def downsampler(c_in, c_out, factor=2, norm_fn=LayerNorm):
    to_out = DropKwargs(
        nn.Sequential(
            nn.Conv2d(c_in, c_out // factor**2, 3, padding=1),
            nn.PixelUnshuffle(factor),
        ))
    return pre_norm(c_in, to_out, norm_fn=norm_fn)


def upsampler(c_in, c_out, factor=2, norm_fn=LayerNorm):
    to_out = DropKwargs(
        nn.Sequential(
            nn.Conv2d(c_in, c_out * factor**2, 3, padding=1),
            nn.PixelShuffle(factor),
        ))
    return pre_norm(c_in, to_out, norm_fn=norm_fn)
