from abc import abstractmethod
from functools import partial
from typing import Any, Callable, Dict, Tuple, Type
import torch
from utils import factor_int
from torch import nn
import math
from einops.layers.torch import Rearrange

from .utils import clamp_exp
from .attn import ScaledCosineAttention


class ModelBuilder:
    def __init__(self, **kwargs):
        self.hparams = kwargs

    def get_param(self, name: str, default: Callable[[], Any] = None):
        try:
            return self.hparams[name]
        except KeyError as e:
            if default is None:
                raise e
            else:
                v = default()
                self.hparams[name] = v
                return v

    @abstractmethod
    def build_inner(self) -> nn.Module:
        pass

    def build(self, **kwargs: Dict[str, Any]) -> nn.Module:
        self.hparams.update(kwargs)
        return self.build_inner()

    def sub_builder(self, builder_class, **kwargs):
        hparams = self.hparams | kwargs
        return builder_class(**hparams)


class HasConvLayer(ModelBuilder):
    def conv_layer(
        self,
        c_in: int,
        c_out: int,
        k: int = 1,
        stride: int = 1,
        padding="same",
        groups: int = 1,
        bias: bool = True,
        **kwargs,
    ):
        return nn.Conv2d(
            c_in, c_out, k, stride, padding=padding, groups=groups, bias=bias
        )


class ShuffleConvLayer(HasConvLayer):
    def conv_layer(
        self,
        c_in: int,
        c_out: int,
        k: int = 1,
        stride: int = 1,
        padding="same",
        groups: int = 1,
        bias: bool = True,
        k_2: int = 3,
        **kwargs,
    ):
        conv_in = super().conv_layer(
            c_in, c_out, k, stride, padding, groups, bias=False
        )
        g1, g2 = factor_int(c_out)
        g = min(g1, g2)
        c1 = nn.Conv2d(c_out, c_out, k_2, padding="same", groups=g, bias=False)
        shuf = nn.ChannelShuffle(g)
        c2 = nn.Conv2d(c_out, c_out, k_2, padding="same", groups=g, bias=bias)
        return nn.Sequential(conv_in, c1, shuf, c2)


class DepthwiseConvLayer(HasConvLayer):
    def conv_layer(
        self,
        c_in: int,
        c_out: int,
        k: int = 1,
        stride: int = 1,
        padding="same",
        bias: bool = True,
        **kwargs,
    ):
        c1 = super().conv_layer(c_in, c_out, bias=False)
        c2 = super().conv_layer(
            c_out, c_out, k=k, stride=stride, padding=padding, groups=c_out, bias=bias
        )
        return nn.Sequential(c1, c2)


class HasChannelNorm(ModelBuilder):
    def channel_norm(self, dim: int, affine: bool = True):
        return nn.BatchNorm2d(dim, affine=affine)


class HasGroupNorm(ModelBuilder):
    def group_norm(self, dim: int, groups: int, affine: bool = True):
        return nn.GroupNorm(groups, dim, affine=affine)


class HasFeedForward(HasConvLayer, HasChannelNorm):
    def ff_act():
        return nn.ReLU(inplace=True)

    def feedforward(self, dim: int) -> nn.Module:
        mult = self.get_param("ff_mult", lambda: 4)
        res_weight = self.get_param("res_weight", lambda: 1e-1)
        layers = [
            self.channel_norm(dim),
            self.ff_act(),
            self.conv_layer(dim, dim * mult, 3),
            self.ff_act(),
            self.conv_layer(dim * mult, dim, 3),
            self.ff_act(),
            self.channel_norm(dim),
        ]
        with torch.no_grad():
            layers[-1].weight.fill_(res_weight)
        return nn.Sequential(*layers)


class DownsamplingStage(nn.Module):
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class UpsamplingStage(nn.Module):
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        pass


# class HasDownsamplingStage(HasConvLayer):
