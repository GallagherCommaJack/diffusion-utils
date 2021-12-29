from abc import abstractmethod
from typing import Any, Callable, Dict, Type
import torch
from utils import factor_int
from torch import nn


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

    def sub_builder(self, builder_class: Type[ModelBuilder], **kwargs) -> ModelBuilder:
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


# class MoEConvLayer(HasConvLayer):
#     def conv_layer(self, c_in: int, c_out: int, k: int = 1, stride: int = 1, padding="same",  bias: bool = True, **kwargs):
