import math
from math import pi
from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from einops import rearrange, repeat  # type: ignore

def apply_rotary_emb(q: Tensor, k: Tensor,
                     pos_emb: Tensor) -> Tuple[Tensor, Tensor]:
    sin, cos = pos_emb
    dim_rotary = sin.shape[-1]
    (q, q_pass), (k, k_pass) = map(
        lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    q, k = map(lambda t: torch.cat(t, dim=-1), ((q, q_pass), (k, k_pass)))
    return q, k


def rotate_every_two(x: Tensor) -> Tensor:
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d j -> ... (d j)')


class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_freq: int = 10):
        super().__init__()
        self.dim = dim
        scales = torch.linspace(1., max_freq / 2, self.dim // 4)
        self.register_buffer('scales', scales)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        device, dtype, h, w = x.device, x.dtype, *x.shape[-2:]  # type: ignore

        seq_x = torch.linspace(-1., 1., steps=h, device=device, dtype=dtype)
        seq_x = seq_x.unsqueeze(-1)

        seq_y = torch.linspace(-1., 1., steps=w, device=device, dtype=dtype)
        seq_y = seq_y.unsqueeze(-1)

        scales: Tensor = self.scales  # type: ignore
        scales_x = scales[(*((None, ) * (len(seq_x.shape) - 1)), Ellipsis)]
        scales_y = scales[(*((None, ) * (len(seq_y.shape) - 1)), Ellipsis)]

        seq_x = seq_x * scales_x.to(x) * pi
        seq_y = seq_y * scales_y.to(x) * pi

        x_sinu = repeat(seq_x, 'i d -> i j d', j=w)
        y_sinu = repeat(seq_y, 'j d -> i j d', i=h)

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim=-1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim=-1)

        sin, cos = map(lambda t: rearrange(t, 'i j d -> i j d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'i j d -> () i j (d r)', r=2),
                       (sin, cos))
        return sin, cos


class TimeSinuPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = self.dim // 2
        self.emb_scale = -math.log(10000) / (half_dim - 1)

    def forward(self, x: Tensor) -> Tensor:
        device, dtype = x.device, x.dtype
        half_dim = self.dim // 2
        emb = torch.arange(half_dim, dtype=dtype, device=device)
        emb = torch.exp(emb * self.emb_scale)
        emb = torch.einsum('i, j -> i  j', x, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FourierFeatures(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        assert out_features % 2 == 0
        self.map = nn.Linear(in_features, out_features // 2, bias=False)
        with torch.no_grad():
            self.map.weight.mul_(2 * math.pi)

    def forward(self, input: Tensor) -> Tensor:
        f = self.map(input)
        return torch.cat([f.cos(), f.sin()], dim=-1)


class FourierPosEmb(nn.Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.map = nn.Conv2d(2, out_features // 2, 1, bias=False)
        with torch.no_grad():
            self.map.weight.mul_(2 * math.pi)

    def forward(
        self,
        x: Optional[Tensor] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
    ) -> Tensor:
        if x is not None:
            b, _, h, w = x.shape
            dtype, device = x.dtype, x.device
        else:
            if h is None and w is not None:
                h = w
            elif w is None and h is not None:
                w = h
            else:
                raise ValueError(h, w, 'one of h or w must not be None')
            device = self.map.weight.device
            dtype = self.map.weight.dtype

        h_axis = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        w_axis = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        grid = torch.cat(
            torch.broadcast_tensors(
                h_axis[None, None, :, None],
                w_axis[None, None, None, :],
            ),
            dim=1,
        )
        f = self.map(grid)
        f = torch.cat([f.cos(), f.sin()], dim=1)
        f = f.broadcast_to(b, -1, h, w)
        return f
