import math
from math import pi

import torch
from torch import nn

from einops import rearrange, repeat


def apply_rotary_emb(q, k, pos_emb):
    sin, cos = pos_emb
    dim_rotary = sin.shape[-1]
    (q, q_pass), (k, k_pass) = map(
        lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    q, k = map(lambda t: torch.cat(t, dim=-1), ((q, q_pass), (k, k_pass)))
    return q, k


def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d j -> ... (d j)')


class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq=10):
        super().__init__()
        self.dim = dim
        scales = torch.linspace(1., max_freq / 2, self.dim // 4)
        self.register_buffer('scales', scales)

    def forward(self, x):
        device, dtype, h, w = x.device, x.dtype, *x.shape[-2:]

        seq_x = torch.linspace(-1., 1., steps=h, device=device, dtype=dtype)
        seq_x = seq_x.unsqueeze(-1)

        seq_y = torch.linspace(-1., 1., steps=w, device=device, dtype=dtype)
        seq_y = seq_y.unsqueeze(-1)

        scales = self.scales[(*((None, ) * (len(seq_x.shape) - 1)), Ellipsis)]
        scales = scales.to(x)

        scales = self.scales[(*((None, ) * (len(seq_y.shape) - 1)), Ellipsis)]
        scales = scales.to(x)

        seq_x = seq_x * scales * pi
        seq_y = seq_y * scales * pi

        x_sinu = repeat(seq_x, 'i d -> i j d', j=w)
        y_sinu = repeat(seq_y, 'j d -> i j d', i=h)

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim=-1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim=-1)

        sin, cos = map(lambda t: rearrange(t, 'i j d -> i j d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'i j d -> () i j (d r)', r=2),
                       (sin, cos))
        return sin, cos


class TimeSinuPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, dtype=x.dtype, device=x.device) * -emb)
        emb = torch.einsum('i, j -> i  j', x, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
