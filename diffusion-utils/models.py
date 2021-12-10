from torch import nn, Tensor

from utils import *
from attn import *
from ff import *
from norm import *
from pos_emb import *


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        time_emb_dim: Optional[int] = None,
        rotary_emb: bool = True,
        cross_attn: bool = False,
        use_channel_attn: bool = True,
        d_cond: int = 512,
        skip: bool = False,
        norm_fn: NormFnType = LayerNorm,
    ):
        super().__init__()

        self.attn_time_emb = None
        self.ff_time_emb = None
        d_hidden = int(dim * ff_mult)
        if time_emb_dim is not None:
            self.attn_time_emb = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, dim),
            )
            self.ff_time_emb = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, d_hidden),
            )

        self.pos_emb = AxialRotaryEmbedding(dim_head) if rotary_emb else None

        self.attns = nn.ModuleList([])
        self.ffs = nn.ModuleList([])
        self.cross: Optional[nn.ModuleList]
        if cross_attn:
            self.cross = nn.ModuleList([])
        else:
            self.cross = None

        def mk_attn():
            if use_channel_attn:
                attn = ChannelAttention(
                    dim,
                    heads=heads,
                    norm_fn=norm_fn,
                )
            else:
                attn = SelfAttention2d(
                    dim,
                    dim_head=dim_head,
                    heads=heads,
                    skip=skip,
                    norm_fn=norm_fn,
                )
            return pre_norm(dim, attn, norm_fn=norm_fn)

        for _ in range(depth):
            self.attns.append(mk_attn())
            self.ffs.append(
                pre_norm(
                    dim,
                    FeedForward(dim, mult=ff_mult, norm_fn=norm_fn),
                    norm_fn=norm_fn,
                ))
            if self.cross is not None:
                self.cross.append(
                    pre_norm(
                        dim,
                        DuplexAttn(
                            dim,
                            dim_head=dim_head,
                            heads=heads,
                            d_cond=d_cond,
                            norm_fn=norm_fn,
                        ),
                        norm_fn=norm_fn,
                    ))

    def forward(
        self,
        x: Tensor,
        time: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
        global_cond: Optional[Tensor] = None,
        classes: Optional[Tensor] = None,
        *args,
        skip: Optional[Tensor] = None,
    ):
        kwargs = {
            'cond': cond,
            'time': time,
            'global_cond': global_cond,
            'classes': classes,
            'skip': skip,
        }

        attn_time_emb = None
        ff_time_emb = None
        if time is not None:
            assert self.attn_time_emb is not None and self.ff_time_emb is not None, 'time_emb_dim must be given on init if you are conditioning based on time'
            attn_time_emb = self.attn_time_emb(time)
            ff_time_emb = self.ff_time_emb(time)

        pos_emb = None
        if self.pos_emb is not None:
            pos_emb = self.pos_emb(x)

        if cond is not None and self.cross is not None:
            for attn, ff, cross in zip(self.attns, self.ffs, self.cross):
                x = attn(
                    x,
                    time_emb=attn_time_emb,
                    pos_emb=pos_emb,
                    **kwargs,
                )
                x = ff(x, time_emb=ff_time_emb, **kwargs)
                x, cond = cross(x, time_emb=attn_time_emb, **kwargs)
                kwargs['cond'] = cond
        else:
            for attn, ff in zip(self.attns, self.ffs):
                x = attn(
                    x,
                    time_emb=attn_time_emb,
                    pos_emb=pos_emb,
                    **kwargs,
                )
                x = ff(x, time_emb=ff_time_emb, **kwargs)

        return x


class SequentialBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.block = Block(*args, **kwargs)

    def forward(self, tup: MutableSequence[Tensor]):
        tup[0] = self.block.forward(*tup)
        return tup


class SequentialDownBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        factor: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.block = PushBack(Block(c_in, *args, **kwargs))
        self.to_down = nn.Sequential(
            nn.Conv2d(c_in, c_out // factor**2, 3, padding=1),
            nn.PixelUnshuffle(factor),
        )

    def forward(self, tup: MutableSequence[Tensor]):
        tup = self.block(tup)
        tup[0] = self.to_down(tup[0])
        return tup


class SkipConv(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.inner = nn.Conv2d(ch * 2, ch, 3, padding=1)

    def forward(self, x, skip):
        stacked = torch.cat([x, skip], dim=1)
        return self.inner(stacked)


class SequentialUpBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        factor: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.to_up = nn.Sequential(
            nn.Conv2d(c_in, c_out * factor**2, 3, padding=1),
            nn.PixelShuffle(factor),
        )
        self.block = Block(c_out, *args, **kwargs)
        self.skip = SkipConv(c_out)

    def forward(self, tup: MutableSequence[Tensor]):
        tup[0] = self.to_up(tup[0])
        skip = tup.pop()
        tup[0] = self.skip(tup[0], skip)
        tup[0] = self.block(*tup, skip=skip)
        return tup


def unet(
    dim=64,
    channels=3,
    stages=4,
    num_blocks=2,
    dim_head=64,
    heads=8,
    ff_mult=4,
    time_emb=False,
    input_channels=None,
    output_channels=None,
    cross_attn=False,
    input_res=128,
    d_cond=512,
    mults=None,
    num_classes=None,
    global_cond=False,
):
    if global_cond:
        assert num_classes is None
        norm_fn = partial(ConditionalLayerNorm2d, d_cond=d_cond)
    elif num_classes:
        norm_fn = partial(ClassConditionalLayerNorm2d, n_classes=num_classes)
    else:
        norm_fn = LayerNorm

    if mults is None:
        mults = [2**(i + 1) for i in range(stages)]
    elif len(mults) < stages:
        mults = mults + [mults[-1] for _ in range(stages - len(mults))]
    elif len(mults) > stages:
        mults = mults[:stages]
    mults = [1] + mults
    ins = [dim * m for m in mults[:-1]]
    outs = [dim * m for m in mults[1:]]

    resolutions = [input_res // 2**i for i in range(stages)]
    use_channel_attn = [
        res**2 > (dim * m) for m, res in zip(mults, resolutions)
    ]

    input_channels = default(input_channels, channels)
    output_channels = default(output_channels, channels)

    to_time_emb = None
    time_emb_dim = None

    if time_emb:
        time_emb_dim = dim
        to_time_emb = nn.Sequential(
            TimeSinuPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    project_in = nn.Sequential(
        nn.Conv2d(input_channels, dim, 3, padding=1),
        nn.LeakyReLU(inplace=True),
    )

    project_out = nn.Conv2d(dim, output_channels, 3, padding=1)

    downs = []
    ups = []

    heads, dim_head, num_blocks = map(partial(cast_tuple, depth=stages),
                                      (heads, dim_head, num_blocks))

    for ind, heads, dim_head, num_blocks, use_channel_attn, d_in, d_out in zip(
            range(stages),
            heads,
            dim_head,
            num_blocks,
            use_channel_attn,
            ins,
            outs,
    ):

        is_last = ind == (stages - 1)

        downs.append(
            SequentialDownBlock(
                d_in,
                d_out,
                depth=num_blocks,
                dim_head=dim_head,
                heads=heads,
                ff_mult=ff_mult,
                time_emb_dim=time_emb_dim,
                cross_attn=cross_attn,
                use_channel_attn=use_channel_attn,
                d_cond=d_cond,
                norm_fn=norm_fn,
            ))

        ups.append(
            SequentialUpBlock(
                d_out,
                d_in,
                depth=num_blocks,
                dim_head=dim_head,
                heads=heads,
                ff_mult=ff_mult,
                time_emb_dim=time_emb_dim,
                cross_attn=cross_attn,
                use_channel_attn=use_channel_attn,
                d_cond=d_cond,
                skip=True,
                norm_fn=norm_fn,
            ))

        if dim_head * heads < d_out:
            if heads < dim_head:
                heads = d_out // dim_head
            else:
                dim_head = d_out // heads

        if is_last:
            mid1 = SequentialBlock(
                dim=d_out,
                depth=num_blocks,
                dim_head=dim_head,
                heads=heads,
                ff_mult=ff_mult,
                time_emb_dim=time_emb_dim,
                cross_attn=cross_attn,
                use_channel_attn=use_channel_attn,
                d_cond=d_cond,
                norm_fn=norm_fn,
            )
            mid2 = SequentialBlock(
                dim=d_out,
                depth=num_blocks,
                dim_head=dim_head,
                heads=heads,
                ff_mult=ff_mult,
                time_emb_dim=time_emb_dim,
                cross_attn=cross_attn,
                use_channel_attn=use_channel_attn,
                d_cond=d_cond,
                norm_fn=norm_fn,
            )
            mid = [mid1, mid2]

    return nn.Sequential(
        ApplyMods(
            project_in,
            to_time_emb,
        ),
        *downs,
        *mid,
        *reversed(ups),
        RetIndex(0),
        project_out,
    )
