from typing import Sequence
from torch import nn, Tensor

from diffusion_utils.utils import *
from diffusion_utils.attn import *
from diffusion_utils.ff import *
from diffusion_utils.norm import *
from diffusion_utils.pos_emb import *


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
        conv_fft: bool = False,
        use_depthwise: bool = False,
        ff_act: activation_type = None,
        depthwise_act: activation_type = default_activation,
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

        def mk_ff():
            if conv_fft:
                ff = ConvFFT(
                    dim,
                    mult=ff_mult,
                    norm_fn=norm_fn,
                    depthwise_act=depthwise_act,
                    use_depthwise=use_depthwise,
                )
            else:
                ff = FeedForward(
                    dim,
                    mult=ff_mult,
                    norm_fn=norm_fn,
                    depthwise_act=depthwise_act,
                    act=ff_act,
                    use_depthwise=use_depthwise,
                )
            return pre_norm(dim, ff, norm_fn=norm_fn)

        for _ in range(depth):
            self.attns.append(mk_attn())
            self.ffs.append(mk_ff())
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
                    )
                )

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
            "cond": cond,
            "time": time,
            "global_cond": global_cond,
            "classes": classes,
            "skip": skip,
        }

        attn_time_emb = None
        ff_time_emb = None
        if time is not None:
            assert (
                self.attn_time_emb is not None and self.ff_time_emb is not None
            ), "time_emb_dim must be given on init if you are conditioning based on time"
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
                kwargs["cond"] = cond
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


BlockTemplate = Callable[[int, int], Block]


def mk_template(kwargs):
    def block_template(ch: int, depth: int):
        return Block(ch, depth, **kwargs)

    return block_template


class SequentialBlock(nn.Module):
    def __init__(self, ch: int, depth: int, template: BlockTemplate):
        super().__init__()
        self.inner = template(ch, depth)

    def forward(self, tup: MutableSequence[Tensor]):
        tup[0] = self.inner.forward(*tup)
        return tup


class DownBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        depth: int = 2,
        factor: int = 2,
        inner: BlockTemplate = Block,
    ):
        super().__init__()
        self.block = PushBack(inner(c_in, depth))
        self.to_down = nn.Sequential(
            nn.Conv2d(c_in, c_out // factor ** 2, 3, padding=1),
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


class UpBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        depth: int = 2,
        factor: int = 2,
        inner: BlockTemplate = Block,
    ):
        super().__init__()
        self.to_up = nn.Sequential(
            nn.Conv2d(c_in, c_out * factor ** 2, 3, padding=1),
            nn.PixelShuffle(factor),
        )
        self.block = inner(c_out, depth)
        self.skip = SkipConv(c_out)

    def forward(self, tup: MutableSequence[Tensor]):
        tup[0] = self.to_up(tup[0])
        skip = tup.pop()
        tup[0] = self.skip(tup[0], skip)
        tup[0] = self.block(*tup, skip=skip)
        return tup


def unet(
    dim: int,
    channels: int = 3,
    stages: int = 4,
    num_blocks: int = 2,
    dim_head: int = 64,
    heads: int = 8,
    ff_mult: int = 4,
    time_emb: bool = False,
    time_emb_dim: Optional[int] = None,
    cross_attn: bool = False,
    rotary_emb: bool = True,
    conv_fft: bool = False,
    use_depthwise: bool = False,
    ff_act: activation_type = None,
    depthwise_act: activation_type = default_activation,
    input_channels: Optional[int] = None,
    output_channels: Optional[int] = None,
    d_cond: int = 512,
    norm_fn: NormFnType = LayerNorm,
    input_res: int = 128,
    mults: Sequence[int] = None,
    num_classes: Optional[int] = None,
    global_cond: bool = False,
):
    if global_cond:
        assert num_classes is None
        norm_fn = partial(ConditionalLayerNorm2d, d_cond)
    elif num_classes:
        norm_fn = partial(ClassConditionalLayerNorm2d, num_classes)
    else:
        norm_fn = LayerNorm

    if mults is None:
        mults = [2 ** (i + 1) for i in range(stages)]
    else:
        mults = list(mults)

    if len(mults) < stages:
        mults = mults + [mults[-1] for _ in range(stages - len(mults))]
    elif len(mults) > stages:
        mults = mults[:stages]

    mults = [1] + mults
    ins = [dim * m for m in mults[:-1]]
    outs = [dim * m for m in mults[1:]]

    resolutions = [input_res // 2 ** i for i in range(stages)]
    use_channel_attn = [res ** 2 > d for res, d in zip(resolutions, outs)]

    input_channels = default(input_channels, channels)
    output_channels = default(output_channels, channels)

    to_time_emb = None
    time_emb_dim = None

    project_in = nn.Sequential(
        nn.Conv2d(input_channels, dim, 3, padding=1),
        nn.LeakyReLU(inplace=True),
    )

    emb_in: nn.Module
    if time_emb:
        time_emb_dim = dim
        to_time_emb = nn.Sequential(
            TimeSinuPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        emb_in = ApplyMods(project_in, to_time_emb)
    else:
        emb_in = ApplyMod(project_in, ix=0)

    project_out = nn.Conv2d(dim, output_channels, 3, padding=1)

    downs = []
    ups = []
    mid = []

    for ind, use_channel_attn, d_in, d_out in zip(
        range(stages),
        use_channel_attn,
        ins,
        outs,
    ):
        is_last = ind == (stages - 1)
        kwargs = {
            "dim_head": dim_head,
            "heads": heads,
            "ff_mult": ff_mult,
            "time_emb_dim": time_emb_dim,
            "cross_attn": cross_attn,
            "use_channel_attn": use_channel_attn,
            "d_cond": d_cond,
            "norm_fn": norm_fn,
            "rotary_emb": rotary_emb,
            "conv_fft": conv_fft,
            "use_depthwise": use_depthwise,
            "ff_act": ff_act,
            "depthwise_act": depthwise_act,
        }

        block_template = mk_template(kwargs)

        downs.append(
            DownBlock(
                d_in,
                d_out,
                depth=num_blocks,
                inner=block_template,
            )
        )

        kwargs["skip"] = True
        block_template = mk_template(kwargs)

        ups.append(
            UpBlock(
                d_out,
                d_in,
                depth=num_blocks,
                inner=block_template,
            )
        )

        kwargs["skip"] = False
        if dim_head * heads < d_out:
            if heads < dim_head:
                heads = d_out // dim_head
                kwargs["heads"] = heads
            else:
                dim_head = d_out // heads
                kwargs["dim_head"] = dim_head
        block_template = mk_template(kwargs)

        if is_last:
            for _ in range(2):
                mid.append(
                    SequentialBlock(
                        ch=d_out,
                        depth=num_blocks,
                        template=block_template,
                    )
                )

    return nn.Sequential(
        emb_in,
        *downs,
        *mid,
        *reversed(ups),
        RetIndex(0),
        project_out,
    )


class UNet(nn.Module):
    def __init__(
        self,
        dim: int,
        channels: int = 3,
        stages: int = 4,
        num_blocks: int = 2,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        time_emb: bool = False,
        time_emb_dim: Optional[int] = None,
        cross_attn: bool = False,
        rotary_emb: bool = True,
        conv_fft: bool = False,
        use_depthwise: bool = False,
        ff_act: activation_type = None,
        depthwise_act: activation_type = default_activation,
        input_channels: Optional[int] = None,
        output_channels: Optional[int] = None,
        d_cond: int = 512,
        norm_fn: NormFnType = LayerNorm,
        input_res: int = 128,
        mults: Sequence[int] = None,
        num_classes: Optional[int] = None,
        global_cond: bool = False,
    ):
        super().__init__()
        self.inner = unet(
            dim=dim,
            channels=channels,
            stages=stages,
            num_blocks=num_blocks,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            time_emb=time_emb,
            time_emb_dim=time_emb_dim,
            cross_attn=cross_attn,
            rotary_emb=rotary_emb,
            conv_fft=conv_fft,
            use_depthwise=use_depthwise,
            ff_act=ff_act,
            depthwise_act=depthwise_act,
            input_channels=input_channels,
            output_channels=output_channels,
            d_cond=d_cond,
            norm_fn=norm_fn,
            input_res=input_res,
            mults=mults,
            num_classes=num_classes,
            global_cond=global_cond,
        )

    def forward(
        self,
        x: Tensor,
        time: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
        global_cond: Optional[Tensor] = None,
        classes: Optional[Tensor] = None,
    ):
        return self.inner((x, time, cond, global_cond, classes))
