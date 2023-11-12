import math
from abc import abstractmethod
from functools import partial
import numpy as np

import torch
from torch.utils.checkpoint import checkpoint
from torch import nn
from torch.nn import functional as F

# =============================================
# Adapted from github repo: https://github.com/AlexGraikos/diffusion_priors

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def max_adt_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D adaptive max pooling module.
    """
    if dims == 1:
        return nn.AdaptiveMaxPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AdaptiveMaxPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AdaptiveMaxPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def avg_adt_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D adaptive average pooling module.
    """
    if dims == 1:
        return nn.AdaptiveAvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AdaptiveAvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AdaptiveAvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(min(32, channels), channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]

class ReferenceBlock(nn.Module):
    """
    A block that takes a reference tensor as an input.
    """

    @abstractmethod
    def forward(self, x, ref):
        """
        Apply the module to `x` given `ref` as a reference tensor.
        """


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock, ReferenceBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb = None, ref_emb = None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, ReferenceBlock):
                x = layer(x, ref_emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, z_down = False, out_size = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.z_down = z_down
        self.out_size = out_size
        
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels, f"input channel({x.shape[1]}) must be equal to {self.channels}"
        if self.dims == 3:
            if self.out_size is None:
                if self.z_down:
                    x = F.interpolate(x, scale_factor= 2, mode="nearest")
                else: 
                    x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
            else:
                x = F.interpolate(x, self.out_size, mode="nearest")
        else:
            if self.out_size is None:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            else:
                x = F.interpolate(x, self.out_size, mode="nearest")
            
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, z_down = False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.z_down = z_down
        if dims == 3 and not z_down:
            stride = (1, 2, 2)
        else:
            stride = 2
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class MixBlock(ReferenceBlock):
    def __init__(self, channels, ref_channels, out_channels: None | int = None, mode = "concat", dims = 2):
        super().__init__()
        self.channels = channels
        self.ref_channels = ref_channels
        self.out_channels = out_channels or channels
        self.mode = mode
        if mode == "concat":
            self.mix = lambda x, y: torch.cat([x, y], dim=1)
            self.op = conv_nd(dims, channels + ref_channels, out_channels, 1)
        elif mode == "add":
            assert channels == ref_channels, f"channels({channels}) must be equal to ref_channels({ref_channels}) when mode is add"
            self.mix = torch.add
            self.op = conv_nd(dims, channels, out_channels, 1)
        elif mode == "dot":
            assert channels == ref_channels, f"channels({channels}) must be equal to ref_channels({ref_channels}) when mode is dot"
            self.mix = torch.mul
            self.op = conv_nd(dims, channels, out_channels, 1)
            
    def forward(self, x, ref):
        x = self.mix(x, ref)
        x = self.op(x)
        return x
        
class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False, use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False, z_down = False, padding_mode="reflect"):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.z_down = z_down

        self.in_layers = nn.Sequential()
        self.in_layers.add_module("norm", normalization(channels))
        self.in_layers.add_module("act", nn.SiLU())
        self.in_layers.add_module("conv", conv_nd(dims, channels, self.out_channels, 3, padding=1, padding_mode=padding_mode))

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, z_down = z_down)
            self.x_upd = Upsample(channels, False, dims, z_down = z_down)
        elif down:
            self.h_upd = Downsample(channels, False, dims, z_down = z_down)
            self.x_upd = Downsample(channels, False, dims, z_down = z_down)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        
        if self.emb_channels is not None:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                ),
            )
        self.out_layers = nn.Sequential()
        self.out_layers.add_module("norm", normalization(self.out_channels))
        self.out_layers.add_module("act", nn.SiLU())
        self.out_layers.add_module("drop", nn.Dropout(p=dropout))
        self.out_layers.add_module("conv", conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1))

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.use_checkpoint:
            return checkpoint(self._forward, x, emb)
        else:
            return self._forward(x, emb)
        # return checkpoint(
        #     self._forward, (x, emb), self.parameters(), self.use_checkpoint
        # )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
            
        if self.emb_channels is not None:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            if self.use_scale_shift_norm:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = h + emb_out
                h = self.out_layers(h)
        else:
            h = self.out_layers(h)
            
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
        position_encode = True,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.q = conv_nd(1, channels, channels, 1)
        self.k = conv_nd(1, channels, channels, 1)
        self.v = conv_nd(1, channels, channels, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)
        
        self.position_encode = PositionalEncoding(channels) if position_encode else None
        
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        h = self.norm(x)
        q = k = self._pos_emb(h)
        qkv = torch.cat([self.q(q), self.k(k), self.v(h)], dim=1)
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)
    
    def _pos_emb(self, x, channel_first = True):
        if self.position_encode is None:
            return x
        else:
            if channel_first:
                b, c, *spatial = x.shape
                shape = (b, *spatial, c)
                pos = self.position_encode(shape, device = x.device)
                pos.transpose_(1, 2)
            else:
                shape = x.shape
                pos = self.position_encode(shape, device = x.device)
            return x + pos


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)
        
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha: torch.Tensor):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha * grad_output
        return grad_input, None
    
def grad_reverse(x, alpha = 0.05):
    alpha = torch.tensor(alpha, device = x.device)
    return GradReverse.apply(x, alpha)

# =====================================
# copied and adapted from github: https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py

def positional_encoding(x: torch.Tensor | tuple, 
                        channels: int | None = None, 
                        temperture: int = 10000, 
                        flatten: bool = True, 
                        scale: float = 2* math.pi) -> torch.Tensor:
    # x: (B, x, y, z, ch)
    if isinstance(x, tuple):
        b, *axis, c = x
    else:
        b, *axis, c = x.shape
    channels = channels or c
    axis_space = tuple([torch.linspace(0, 1, i) for i in axis])
    axis_dim = (channels // len(axis_space)) + 1
    
    dim_t = torch.arange(axis_dim).float()
    dim_t = temperture ** (dim_t / axis_dim) # (axis_dim)
    
    axis_embed = torch.stack(torch.meshgrid(*axis_space, indexing="ij"), dim=-1) * scale # (x, y, z, 3)
    axis_embed = axis_embed.unsqueeze(-1) / dim_t # (x, y, z, 3, axis_dim)
    axis_embed[..., 0::2].sin_()
    axis_embed[..., 1::2].cos_()
    axis_embed = axis_embed.transpose(-1, -2).flatten(-2)[..., :channels] # x, y, z, channels
    if flatten:
        axis_embed = axis_embed.flatten(0, -2) # (x * y * z, channels)
    return axis_embed.unsqueeze(0) # (1, x * y * z, c or 1, x, y, z, c)

class PositionalEncoding(nn.Module):
    def __init__(self, channels: int | None = None, 
                 temperture: int = 10000, 
                 flatten: bool = True, 
                 scale: float = 2* math.pi):
        super().__init__()
        self.channels = channels
        self.temperture = temperture
        self.flatten = flatten
        self.scale = scale
        self._cache_shape = None
        self._cache = None
        
    def forward(self, x: torch.Tensor | tuple, device = torch.device("cpu")) -> torch.Tensor:
        if isinstance(x, tuple):
            xshape = x
        else:
            xshape = x.shape[1:-1]
        if xshape == self._cache_shape:
            return self._cache
        else:
            self._cache = positional_encoding(x, self.channels, self.temperture, self.flatten, self.scale).to(device)
            self._cache_shape = self._cache.shape[1:-1]
            return self._cache
    