import os
import torch
import torch.nn.functional as F
import einops
from torch import nn
from typing import Dict, Tuple
import numpy as np

from .unet import UNetModel as unet
from .regression_head import Regression as reg_v0
from .utils import basicModel, basicParallel
from .op import *

def build_basic_model():
    module = nn.Sequential()
    module.add_module("unet", unet(
                                image_size=(16, 128, 128),
                                in_channels=1,
                                model_channels=32,
                                out_channels=32,
                                num_res_blocks=1,
                                attention_resolutions=(4, 8),
                                dropout=0.1,
                                channel_mult=(1, 2, 4, 8),
                                z_down = (1, 2, 4),
                                dims = 3,
                                num_heads = 4,
                                time_embed=None,
                                use_checkpoint=False))
    module.add_module("reg", reg_v0())
    return module


# ============================================
# Regression Network
#

class Regression(basicModel):
    """
        Regression model for the UNet output
    """
    def __init__(self, 
                 in_size: Tuple[int] =(16, 128, 128),
                 out_size: Tuple[int] = (4, 32, 32),
                 in_channels: int = 32,
                 out_channels: Tuple[int] | int = 8,
                 layers: int = 2,
                 dims: int = 3,
                 use_checkpoint: bool = False,
                 after_process: bool = True
                 ):
        super(Regression, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dims = dims
        self.in_channels = in_channels
        if isinstance(out_channels, int):
            self.out_channels = [out_channels]
        else:
            self.out_channels = out_channels
        self.use_checkpoint = use_checkpoint
        self.after_process = after_process

        size = np.asarray(self.in_size)
        self.in_blocks = nn.Sequential()
        ch = in_channels
        for i in range(layers):
            if i != layers - 1:
                size = np.max(np.stack([size // 2, out_size]), axis=0)
            else:
                size = out_size

            self.in_blocks.add_module(f"conv{i}", conv_nd(
                dims, ch, ch * 2, kernel_size=3, padding=1, padding_mode='replicate'))
            self.in_blocks.add_module(f"act{i}", nn.SiLU())
            self.in_blocks.add_module(f"pool{i}", avg_adt_pool_nd(dims, size))

            ch *= 2

        self.out_blocks = nn.ModuleList()
        for i, out_ch in enumerate(self.out_channels):
            out_blocks = nn.Sequential()
            out_blocks.add_module(f"conv{i}_0", conv_nd(dims, ch, ch // 2, kernel_size=1))
            out_blocks.add_module(f"act{i}", nn.SiLU())
            out_blocks.add_module(f"conv{i}_1", conv_nd(dims, ch // 2, out_ch, kernel_size=1))
            self.out_blocks.append(out_blocks)
            
    def forward(self, x):
        """
        Apply the block to a Tensor.

        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.use_checkpoint:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)

    def _forward(self, x):
        x = self.in_blocks(x)
        out = []
        for i, out_blcok in enumerate(self.out_blocks):
            y = out_blcok(x)
            if self.after_process:
                if len(self.out_blocks) == 1:
                    x = einops.rearrange(x, "b (E C) Z X Y -> b Z X Y E C", C=4)
                    x = torch.cat([x[..., (0,)], x[..., 1:].sigmoid()* 1.2 - 0.1], dim=-1).contiguous()
                else:
                    if i == 0:
                        # This is for position regression, channels should be 3
                        y = y.sigmoid() * 1.2 - 0.1
                    if i == 1:
                        # This is for classification, channels should be n + 1
                        y = y.softmax(dim = 1)
            out.append(y)
    
        if len(self.out_blocks) == 1:
            return out[0]
        else:
            return out

# ============================================
# UNet
# Adapted from github repo: https://github.com/AlexGraikos/diffusion_priors

class UNetModel(basicModel):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        time_embed: None | int = 4,
        reference_channels: None | int = None,
        reference_mode="concat",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.ref_channels = reference_channels
        self.ref_mode = reference_mode
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.time_embed = time_embed
        if self.time_embed is not None:
            time_embed_dim = model_channels * self.time_embed
            self.time_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        else:
            time_embed_dim = None

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)

        # Turn input channels to model channels, if reference channels are used, add a MixBlock.

        input_block = TimestepEmbedSequential()
        input_block.add_module("conv", conv_nd(
            dims, in_channels, ch, 3, padding=1))
        if reference_channels is not None:
            input_block.add_module("mix", MixBlock(
                ch, reference_channels, ch, mode=self.ref_mode, dims=dims))
        self.input_blocks = nn.ModuleList([input_block])

        # the start of downsampling

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for n in range(num_res_blocks):
                layers = TimestepEmbedSequential()
                layers.add_module(f"res{n}",
                                  ResBlock(
                                      ch,
                                      time_embed_dim,
                                      dropout,
                                      out_channels=int(mult * model_channels),
                                      dims=dims,
                                      use_checkpoint=use_checkpoint,
                                      use_scale_shift_norm=use_scale_shift_norm,
                                  )
                                  )
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.add_module(f"attn{n}",
                                      AttentionBlock(
                                          ch,
                                          use_checkpoint=use_checkpoint,
                                          num_heads=num_heads,
                                          num_head_channels=num_head_channels,
                                          use_new_attention_order=use_new_attention_order,
                                      )
                                      )
                self.input_blocks.append(layers)
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                layer = TimestepEmbedSequential()
                if resblock_updown:
                    layer.add_module("res",
                                     ResBlock(
                                         ch,
                                         time_embed_dim,
                                         dropout,
                                         out_channels=out_ch,
                                         dims=dims,
                                         use_checkpoint=use_checkpoint,
                                         use_scale_shift_norm=use_scale_shift_norm,
                                         down=True,
                                     )
                                     )
                else:
                    layer.add_module("down",
                                     Downsample(
                                         ch, conv_resample, dims=dims, out_channels=out_ch
                                     )
                                     )
                self.input_blocks.append(layer)
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # bottom layer of the network ( R - A - R )

        self.middle_block = TimestepEmbedSequential()
        self.middle_block.add_module("res0",
                                     ResBlock(
                                         ch,
                                         time_embed_dim,
                                         dropout,
                                         dims=dims,
                                         use_checkpoint=use_checkpoint,
                                         use_scale_shift_norm=use_scale_shift_norm)
                                     )
        self.middle_block.add_module("attn",
                                     AttentionBlock(
                                         ch,
                                         use_checkpoint=use_checkpoint,
                                         num_heads=num_heads,
                                         num_head_channels=num_head_channels,
                                         use_new_attention_order=use_new_attention_order)
                                     )
        self.middle_block.add_module("res1",
                                     ResBlock(
                                         ch,
                                         time_embed_dim,
                                         dropout,
                                         dims=dims,
                                         use_checkpoint=use_checkpoint,
                                         use_scale_shift_norm=use_scale_shift_norm)
                                     )
        self._feature_size += ch

        # the start of upsampling
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = TimestepEmbedSequential()
                layers.add_module(f"res{i}",
                                  ResBlock(
                                      ch + ich,
                                      time_embed_dim,
                                      dropout,
                                      out_channels=int(model_channels * mult),
                                      dims=dims,
                                      use_checkpoint=use_checkpoint,
                                      use_scale_shift_norm=use_scale_shift_norm,
                                  )
                                  )
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.add_module(f"attn{i}",
                                      AttentionBlock(
                                          ch,
                                          use_checkpoint=use_checkpoint,
                                          num_heads=num_heads_upsample,
                                          num_head_channels=num_head_channels,
                                          use_new_attention_order=use_new_attention_order,
                                      )
                                      )
                if level and i == num_res_blocks:
                    out_ch = ch
                    if resblock_updown:
                        layers.add_module("up",
                                          ResBlock(
                                              ch,
                                              time_embed_dim,
                                              dropout,
                                              out_channels=out_ch,
                                              dims=dims,
                                              use_checkpoint=use_checkpoint,
                                              use_scale_shift_norm=use_scale_shift_norm,
                                              up=True,
                                          )
                                          )
                    else:
                        layers.add_module("up",
                                          Upsample(
                                              ch, conv_resample, dims=dims, out_channels=out_ch)
                                          )
                    ds //= 2
                self.output_blocks.append(layers)
                self._feature_size += ch

        # output convolution

        self.out = nn.Sequential()
        self.out.add_module("norm", normalization(ch))
        self.out.add_module("act", nn.SiLU())
        self.out.add_module("conv", zero_module(
            conv_nd(dims, ch, out_channels, 3, padding=1)))

    def forward(self, x, timesteps=None, y=None, ref=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        if self.time_embed is not None:
            emb = self.time_embed(timestep_embedding(
                timesteps, self.model_channels))

            if self.num_classes is not None:
                assert y.shape == (x.shape[0],)
                emb = emb + self.label_emb(y)

        else:
            emb = None
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, ref)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(
            low_res, (new_height, new_width), mode="bilinear")
        x = torch.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class EncoderUNetModel(basicModel):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        time_embed: None | int = 4,
        pool="adaptive",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.time_embed = time_embed

        if self.time_embed is not None:
            time_embed_dim = model_channels * self.time_embed
            self.time_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        else:
            time_embed_dim = None

        ch = int(channel_mult[0] * model_channels)
        
        input_block = TimestepEmbedSequential()
        input_block.add_module("conv", conv_nd(
            dims, in_channels, ch, 3, padding=1))
        self.input_blocks = nn.ModuleList([input_block])
        
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for n in range(num_res_blocks):
                layers = TimestepEmbedSequential()
                layers.add_module(f"res{n}",
                                  ResBlock(
                                      ch,
                                      time_embed_dim,
                                      dropout,
                                      out_channels=int(mult * model_channels),
                                      dims=dims,
                                      use_checkpoint=use_checkpoint,
                                      use_scale_shift_norm=use_scale_shift_norm,
                                  )
                                  )
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.add_module(f"attn{n}",
                                      AttentionBlock(
                                          ch,
                                          use_checkpoint=use_checkpoint,
                                          num_heads=num_heads,
                                          num_head_channels=num_head_channels,
                                          use_new_attention_order=use_new_attention_order,
                                      )
                                      )
                self.input_blocks.append(layers)
                self._feature_size += ch
                input_block_chans.append(ch)
                
            if level != len(channel_mult) - 1:
                out_ch = ch
                layer = TimestepEmbedSequential()
                if resblock_updown:
                    layer.add_module("res",
                                     ResBlock(
                                         ch,
                                         time_embed_dim,
                                         dropout,
                                         out_channels=out_ch,
                                         dims=dims,
                                         use_checkpoint=use_checkpoint,
                                         use_scale_shift_norm=use_scale_shift_norm,
                                         down=True)
                                     )
                else:
                    layer.add_module("down",
                                     Downsample(
                                         ch, conv_resample, dims=dims, out_channels=out_ch)
                                     )
                self.input_blocks.append(layer)
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential()
        self.middle_block.add_module("res0",
                                     ResBlock(
                                         ch,
                                         time_embed_dim,
                                         dropout,
                                         dims=dims,
                                         use_checkpoint=use_checkpoint,
                                         use_scale_shift_norm=use_scale_shift_norm)
                                     )
        self.middle_block.add_module("attn",
                                     AttentionBlock(
                                         ch,
                                         use_checkpoint=use_checkpoint,
                                         num_heads=num_heads,
                                         num_head_channels=num_head_channels,
                                         use_new_attention_order=use_new_attention_order)
                                     )
        self.middle_block.add_module("res1",
                                     ResBlock(
                                         ch,
                                         time_embed_dim,
                                         dropout,
                                         dims=dims,
                                         use_checkpoint=use_checkpoint,
                                         use_scale_shift_norm=use_scale_shift_norm)
                                     )
        
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                avg_adt_pool_nd(dims, (1, ) * dims),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
            
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "none":
            self.out = conv_nd(dims, ch, out_channels, 1)
            
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def forward(self, x, timesteps = None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        if self.time_embed is not None:
            emb = self.time_embed(timestep_embedding(
                timesteps, self.model_channels))
        else:
            emb = None
            
        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = torch.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)


# ============================================
# Discrimination Network
# github:

class NLayerDiscriminator(basicModel):
    """Defines a discriminator
    Using the idea of SN-GAN and WGAN-gp, don't use batchnorm, min input size is (2, 2, 2)"""

    def __init__(self,
                 in_channels=256,
                 model_channels=32,
                 channel_mult: tuple = (1, 2, 4),
                 max_down_mult: tuple = (4, 8, 8),
                 kernel_size=3,
                 padding=1,
                 reduce="none",
                 dim=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        self.dim = dim
        channels = (in_channels, ) + \
            tuple(map(lambda x: x * model_channels, channel_mult))
        self.blocks = nn.ModuleList()
        self.down_stride = []
        down_num = np.log2(max_down_mult).astype(int)
        for i in range(len(channel_mult)):
            self.down_stride.append([2 if i < n else 1 for n in down_num])
            
        for in_ch, out_ch, stride in zip(channels[:-1], channels[1:], self.down_stride):
            self.blocks.append(nn.Sequential(
                conv_nd(self.dim, in_channels=in_ch, out_channels=out_ch,
                        kernel_size=kernel_size, padding=padding, stride=stride),
                nn.GroupNorm(32, out_ch),
                nn.SiLU()))

        self.out_block = nn.Sequential(
            conv_nd(self.dim, in_channels=channels[-1], out_channels=model_channels,
                    kernel_size = kernel_size, padding = padding, padding_mode = "replicate"),
            nn.SiLU(),
            conv_nd(self.dim, in_channels=model_channels,
                    out_channels=1, kernel_size=1)
        )
        if reduce == "none":
            self.reduce = lambda x: x
        elif reduce == "mean":
            self.reduce = lambda x: torch.mean(torch.flatten(x, start_dim = 1), dim=1)
        elif reduce == "sum":
            self.reduce = lambda x: torch.sum(torch.flatten(x, start_dim = 1), dim=1)

    def forward(self, x):
        """Standard forward."""
        B, C, *S = x.shape

        for module in self.blocks:
            x = module(x)

        x = self.out_block(x)
        return self.reduce(x)
