import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .op import ResBlock, conv_nd, TimestepEmbedSequential

# ============================================
# Discrimination Network
# github:

class NLayerDiscriminator(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 model_channels: int = 32,
                 out_channels: int = 1,
                 channel_mult: tuple[int] = (1, 2, 4),
                 z_down: tuple[int] = (1, 2),
                 reduce: str = "none",
                 dim: int = 3):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.z_down = z_down

        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(conv_nd(dim, in_channels, model_channels, kernel_size = 4, padding=1, stride = 2))

        ds = 1
        for level, mult in enumerate(channel_mult):
            layer = nn.Sequential()
            ch = model_channels * mult
            if level != len(channel_mult) - 1:
                out_ch = int(model_channels * channel_mult[level + 1])
                layer.add_module(f"conv", conv_nd(dim, ch, out_ch, kernel_size = 4, padding=1, stride = 1))
                layer.add_module(f"norm", nn.BatchNorm3d(out_ch))
                layer.add_module(f"act", nn.LeakyReLU(0.2, True))
            else:
                layer.add_module(f"conv", conv_nd(dim, ch, 1, kernel_size = 4, padding=1, stride = 1))
            self.input_blocks.append(layer)
            
        if reduce == "none":
            self.reduce = lambda x: x
        elif reduce == "mean":
            self.reduce = lambda x: torch.mean(
                torch.flatten(x, start_dim=1), dim=1)
        elif reduce == "sum":
            self.reduce = lambda x: torch.sum(
                torch.flatten(x, start_dim=1), dim=1)

    def forward(self, x):
        """Standard forward."""
        for layer in self.input_blocks:
            x = layer(x)
        return self.reduce(x)
        


class ImprovedNLayerDiscriminator(nn.Module):
    """Defines a discriminator Using the idea of SN-GAN and WGAN-gp, don't use batchnorm, min input size is (2, 2, 2)"""

    def __init__(self,
                 in_channels: int = 1,
                 model_channels: int = 32,
                 out_channels: int = 1,
                 channel_mult: tuple[int] = (1, 2, 4),
                 z_down: tuple[int] = (1, 2),
                 reduce: str = "none",
                 dim: int = 3):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.z_down = z_down

        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(
            conv_nd(dim, in_channels, model_channels, 3, padding=1))

        ds = 1
        for level, mult in enumerate(channel_mult):
            ch = model_channels * mult
            layer = TimestepEmbedSequential()
            layer.add_module(f"res", ResBlock(ch, None, 0, dims=dim))
            if level != len(channel_mult) - 1:
                out_ch = int(model_channels * channel_mult[level + 1])
                layer.add_module(f"down", ResBlock(
                    ch, None, 0, out_ch, dims=dim, down=True, z_down=ds in z_down))
                ds *= 2
            self.input_blocks.append(layer)

        self.output_blocks = nn.Sequential()
        self.output_blocks.add_module("conv", conv_nd(dim, ch, 2048, 1))
        self.output_blocks.add_module("norm", nn.GroupNorm(32, 2048))
        self.output_blocks.add_module("act", nn.SiLU())
        self.output_blocks.add_module("conv2", conv_nd(dim, 2048, 1, 1))

        if reduce == "none":
            self.reduce = lambda x: x
        elif reduce == "mean":
            self.reduce = lambda x: torch.mean(
                torch.flatten(x, start_dim=1), dim=1)
        elif reduce == "sum":
            self.reduce = lambda x: torch.sum(
                torch.flatten(x, start_dim=1), dim=1)

    def forward(self, x):
        """Standard forward."""
        for layer in self.input_blocks:
            x = layer(x)
        x = self.output_blocks(x)
        return self.reduce(x)
