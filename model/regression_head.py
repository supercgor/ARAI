import os
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import numpy as np

from .utils import basicModel
from .op import *

class Regression(basicModel):
    """
        Regression model for the UNet output
    """
    def __init__(self, 
                 in_size: tuple[int] =(16, 128, 128),
                 out_size: tuple[int] = (4, 32, 32),
                 in_channels: int = 32,
                 out_channels: tuple[int] | int = 8,
                 layers: int = 2,
                 dims: int = 3,
                 use_checkpoint: bool = False,
                 after_process: bool = True
                 ):
        super().__init__()
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

            self.in_blocks.add_module(f"conv{i}", conv_nd(dims, ch, ch * 2, kernel_size=3, padding=1, padding_mode='replicate'))
            self.in_blocks.add_module(f"act{i}", nn.SiLU())
            self.in_blocks.add_module(f"pool{i}", max_adt_pool_nd(dims, size))

            ch *= 2

        self.head = nn.Sequential()
        self.head.add_module(f"conv{i}_0", conv_nd(dims, ch, ch, kernel_size=1))
        self.head.add_module(f"act{i}", nn.LeakyReLU())
        self.head.add_module(f"conv{i}_1", conv_nd(dims, ch, 8, kernel_size=1))
            
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
        x = rearrange(x, "b (E C) Z X Y -> b Z X Y E C", C=4)
        x = torch.cat([x[..., (0,)], x[..., 1:].sigmoid()* 1.2 - 0.1], dim=-1).contiguous()

        
class Regression_v1(basicModel):
    """
        Regression model for the UNet output, return cls and reg. reg is already sigmoided and cls is not softmaxed.
    """
    def __init__(self, 
                 in_size: tuple[int] =(16, 128, 128),
                 out_size: tuple[int] = (4, 32, 32),
                 in_channels: int = 32,
                 num_cls: int = 3,
                 layers: int = 2,
                 dims: int = 3,
                 use_checkpoint: bool = False,
                 ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dims = dims
        self.in_channels = in_channels
        self.num_cls = num_cls
        self.use_checkpoint = use_checkpoint

        size = np.asarray(self.in_size)
        self.in_blocks = nn.Sequential()
        ch = in_channels
        for i in range(layers):
            if i != layers - 1:
                size = np.max(np.stack([size // 2, out_size]), axis = 0)
            else:
                size = out_size

            self.in_blocks.add_module(f"conv{i}", conv_nd(dims, ch, ch * 2, kernel_size=3, padding=1, padding_mode='replicate'))
            self.in_blocks.add_module(f"act{i}", nn.SiLU())
            self.in_blocks.add_module(f"pool{i}", max_adt_pool_nd(dims, size))
            ch *= 2

        self.cls = nn.Sequential()
        self.cls.add_module(f"conv0", conv_nd(dims, ch, ch, kernel_size=1))
        self.cls.add_module(f"act", nn.LeakyReLU())
        self.cls.add_module(f"conv1", conv_nd(dims, ch, num_cls, kernel_size=1))
        
        self.reg = nn.Sequential()
        self.reg.add_module(f"conv0", conv_nd(dims, ch, ch, kernel_size=1))
        self.reg.add_module(f"act", nn.LeakyReLU())
        self.reg.add_module(f"conv1", conv_nd(dims, ch, 3, kernel_size=1))
            
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
        cls, reg = self.cls(x), (self.reg(x).sigmoid() * 1.2) - 0.1 
        cls = rearrange(cls, "B C Z X Y -> B Z X Y C")
        reg = rearrange(reg, "B C Z X Y -> B Z X Y C")
        return torch._nested_tensor_from_tensor_list([cls, reg])