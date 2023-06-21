import os
import torch
import torch.nn.functional as F
import einops
from torch import nn
from typing import Dict, Tuple
import numpy as np

from .unet import UNetModel as unet
from .disc import NLayerDiscriminator as disc_v0
from .regression_head import Regression as reg_v0
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

def build_cyc_model(disc = False):
    cycleNet = unet(image_size=(16, 128, 128),
                               in_channels=1,
                               model_channels=32,
                               out_channels=1,
                               num_res_blocks=2,
                               attention_resolutions=tuple(),
                               dropout=0.1,
                               channel_mult=(1, 2, 4, 4),
                               dims=3,
                               time_embed=None)
    if disc:
        discNet = disc_v0(in_channels=1, model_channels=32, out_channels=1, channel_mult=(1, 2, 4, 8), reduce="mean")
        return cycleNet, discNet
    else:
        return cycleNet