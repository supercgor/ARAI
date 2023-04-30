import os
import torch 
import torch.nn.functional as F
import einops
from torch import nn
from typing import Dict

from .utils import basicModel, basicParallel
from .op import *

class basichead(basicModel):
    def __init__(self, in_channels=256, out_channels = 8, kernel_size = 3):
        super(basichead, self).__init__()
        self.head = nn.Sequential(
            SingleConv(in_channels= in_channels     , out_channels= in_channels // 2, kernel_size= kernel_size, order= "cl", padding = (kernel_size - 1) // 2),
            SingleConv(in_channels= in_channels // 2, out_channels= in_channels // 4, kernel_size= kernel_size, order= "cl", padding = (kernel_size - 1) // 2),
            SingleConv(in_channels= in_channels // 4, out_channels= in_channels // 4, kernel_size= kernel_size, order= "cl", padding = (kernel_size - 1) // 2),
            SingleConv(in_channels= in_channels // 4, out_channels= in_channels // 4, kernel_size= kernel_size, order= "cl", padding = (kernel_size - 1) // 2),
            SingleConv(in_channels= in_channels // 4, out_channels= in_channels // 4, kernel_size= 1          , order= "cr", padding = 0),
            SingleConv(in_channels= in_channels // 4, out_channels= in_channels // 4, kernel_size= 1          , order= "cr", padding = 0),
            nn.Conv3d(in_channels= in_channels // 4 , out_channels= out_channels    , kernel_size= 1          , padding = 0)
        )

    def forward(self, x):
        x = self.head(x)
        return x

# ============================================
# Super-Resolution Network: VapSR
# Github: https://github.com/zhoumumu/VapSR
class VapSR(basicModel):
    def __init__(self, in_channels = 256, out_channels = 8, lattent = 64, scale: int = 4, VAB_number=16, attn_channels=80, groups=1):
        super(VapSR, self).__init__()
        self.conv1 = nn.Conv3d(in_channels = in_channels, out_channels = lattent, kernel_size = 3, stride = 1, padding = 1)
        self.body = nn.ModuleList()
        for _ in range(VAB_number):
            self.body.append(VAB(lattent, attn_channels))
        self.body = nn.Sequential(*self.body)
        
        self.conv2 = nn.Conv3d(in_channels = lattent, out_channels = lattent, kernel_size = 3, stride = 1, padding = 1, groups=groups) #conv_groups=2 for VapSR-S

        # upsample
        self.upsampler = pixelshuffle(in_channels = lattent, out_channels = out_channels, scale = scale // 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.body(x)
        x = x + self.conv2(x)
        x = self.upsampler(x)
        return x
    
# ============================================
# Feature Maxing Network

class FMN(basicModel):
    def __init__(self, in_channels = (128, 64, 32, 32), out_channels = 256, box_shape = (8, 32, 32)):
        super().__init__()
        self.box_shape = box_shape
        self.pool1 = nn.AdaptiveAvgPool3d(box_shape)
        self.pool2 = nn.AdaptiveMaxPool3d(box_shape)
        self.CBAM1 = CBAM(channels= 2 * sum(in_channels))
        self.CBAM2 = CBAM(channels= out_channels // 2)
        self.CBAM3 = CBAM(channels= out_channels)
        
        self.ConvAttn1 = nn.Sequential(
            SingleConv(in_channels= 2 * sum(in_channels), out_channels= out_channels // 2, kernel_size= 1, order= "cgl", num_groups= 4),
            FullAttention(in_channels= out_channels // 2)
        )
        self.ConvAttn2 = nn.Sequential(
            SingleConv(in_channels= out_channels // 2, out_channels= out_channels, kernel_size= 3, padding = 1, order= "cgl", num_groups= 4),
            FullAttention(in_channels= out_channels)
        )
        self.refine = SingleConv(in_channels= out_channels, out_channels= out_channels, kernel_size= 1, order= "cgl", num_groups= 4)
        
    def forward(self, features: Dict[str, torch.Tensor]):
        x = []
        for key, value in features.items():
            B, C, *S = value.shape
            if all([s >= self.box_shape[i] for i, s in enumerate(S)]):
                x.append(self.pool1(value))
                x.append(self.pool2(value))
            else:
                x.append(F.interpolate(value, size = self.box_shape, mode = 'trilinear'))
                x.append(F.interpolate(value, size = self.box_shape, mode = 'nearest'))
        x = torch.cat(x, dim = 1)
        x = self.CBAM1(x)
        x = self.ConvAttn1(x)
        x = self.CBAM2(x)
        x = self.ConvAttn2(x)
        x = self.CBAM3(x)
        x = self.refine(x)

        return x

# ============================================
# Feature Extracting Network

class UNet3D(basicModel):
    def __init__(self, in_channels = 1, hidden_channels = 32, out = ["p1", "p2", "p3", "p4"]):
        super(UNet3D, self).__init__()
        self.inc = DoubleConv(in_channels=in_channels, out_channels=hidden_channels, encoder=True)
        self.down1 = Down(hidden_channels, 2 * hidden_channels, all_dim=0)
        self.down2 = Down(2 * hidden_channels, 4 * hidden_channels, all_dim=0)
        self.down3 = Down(4 * hidden_channels, 8 * hidden_channels)
        self.down4 = Down(8 * hidden_channels, 8 * hidden_channels)
        self.up1 = Up(16 * hidden_channels, 4 * hidden_channels)
        self.up2 = Up(8 * hidden_channels, 2 * hidden_channels)
        self.up3 = Up(4 * hidden_channels, hidden_channels)
        self.up4 = Up(2 * hidden_channels, hidden_channels)
        self.out = out

    def forward(self, x):    # (batch_size, 1, 16, 128, 128)
        x1 = self.inc(x)     # (batch_size, 32, 16, 128, 128)
        x2 = self.down1(x1)  # (batch_size, 64, 16, 64, 64)
        x3 = self.down2(x2)  # (batch_size, 128, 16, 32, 32)
        x4 = self.down3(x3)  # (batch_size, 256, 8, 16, 16)
        x5 = self.down4(x4)  # (batch_size, 256, 4, 8, 8)
        x4 = self.up1(x4, x5) # (batch_size, 256+256 -> 128, 8, 16, 16)
        x3 = self.up2(x3, x4)  # (batch_size, 64, 16, 32, 32)
        x2 = self.up3(x2, x3)  # (batch_size, 32, 16, 64, 64)
        x1 = self.up4(x1, x2)  # (batch_size, 32, 16, 128, 128)
        out = {"p1": x1, "p2": x2, "p3": x3, "p4": x4, "p5": x5}
        return {key: value for key, value in out.items() if key in self.out}
       
# ============================================
# Combine Network

class CombineModel(basicModel):
    def __init__(self, fea: nn.Module = UNet3D, neck: nn.Module = FMN, head: nn.Module = basichead, fea_setting: Dict = {}, neck_setting: Dict = {}, head_setting: Dict = {}):
        super().__init__()
        self.fea = fea(**fea_setting)
        self.neck = neck(**neck_setting)
        self.head = head(**head_setting)
        self.register_buffer("out_scale", torch.tensor([1.2, 1.2, 1.2]))
        self.register_buffer("out_offset", torch.tensor([-0.1, -0.1, -0.1]))
    
    def forward(self, x: torch.Tensor):
        # Tensor ( B * 1 * 16 * 128 * 128)
        x = self.fea(x)
        # Dict[str, tensor]
        x = self.neck(x)
        # Tensor ( B * 256 * 8 * 32 * 32)
        x = self.head(x)
        # Tensor ( B * 8 * 8 * 32 * 32)
        
        # ========================================================
        # Turn the output into confidence and offset (z, x, y) format
        # There would be a little leak for offset in order to let it more easier to fit
        
        x = einops.rearrange(x, "b (c e) d h w -> b d h w c e", e = 4)
        c, o = x.split([1,3], dim = -1)
        o = torch.sigmoid(o) * self.out_scale + self.out_offset
        x = torch.cat([c,o], dim = -1)
        
        return x
    
    def parallel(self, devices_ids):
        self.fea = basicParallel(self.fea, device_ids = devices_ids)
        self.neck = basicParallel(self.neck, device_ids = devices_ids)
        self.head = basicParallel(self.head, device_ids = devices_ids)
        
    def load(self, paths: Dict[str, str], pretrained = True):
        
        match_list = []
        for key, path in paths.items():
            if os.path.isdir(path):
                continue
            match_list.extend(self.get_submodule(key).load(path, pretrained = pretrained))
        return match_list
    
    def save(self, path: str, name: str | Dict[str, str] = ""):
        """Should have '.pkl' in name"""
        self.fea.save(f"{path}/fea_{name}")
        self.neck.save(f"{path}/neck_{name}")
        self.head.save(f"{path}/head_{name}")
