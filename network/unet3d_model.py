import torch.autograd
from torch import nn
from torch.nn import functional as F
from .basic import basicModel
from .unet_part import *
from .parts import *

class UNet3D(basicModel):
    def __init__(self, img_channels = 1, hidden_channels = 32, inp_size = (16,128,128), out_size = (4, 32, 32), *args, **kwargs):
        super(UNet3D, self).__init__()
        self.inp_size = inp_size
        self.out_size = out_size
        self.inc = DoubleConv(in_channels=img_channels, out_channels=hidden_channels, encoder=True)
        self.down1 = Down(hidden_channels, 2 * hidden_channels, all_dim=0)
        self.down2 = Down(2 * hidden_channels, 4 * hidden_channels, all_dim=0)
        self.down3 = Down(4 * hidden_channels, 8 * hidden_channels)
        self.down4 = Down(8 * hidden_channels, 8 * hidden_channels)
        self.up1 = Up(16 * hidden_channels, 4 * hidden_channels)
        self.up2 = Up(8 * hidden_channels, 2 * hidden_channels)
        self.up3 = Up(4 * hidden_channels, hidden_channels)
        self.up4 = Up(2 * hidden_channels, hidden_channels)
        self.out = Out(in_channels=hidden_channels)

    def forward(self, x):    # (batch_size, 1, 16, 128, 128)
        x1 = self.inc(x)     # (batch_size, 32, 16, 128, 128)
        x2 = self.down1(x1)  # (batch_size, 64, 16, 64, 64)
        x3 = self.down2(x2)  # (batch_size, 128, 16, 32, 32)
        x4 = self.down3(x3)  # (batch_size, 256, 8, 16, 16)
        x5 = self.down4(x4)  # (batch_size, 256, 4, 8, 8)
        x = self.up1(x4, x5) # (batch_size, 256+256 -> 128, 8, 16, 16)
        x = self.up2(x3, x)  # (batch_size, 64, 16, 32, 32)
        x = self.up3(x2, x)  # (batch_size, 32, 16, 64, 64)
        x = self.up4(x1, x)  # (batch_size, 32, 16, 128, 128)
        x = F.interpolate(x, (self.out_size[0], self.out_size[1] * 4, self.out_size[2] * 4), mode='trilinear', align_corners=True)
        # (batch_size, 32, output_z, 128, 128)
        x = self.out(x)  # (batch_size, 8, 9, 32, 32)
        x = x.permute([0, 3, 4, 2, 1]).contiguous()
        # (batch_size, 32, 32, output_z, 8)
        shape = x.shape
        x = x.view(-1, 4)
        x[..., :3] = torch.sigmoid(x[..., :3])
        x = x.view(shape)
        return x

class TransUNet3D(basicModel):
    def __init__(self, img_channels = 1, hidden_channels = 32, inp_size = (16,128,128), out_size = (4, 32, 32), out_feature = False, *args, **kwargs):
        super(TransUNet3D, self).__init__()
        self.inp_size = inp_size
        self.vit_size = (inp_size[0] // 4, inp_size[1] // 16, inp_size[2] // 16)
        self.out_size = out_size
        self.inc = DoubleConv(in_channels=img_channels, out_channels=hidden_channels, encoder=True)
        self.down1 = Down(hidden_channels, 2 * hidden_channels, all_dim=0) # 這兩次down不會影響Z軸的維度
        self.down2 = Down(2 * hidden_channels, 4 * hidden_channels, all_dim=0) # 這兩次down不會影響Z軸的維度
        self.down3 = Down(4 * hidden_channels, 8 * hidden_channels) # Z軸的維度 // 2
        self.down4 = Down(8 * hidden_channels, 8 * hidden_channels) # Z軸的維度 // 2
        self.vit = ViT(8 * hidden_channels, 8 * hidden_channels, img_size=self.vit_size)
        self.up1 = Up(16 * hidden_channels, 4 * hidden_channels)
        self.up2 = Up(8 * hidden_channels, 2 * hidden_channels)
        self.up3 = Up(4 * hidden_channels, hidden_channels)
        self.up4 = Up(2 * hidden_channels, hidden_channels)
        self.out = Out(in_channels=hidden_channels)
        self.out_feature = out_feature

    def forward(self, x):    # (batch_size, 1, 10, 128, 128)
        x1 = self.inc(x)     # (batch_size, 32, 10, 128, 128)
        x2 = self.down1(x1)  # (batch_size, 64, 10, 64, 64)
        x3 = self.down2(x2)  # (batch_size, 128, 10, 32, 32)
        feature = x3
        x4 = self.down3(x3)  # (batch_size, 256, 5, 16, 16)
        x5 = self.down4(x4)  # (batch_size, 256, 2, 8, 8)
        x5 = self.vit(x5)
        x = self.up1(x4, x5)
        x = self.up2(x3, x)  # (batch_size, 64, 5, 32, 32)
        # torch.Size([batch_size, 64, Z, 32, 32])
        x = self.up3(x2, x)
        x = self.up4(x1, x)  # (batch_size, 32, 5, 128, 128)
        x = F.interpolate(x, (self.out_size[0], self.out_size[1] * 4, self.out_size[2] * 4), mode='trilinear', align_corners=True)
        # (batch_size, 32, output_z, 128, 128)
        x = self.out(x)  # (batch_size, 8, 9, 32, 32)
        x = x.permute([0, 3, 4, 2, 1]).contiguous()
        # (batch_size, 32, 32, output_z, 8)
        shape = x.shape
        x = x.view(-1, 4)
        x[..., :3] = torch.sigmoid(x[..., :3])
        x = x.view(shape)
        if self.out_feature:
            return x, feature
        else:
            return x

