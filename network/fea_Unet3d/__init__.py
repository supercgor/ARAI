import torch.autograd
from torch import nn
from torch.nn import functional as F
from .op import *
from ..basic import basicModel

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
        self.up1 = Up(16 * hidden_channels, 4 * hidden_channels, mode = "trilinear")
        self.up2 = Up(8 * hidden_channels, 2 * hidden_channels, mode = "trilinear")
        self.up3 = Up(4 * hidden_channels, hidden_channels, mode = "trilinear")
        self.up4 = Up(2 * hidden_channels, hidden_channels, mode = "trilinear")
        self.out = Out(in_channels=hidden_channels, out_channels = 4)

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
        x = F.interpolate(x1, (self.out_size[0], self.out_size[1] * 4, self.out_size[2] * 4), mode='trilinear', align_corners=True)
        # (batch_size, 32, output_z, 128, 128)
        x = self.out(x)  # (batch_size, 8, 4, 32, 32)
        x[:, ::4,...] = F.sigmoid(x[:,::4,...])
        return x, [x5,x4,x3,x2,x1]