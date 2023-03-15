import torch
import torch.nn as nn
from .basic import SingleConv


class Fire(nn.Module):

    def __init__(self, in_channels, squeeze_channels,
                 expand1x1_channels, expand3x3_channels,
                 use_bypass=False):
        super(Fire, self).__init__()
        self.use_bypass = use_bypass
        self.in_channels = in_channels
        self.relu = nn.ReLU(inplace=True)
        self.squeeze = SingleConv(
            in_channels, squeeze_channels, kernel_size=1, order="cbr")
        self.expand1x1 = SingleConv(
            squeeze_channels, expand1x1_channels, kernel_size=1, order="cb")
        self.expand3x3 = SingleConv(
            squeeze_channels, expand3x3_channels, kernel_size=3, order="cb", padding=1)

    def forward(self, x):
        # squeeze
        out = self.squeeze(x)
        # expand
        out1 = self.expand1x1(out)
        out2 = self.expand3x3(out)

        out = torch.cat([out1, out2], 1)

        if self.use_bypass:
            out += x

        out = self.relu(out)

        return out
