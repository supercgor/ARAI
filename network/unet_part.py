import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from functools import partial
from .basic import SingleConv

class OutConv(nn.Sequential):
    """
    input layer should be conv first before the down sampling
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcl', num_groups=8, padding=1):
        super(OutConv, self).__init__()
        conv1_in_channels = in_channels
        conv1_out_channels = in_channels // 2
        conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size = kernel_size, order = order, num_groups = num_groups,
                                   padding=padding))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size = kernel_size, order = order, num_groups = num_groups,
                                   padding=padding))


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+leakyReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add replicate-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcl', num_groups=8, padding = 1):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(in_channels = conv1_in_channels, 
                                   out_channels = conv1_out_channels, 
                                   kernel_size = kernel_size, 
                                   order = order, 
                                   num_groups = num_groups,
                                   padding=padding))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(in_channels = conv2_in_channels, 
                                   out_channels = conv2_out_channels, 
                                   kernel_size = kernel_size, 
                                   order = order, 
                                   num_groups = num_groups,
                                   padding=padding))


class Down(nn.Module):
    """
    construct a pooling + double conv or anything else
    all the parameters follow the previous functrion definition
    several para:
        pool type : choose the pooling type: we have max and avg
        pool_kernel_size: the pooling size
        all_dim: apply pooling to all dimension: 3d: use i, if 2d: use 0
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=2, pool_type='max', basic_module=DoubleConv, conv_layer_order='gcl',
                 num_groups=8, padding=1, all_dim=1):
        super(Down, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if all_dim == 1:
                z_size = pool_kernel_size
            else:
                z_size = 1
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=(
                    z_size, pool_kernel_size, pool_kernel_size))
            else:
                self.pooling = nn.AvgPool3d(kernel_size=(
                    z_size, pool_kernel_size, pool_kernel_size))
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=(2, 2, 2), basic_module=DoubleConv,
                 conv_layer_order='gcl', num_groups=8, mode='nearest', padding=1, upsample=True):
        super(Up, self).__init__()
        if upsample:
            if basic_module == DoubleConv:
                # if DoubleConv is the basic_module use interpolation for upsampling and concatenation joining
                self.upsampling = InterpolateUpsampling(mode=mode)
                # concat joining
                self.joining = partial(self._joining, concat=True)
            else:
                # if basic_module=ExtResNetBlock use transposed convolution upsampling and summation joining
                self.upsampling = TransposeConvUpsampling(in_channels=in_channels, out_channels=out_channels,
                                                          kernel_size=conv_kernel_size, scale_factor=scale_factor)
                # sum joining
                self.joining = partial(self._joining, concat=False)
                # adapt the number of in_channels for the ExtResNetBlock
                in_channels = out_channels
        else:
            # no upsampling
            self.upsampling = NoUpsampling()
            # concat joining
            self.joining = partial(self._joining, concat=True)

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode='nearest'):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return interpolate(x, size=size, mode=mode)


class TransposeConvUpsampling(AbstractUpsampling):
    """
    Args:
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True
    """

    def __init__(self, in_channels=None, out_channels=None, kernel_size=3, scale_factor=(2, 2, 2)):
        # make sure that the output size reverses the MaxPool3d from the corresponding encoder
        upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor,
                                      padding=1)
        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        return x


class Out(nn.Module):
    def __init__(self, in_channels=32, out_channels=8, kernel_size=3):
        super(Out, self).__init__()
        self.out = nn.Sequential(
            nn.Conv3d(in_channels, 2 * in_channels, kernel_size,
                      padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(2 * in_channels, 2 * in_channels, kernel_size,
                      padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(2 * in_channels, 4 * in_channels, kernel_size,
                      padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(4 * in_channels, 4 * in_channels, kernel_size,
                      padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(4 * in_channels, 2 * in_channels, 1,
                      padding=0, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(2 * in_channels, out_channels, 1,
                      padding=0, padding_mode='replicate'),
        )

    def forward(self, x):
        x = self.out(x)
        return x

