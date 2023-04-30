import torch
from torch import nn
from torch.nn import functional as F
from functools import partial

def conv3d(in_channels, out_channels, kernel_size, bias, padding, stride = 1):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, padding_mode='replicate', bias=bias, stride = stride)

def create_conv(in_channels, out_channels, kernel_size: int = 3, order: str = "cr", padding: tuple | int = 0, num_groups: int | None = None, stride: tuple | int = 1):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add replicate-padding added to all three sides of the input
    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(
                ('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding, stride=stride)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(
                num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(
                f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules

class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple):
    """

    def __init__(self, in_channels, out_channels, kernel_size: int = 3, order: str = "cr", padding: tuple | int = 0, num_groups: int | None = None, stride: tuple | int = 1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups = num_groups, padding = padding, stride= stride):
            self.add_module(name, module)
            
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


# =============================================================================
# Up sampling parts: Interpolate / TransposeConv / NoUpsampling

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
        return F.interpolate(x, size=size, mode=mode)

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
    
# =============================================================================
# Super-Resolution Network: VapSR
# Github: https://github.com/zhoumumu/VapSR

class SwitchAttention(nn.Module):
    def __init__(self, channels, dim = 3):
        super().__init__()
        if dim == 2:
            conv = nn.Conv2d
        elif dim == 3:
            conv = nn.Conv3d
            
        self.pointwise = conv(in_channels = channels, out_channels = channels, kernel_size = 1)
        self.depthwise = conv(in_channels = channels, out_channels = channels, kernel_size = 5, padding = 2, groups=channels)
        self.depthwise_dilated = conv(in_channels = channels, out_channels = channels, kernel_size = 5, stride = 1, padding=6, groups = channels, dilation=3)

    def forward(self, x):
        u = x.clone()
        attn = self.pointwise(x)
        attn = self.depthwise(attn)
        attn = self.depthwise_dilated(attn)
        return u * attn

class VAB(nn.Module):
    """The basic module used in VapSR.
    step 1: 1x1x1 Conv
    step 2: attn (1x1x1 Conv + Depthwise Conv + 5x5x5 Depthwise dilated Conv)
    step 3: attn * step 1
    step 4: 1x1x1 Conv
    step 5: add input
    """
    def __init__(self, in_channels, lattent):
        super().__init__()
        self.proj1 = nn.Conv3d(in_channels = in_channels, out_channels = lattent, kernel_size = 1)
        self.act = nn.GELU()
        self.attn = SwitchAttention(lattent)
        self.proj2 = nn.Conv3d(in_channels = lattent, out_channels = in_channels, kernel_size = 1)
        self.norm = nn.LayerNorm(in_channels)
        # default_init_weights([self.pixel_norm], 0.1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj1(x)
        x = self.act(x)
        x = self.attn(x)
        x = self.proj2(x)
        x = x + shorcut

        x = x.permute(0, 2, 3, 4, 1) # B, D, H, W, C
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous() # B, C, D, H, W

        return x

class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    ref: https://github.com/gap370/pixelshuffle3d
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)

def pixelshuffle(in_channels: torch.Tensor, out_channels: torch.Tensor, lattent: int = 256, scale: int = 2) -> nn.Sequential:
    """Upsampling by pixel shuffle
    :param in_channels: input channels
    :param out_channels: output channels
    :param lattent: used as the middle layer in order to reduce parameters, original paper: 64
    :param scale: the scale factor, the lattent channels should be divided by scale ** 3
    :return: nn.Sequential
    
    Example:
      >>> a = torch.randn(1, 64, 16, 16, 16)
      >>> b = pixelshuffle(64, 16)
      >>> b(a).shape
    Output:
      >>> torch.Size([1, 16, 32, 32, 32])
    """
    
    upconv1 = nn.Conv3d(in_channels = in_channels, out_channels = lattent, kernel_size = 3, stride = 1, padding = 1)
    pixel_shuffle = PixelShuffle3d(scale)
    upconv2 = nn.Conv3d(in_channels = lattent // (scale ** 3), out_channels = out_channels * (scale ** 3), kernel_size = 3, stride = 1, padding = 1)
    lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    return nn.Sequential(*[upconv1, pixel_shuffle, lrelu, upconv2, pixel_shuffle])

# ============================================
# CBAM module
# CSDN: https://blog.csdn.net/weixin_38241876/article/details/109853433

class ChannelAttentionModule(nn.Module):
    def __init__(self, channels, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        #使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
 
        self.shared_MLP = nn.Sequential(
            nn.Conv3d(in_channels = channels, out_channels = channels // ratio, kernel_size = 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels = channels // ratio, out_channels = channels, kernel_size = 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)
 
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv3d(in_channels = 2, out_channels = 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        #map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv(out))
        return out
 
class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channels)
        self.spatial_attention = SpatialAttentionModule()
 
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
    
# ============================================
# 3D image Self-Attention module
# ref: https://github.com/heykeetae/Self-Attention-GAN

class FullAttention(nn.Module):
    """ Self attention Layer
    Use 1x Conv to get the Query, Key, Value.
    :param in_channels: input channels
    :param dim: dimension, 2 for 2D image, 3 for 3D image
    :param ratio: the ratio to reduce the channels
    :param activation: activation function
    
    Example:
        >>> a = torch.randn(1, 64, 16, 16, 16)
        >>> b = FullAttention(64)
        >>> b(a).shape
    Output:
        >>> torch.Size([1, 64, 16, 16, 16])"""
    def __init__(self,in_channels: int , dim: int = 3, ratio: int = 8, activation: str = "relu"):
        super(FullAttention,self).__init__()
        self.chanel_in = in_channels
        self.activation = activation
        
        if dim == 3:
            conv = nn.Conv3d
        elif dim == 2:
            conv = nn.Conv2d
        self.QConv = conv(in_channels = in_channels , out_channels = in_channels//ratio , kernel_size= 1)
        self.KConv = conv(in_channels = in_channels , out_channels = in_channels//ratio , kernel_size= 1)
        self.VConv = conv(in_channels = in_channels , out_channels = in_channels , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)
        
    def forward(self,x: torch.Tensor, attn_out: bool = False) -> torch.Tensor:
        """forward attention
        :params x: input feature maps B x C x (D) x W x H
        :params attn_out: if True, return attention map
        :return: attention value + input feature
        """
        B, C, *S = x.shape
        Q  = self.QConv(x).flatten(2).permute(0,2,1) # B x N x C
        K =  self.KConv(x).flatten(2) # B x C x N
        V =  self.VConv(x).flatten(2) # B x C x N
        energy =  torch.bmm(Q,K) # transpose check
        attention = self.softmax(energy) # B x N x N 
        out = torch.bmm(V, attention.permute(0,2,1))
        out = out.view(B, C, *S)
        out = self.gamma * out + x
        
        if attn_out:
            return out,attention
        else:
            return out