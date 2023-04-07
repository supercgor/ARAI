import torch
from torch import nn, einsum
import torch.nn.functional as F
from kornia.filters import filter3d
from math import log2,sqrt
from einops import rearrange
from collections import OrderedDict

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

class EqualLinear(nn.Module):
    def __init__(self, in_channels, out_channels, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))

        self.lr_mul = lr_mul

    def forward(self, x):
        return F.linear(x, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class Blur3d(nn.Module):
    def __init__(self, kernal = (1., 1.7, 1.)): # the blur kernel is f x f x f
        super().__init__()
        k = torch.Tensor(kernal)
        self.register_buffer('kernal', einsum("i, j, k -> ijk", k,k,k)[None,...])
        
    def forward(self, x):
        
        return filter3d(x, self.kernal, normalized=True)

class NoiseInjection3d(nn.Module):
    def __init__(self):
        super().__init__()
 
        self.weight = nn.Parameter(torch.zeros(1))  # 初始为[0]，learnable
 
    def forward(self, image, noise=None):
        if noise is None:
            batch, _, Z, X, Y = image.shape
            noise = image.new_empty(batch, 1, Z, X, Y).normal_()  
            # 返回一个新size的张量，填充未初始化的数据，默认返回的张量与image同dtype和device
            # 向特征图加噪
        return image + self.weight * noise

class ModulatedConv3d(nn.Module):
    def __init__(
        self,
        in_channels:    int,           # input channels (B, Cin, Z, X, Y)
        out_channels:   int,           # output channels (B, Cout Z, X, Y)
        kernel_size:    int = 3,       # convolution kernal
        stride :        int = 1,
        dilation:       int = 1,
        style_dim:      int = 512,     # the input style tensor channels (B, 1, S)
        demodulate:    bool = True,
    ):
        super().__init__()
 
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation

        # Define the conv parameters (cout, cin, P, Q, R)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))

        # Just a ignoreable constant
        self.eps = 1e-8
        
        # modulate method ( B x S ) -> (B x cin)
        # self.mod = EqualLinear(style_dim, in_channels)

        # need demodulate : True
        self.demod = demodulate
 
    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size[0] - 1) * (stride - 1) + dilation * (kernel - 1)) // 2, ((size[1] - 1) * (stride - 1) + dilation * (kernel - 1)) // 2, ((size[2] - 1) * (stride - 1) + dilation * (kernel - 1)) // 2
    
 
    def forward(self, x, S):  # style:(batch, S)
        B, cin, Z, X, Y = x.shape
        # 获取前级feature map的维度信息
        # S = self.mod(S)
        S = S[:, None, :, None, None, None]
        W = self.weight[None, :, :, :, :, :]
        W = W * (S + 1) # (B cout cin Z X Y)
        
        if self.demod: # A kind of Spectral norm
            d = torch.rsqrt((W ** 2).sum(dim = (2,3,4,5), keepdim = True) + self.eps)
            W = W * d

        x = x.reshape(1, -1, Z, X, Y)
        _, _, *ws = W.shape
        W = W.reshape(B * self.out_channels, *ws)
        pad = self._get_same_padding((Z, X, Y), self.kernel_size, self.dilation, self.stride)
        x = F.conv3d(x, W, padding=pad, groups=B)
        x = x.reshape(-1, self.out_channels, Z, X, Y)
        return x

def channel_shuffle(x, groups):
    '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
    batchsize, num_channels, depth, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, depth, height, width)
    #permute
    x = x.permute(0,2,1,3,4,5).contiguous()
    # flatten
    x = x.view(batchsize, num_channels, depth, height, width)
    return x

class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.groups = groups
        mid_planes = out_planes//4
        g = 1 if out_planes==120 else groups
        if self.stride == 2:
            out_planes = out_planes - in_planes
        self.conv1    = nn.Conv3d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1      = nn.BatchNorm3d(mid_planes)
        self.conv2    = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2      = nn.BatchNorm3d(mid_planes)
        self.conv3    = nn.Conv3d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3      = nn.BatchNorm3d(out_planes)
        self.relu     = nn.ReLU(inplace=True)

        if stride == 2:
            self.shortcut = nn.AvgPool3d(kernel_size=(2,3,3), stride=2, padding=(0,1,1))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = channel_shuffle(out, self.groups)
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))

        if self.stride == 2:
            out = self.relu(torch.cat([out, self.shortcut(x)], 1))
        else:
            out = self.relu(out + x)

        return out

class RBGBlock(nn.Module):
    def __init__(self, 
                 input_channels,                        # The input channels
                 out_channels: int = 4 ,                # Output channels, each element is 4
                 latent_channels = 512,                 # The latent spaces dimension
                 upsample_out: tuple | bool = (2, 2, 2) # To upsample it or not
                 ):
        super().__init__()
        self.cin = input_channels
        self.cout = out_channels
        self.clat = latent_channels
        
        # self.toStyle = nn.Linear(latent_channels, input_channels)
        self.mconv = ModulatedConv3d(input_channels, out_channels, 1, demodulate = False)
        
        self.upSample = nn.Sequential(
            nn.Upsample(scale_factor= upsample_out, mode = 'trilinear', align_corners = False),
            Blur3d()
        ) if upsample_out else nn.Identity()
        
    def forward(self, 
                x: torch.Tensor,                    # The input tensor ( B C Z X Y ) 
                S: torch.Tensor,                    # The latent code ( B S ), S = 512
                prev: torch.Tensor | None = None,   # The previous pred tensor ( B 4 Z X Y )
                ):                                  # output is pred tensor (B 4 (F * Z) (F * X) (F *Y) ), F is scale factor  
        # S = self.toStyle(S)
        x = self.mconv(x, S)
        if prev is not None:
            x = x + prev
        x = self.upSample(x)
        
        return x

class styleBlock3d(nn.Module):
    def __init__(self,
                 input_channels: int,                       # The input channels
                 style_channels: int,                       # the channels for style
                 out_channels: int = 4 ,                    # Output channels, the channel is for latent code
                 latent_channels = 512,                     # The latent spaces dimension
                 upsample_lat: tuple | bool = (2, 2, 2),    # To upsample it or not
                 upsample_out: tuple | bool = (2, 2, 2)     # To upsample it or not
                 ):
        super().__init__()
        self.cin = input_channels
        self.cout = out_channels
        self.clat = latent_channels
        
        self.upSample = nn.Upsample(scale_factor= upsample_lat, mode = 'trilinear', align_corners = False) if upsample_lat else nn.Identity()
        
        self.toStyle1 = nn.Linear(latent_channels, input_channels)
        self.toNoise1 = nn.Linear(1, style_channels)
        self.mconv1 = ModulatedConv3d(input_channels, style_channels, kernel_size = 3)
        
        self.toStyle2 = nn.Linear(latent_channels, style_channels)
        self.toNoise2 = nn.Linear(1, style_channels)
        self.mconv2 = ModulatedConv3d(style_channels, style_channels, kernel_size = 3)
        
        self.act = nn.LeakyReLU()
        self.out = RBGBlock(style_channels, out_channels, latent_channels, upsample_out) # the RGB one is bigger than the latent picture
    
    def forward(self, 
                x: torch.Tensor,                    # The input tensor ( B C Z X Y ) 
                S: torch.Tensor,                    # The latent code ( B S ), S = 512
                N: torch.Tensor,                    # The input Noise ( B OUT[Z] OUT[X] OUT[Y] 1)
                prev: torch.Tensor | None = None,   # The previous pred tensor ( B 4 Z X Y )
                ):                                  # output is latent tensor (B C (F * Z) (F * X) (F * Y)) ; pred tensor (B 4 (F * Z) (F * X) (F * Y) )
        x = self.upSample(x)
        B, C, Z, X, Y = x.shape
        N = N[:,:Z,:X,:Y,:]
        
        
        S1 = self.toStyle1(S)
        N1 = self.toNoise1(N).permute((0,4,1,2,3))
        x = self.mconv1(x, S1)
        x = self.act(x + N1)
        
        S2 = self.toStyle2(S)
        N2 = self.toNoise2(N).permute((0,4,1,2,3))
        x = self.mconv2(x, S2)
        x = self.act(x + N2)
        
        prev = self.out(x, S2, prev)
        return x, prev

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class DiscriminatorBlock3d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 downsample: tuple | bool = (1, 2, 2)):
        super().__init__()
        self.conv_res = nn.Conv3d(in_channels, out_channels, 1, stride = (downsample if downsample else 1))

        self.net = nn.Sequential(
            SingleConv(in_channels=in_channels, out_channels= out_channels, kernel_size = 3, order = "cl", padding = 1),
            SingleConv(in_channels=out_channels, out_channels= out_channels, kernel_size = 3, order = "cl", padding = 1)
        )

        self.downSample = nn.Sequential(
            Blur3d(),
            nn.Conv3d(out_channels, out_channels, 1, stride = downsample) if downsample else nn.Identity())

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = self.downSample(x)
        x = (x + res) * (1 / sqrt(2))
        return x

class MapStyle3d(nn.Module):
    def __init__(self,                              # input shape : (B cin Z X Y)
                 out_channels: int = 512,           # output style channels number
                 ):                                 # output is a (B, 1, S) tensor
        super().__init__()
        self.cout = out_channels
        out_planes = [120, 240, 480]
        num_blocks = [4, 8, 4]
        
        #self.conv1 = SingleConv(in_channels= 16, out_channels= 32, order="cbr", stride = (1, 2, 2), padding= 1)
        self.conv1 = SingleConv(in_channels= 32, out_channels= 32, order="cbr", stride = (1, 2, 2), padding= 1)
        self.conv2 = SingleConv(in_channels= 32, out_channels= 64, order="cbr", stride = (1, 2, 2), padding= 1)
        
        self.conv3 = nn.Sequential(
            SingleConv(in_channels= 64, out_channels= 60, order="cbr", kernel_size= 1),
            nn.MaxPool3d(kernel_size=3, stride= (1, 2, 2), padding = 1))
        self.conv4 = nn.Sequential(
            SingleConv(in_channels= 128, out_channels= 120, order="cbr", kernel_size= 1),
            nn.MaxPool3d(kernel_size=3, stride= (1, 2, 2), padding = 1))
        self.conv5 = nn.Sequential(
            SingleConv(in_channels= 256, out_channels= 240, order="cbr", kernel_size= 1),
            nn.MaxPool3d(kernel_size=3, stride= (1, 2, 2), padding = 1))

        self.cin = 60
        
        self.layer1 = nn.AvgPool3d(kernel_size=3, stride=2, padding=1) # (32, 32, 32)
        self.layer2 = self._make_layer(out_planes[0], num_blocks[0], 3) # (16, 16, 16)
        self.layer3 = self._make_layer(out_planes[1], num_blocks[1], 3) # (8, 8, 8)
        self.layer4 = self._make_layer(out_planes[2], num_blocks[2], 3) # (4, 4, 4)
        
        self.pool = nn.AvgPool3d((2, 2, 2))
        self.map = EqualLinear(out_planes[2],out_channels)
        
    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(Bottleneck(self.cin, out_planes, stride=stride, groups=groups))
            self.cin = out_planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if x.shape[2:] == (16, 128, 128):
            x = self.conv1(x)
        if x.shape[2:] == (16, 64, 64):
            x = self.conv2(x)
        if x.shape[2:] == (16, 32, 32):
            x = self.conv3(x)
        if x.shape[2:] == (16, 16, 16):
            x = self.layer2(x)
        if x.shape[2:] == (8, 16, 16):
            x = self.conv4(x)
        if x.shape[2:] == (8, 8, 8):
            x = self.layer3(x)
        if x.shape[2:] == (4, 8, 8):
            x = self.conv5(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x[:,:,0,0,0]
        x = self.map(x)
        return x[:, None, :]