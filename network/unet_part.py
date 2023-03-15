import math
import copy
from torch.nn import Dropout, LayerNorm, Softmax, Linear, LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .basic import SingleConv, conv3d

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


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, in_channels=256, out_channels=128, img_size = (2,8,8)):
        super(Embeddings, self).__init__()

        self.patch_embeddings = conv3d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=1,
                                       bias=False,
                                       padding=0)
        self.position_embeddings = nn.Parameter(torch.zeros(img_size[0] * img_size[1] * img_size[2], out_channels))

        self.dropout = Dropout(0.1)

    def forward(self, x):
        # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        # 20 img ver -> 320
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Encoder(nn.Module):
    def __init__(self, in_channels, MLP_channels, MLP_dropout, attn_dropout, trans_layer = 6):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(in_channels, eps=1e-6)
        for _ in range(trans_layer):
            layer = Block(in_channels, MLP_channels, MLP_dropout, attn_dropout)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        for layer_block in self.layer:
            x = layer_block(x)
        encoded = self.encoder_norm(x)
        return encoded


class Block(nn.Module):
    def __init__(self, in_channels, MLP_channels, MLP_dropout, attn_dropout):
        super(Block, self).__init__()
        self.hidden_size = in_channels
        self.attention_norm = LayerNorm(in_channels, eps=1e-6)
        self.ffn_norm = LayerNorm(in_channels, eps=1e-6)
        self.ffn = Mlp(in_channels, mid_channels=MLP_channels,
                       dropout=MLP_dropout)
        self.attn = Attention(in_channels, dropout=attn_dropout)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

class Mlp(nn.Module):
    def __init__(self, in_channels, mid_channels=1024, dropout=0.1):
        super(Mlp, self).__init__()
        self.fc1 = Linear(in_channels, mid_channels)
        self.fc2 = Linear(mid_channels, in_channels)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, in_channels, dropout=0):
        super(Attention, self).__init__()
        self.nah = 32  # num_attention_heads
        self.ahs = int(in_channels / self.nah)  # attention_head_size

        self.query = Linear(in_features=in_channels, out_features=in_channels)
        self.key = Linear(in_features=in_channels, out_features=in_channels)
        self.value = Linear(in_features=in_channels, out_features=in_channels)

        self.out = Linear(in_channels, in_channels)
        self.attn_dropout = Dropout(dropout)
        self.proj_dropout = Dropout(dropout)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        # ( batchsize * n_batch * head_num * head_size ) ( b * 128 * 32 * 4)
        new_x_shape = x.size()[:-1] + (self.nah, self.ahs)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        mixed_query_layer = self.query(x)
        mixed_key_layer = self.key(x)
        mixed_value_layer = self.value(x)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.ahs)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.ahs * self.nah,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Decoder(nn.Module):
    def __init__(self, in_channels=128, out_channels=256, img_size=(5, 8, 8)):
        super().__init__()
        self.conv_more = SingleConv(in_channels, out_channels, kernel_size = 3, padding= 1, num_groups = 8, order = "crb")
        self.img_size = img_size

    def forward(self, x):
        B, _, hidden = x.size()
        # reshape from (B, n_patch, hidden) to (B, hidden, h, w)
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, *self.img_size)
        x = self.conv_more(x)
        return x


class ViT(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, img_size=(5, 8, 8), hidden_channels=128, MLP_channels=1024, trans_dropout=0.1, attn_dropout=0, trans_layer = 6,vis=False):
        super().__init__()
        self.embeddings = Embeddings(
            in_channels=in_channels, out_channels=hidden_channels, img_size=img_size)
        self.encoder = Encoder(hidden_channels, MLP_channels=MLP_channels,
                               MLP_dropout=trans_dropout, attn_dropout=attn_dropout, trans_layer=trans_layer)
        self.decoder = Decoder(in_channels= hidden_channels, out_channels= out_channels, img_size=img_size)

    def forward(self, x):
        x = self.embeddings(x) # ( B, channels, 2, 8, 8)
        x = self.encoder(x)  # (B, 128, channels//2)
        x = self.decoder(x) # (B, channels, 2, 8, 8)
        return x

    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, strict)

if __name__ == "__main__":
    inp = torch.empty((2, 256, 5, 8, 8), dtype=torch.float32)
    model = ViT()
    out = model(inp)
    print(out.shape)
