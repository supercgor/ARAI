import torch

from torch.nn import Linear, Dropout, Softmax, LayerNorm
from torch import nn
from math import sqrt
from copy import deepcopy
from .basic import SingleConv, conv3d

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

class Mlp(nn.Module):
    def __init__(self, in_channels, mid_channels=1024, dropout=0.1):
        super(Mlp, self).__init__()
        self.fc1 = Linear(in_channels, mid_channels)
        self.fc2 = Linear(mid_channels, in_channels)
        self.act_fn = nn.functional.gelu
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

        attention_scores = (query_layer @ key_layer.transpose(-1, -2)) / sqrt(self.ahs)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = attention_probs @ value_layer
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.ahs * self.nah,)
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
    
class Encoder(nn.Module):
    def __init__(self, in_channels, MLP_channels, MLP_dropout, attn_dropout, trans_layer = 6):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(in_channels, eps=1e-6)
        for _ in range(trans_layer):
            layer = Block(in_channels, MLP_channels, MLP_dropout, attn_dropout)
            self.layer.append(deepcopy(layer))

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

class ViT(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, img_size=(5, 8, 8), hidden_channels=128, MLP_channels=1024, trans_dropout=0.1, attn_dropout=0, trans_layer = 6):
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