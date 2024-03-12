import torch
from torch import nn, Tensor

from ..op import *


class unet_water(nn.Module):
    def __init__(self, 
                 in_size: tuple[int] = (10, 100, 100), 
                 in_channels: int = 1,
                 out_size: tuple[int] = (3, 25, 25),
                 out_channels: tuple[int] | int = 10, 
                 out_conv_blocks = 3,
                 model_channels: int = 32, 
                 embedding_input: int = 1, # 0 for bulk water # 1 for cluster water
                 embedding_channels: int = 128,
                 num_res_blocks: int = (2, 2), 
                 attention_resolutions: tuple[int] = (4, 8, 16), 
                 dropout: int = 0.1, 
                 channel_mult: tuple[int] = (1, 2, 3, 4, 8), 
                 out_mult: int = 4,
                 z_down: tuple[int] = (1, 2, 4),
                 conv_resample: bool = True, 
                 num_heads: int = 8, 
                 ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dims = len(in_size)
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.embedding_input = embedding_input
        self.embedding_channels = embedding_channels
        self.up_blocks = num_res_blocks[0]
        self.down_blocks = num_res_blocks[1]
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.out_mult = out_mult
        self.conv_resample = conv_resample
        self.dtype = torch.float
        self._input_transform = None
        self._output_transform = None
        
        if embedding_input > 0:
            self.embedding = nn.Sequential(
                nn.Linear(embedding_input, embedding_channels),
                nn.ReLU(True),
                nn.Linear(embedding_channels, embedding_channels),
                nn.ReLU(True),
                nn.Linear(embedding_channels, embedding_channels),
            )
        else:
            self.embedding = None
            self.embedding_channels = None
            self.embedding_input = None
            
        
        in_size = np.array(in_size)
        channel_mult = np.array(channel_mult)
        down_size = []
        up_size = []
        for i in range(len(channel_mult)): # 0 -> 1 -> 2 -> 3 -> 4 -> mid -> 3 -> 2 -> out
            down_size.append(in_size)
            ds = 2 ** i
            if ds >= out_mult and i < len(channel_mult) - 1:
                up_size.insert(0, in_size)
            if ds in z_down:
                in_size = np.ceil(in_size / 2).astype(int)
            else:
                in_size = np.ceil(in_size / [1, 2, 2]).astype(int)
            
        #print(down_size, up_size)
        
        down_ch = model_channels * channel_mult
        up_ch = np.flip(down_ch, axis = 0)[:len(up_size)+1]
        
        #print(down_ch, up_ch)
        
        self.inp = conv_nd(self.dims, in_channels, model_channels, 3, padding=1) # 1 -> 32
        
        ds = 1
        skip_chs = []
        self.enc = nn.Sequential()
        for level, (in_ch, out_ch) in enumerate(zip(down_ch[:-1], down_ch[1:])): # [(32,64), (64,128), (128,256), ...]
            for n in range(self.down_blocks):
                layer = TimestepEmbedSequential()
                layer.add_module(f"res", ResBlock(in_ch, self.embedding_channels, dropout = dropout, dims = self.dims))
                if ds in attention_resolutions and n < self.up_blocks - 1:
                    layer.add_module(f"attn", AttentionBlock(in_ch, num_heads = num_heads, use_new_attention_order = True))
                skip_chs.append(in_ch)
                self.enc.add_module(f"layer{level}-{n}", layer)
            self.enc.add_module(f"down{level}", Downsample(in_ch, conv_resample, dims = self.dims, out_channels = out_ch, z_down = ds in z_down))
            ds *= 2
            
        in_ch = out_ch
        
        self.mid = TimestepEmbedSequential()
        self.mid.add_module("res0", ResBlock(in_ch, self.embedding_channels, dropout = dropout, dims = self.dims))
        if ds in attention_resolutions:
            self.mid.add_module("attn", AttentionBlock(in_ch, num_heads = num_heads, use_new_attention_order = True))
            self.mid.add_module("res1", ResBlock(in_ch, self.embedding_channels, dropout = dropout, dims = self.dims))

        
        self.dec = nn.Sequential()
        for level, (in_ch, out_ch, ups) in enumerate(zip(up_ch[:-1], up_ch[1:], up_size)):
            self.dec.add_module(f"up{level}",Upsample(in_ch, True, self.dims, out_ch, out_size = ups.tolist()))
            ds //= 2
            for n in range(self.up_blocks):
                layer = TimestepEmbedSequential()
                layer.add_module(f"res", ResBlock(skip_chs.pop() + out_ch, self.embedding_channels, dropout = dropout, out_channels = out_ch, dims = self.dims))
                if ds in attention_resolutions and n < self.up_blocks - 1:
                    layer.add_module(f"attn", AttentionBlock(out_ch, num_heads = num_heads, use_new_attention_order = True))
                self.dec.add_module(f"layer{level}-{n}", layer)
            if out_mult == ds:
                break
        
        if (out_size == up_size[-1]).all():
            self.resample = nn.Identity()
        else:
            self.resample = TimestepEmbedSequential(
                avg_adt_pool_nd(self.dims, list(out_size)),
                ResBlock(out_ch, None, dropout = dropout, dims = self.dims)
            )
        
        self.out = TimestepEmbedSequential()
        if isinstance(out_channels, int):
            out_channels = [out_channels]
        for i, ch in enumerate(out_channels):
            layer = TimestepEmbedSequential()
            for j in range(out_conv_blocks):
                layer.add_module(f"conv{j}", conv_nd(self.dims, out_ch, out_ch, 1))
                layer.add_module(f"act{j}", nn.LeakyReLU())
            layer.add_module(f"conv{j+1}", conv_nd(self.dims, out_ch, ch, 1, bias = False))
            self.out.add_module(f"out{i}", layer)        

    def forward(self, x: Tensor, emb: Tensor = None) -> Tensor:
        if self._input_transform is not None:
            x = self._input_transform(x)
            
        xs = []
        x = self.inp(x)
        if self.embedding is not None:
            emb = self.embedding(emb)
        else:
            emb = None
        ds = 1
        for i, module in enumerate(self.enc):
            if isinstance(module, Downsample):
                x = module(x)
                ds *= 2
            else:
                x = module(x,emb)
                if ds >= self.out_mult:
                    xs.append(x)
            
        x = self.mid(x, emb)
        
        for i, module in enumerate(self.dec):
            if isinstance(module, Upsample):
                x = module(x)
            else:
                y = xs.pop()
                x = torch.cat([x, y], dim=1)
                x = module(x, emb)
        
        x = self.resample(x)
        
        xs = []
        
        for i, module in enumerate(self.out):
            xs.append(module(x))
        x = torch.cat(xs, dim=1)
        
        if self._output_transform is not None:
            x = self._output_transform(x)
        
        return x
    
    def apply_transform(self, inp = None, out = None):
        if inp is not None:
            self._input_transform = inp
        if out is not None:
            self._output_transform = out