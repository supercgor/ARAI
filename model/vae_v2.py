import torch
from torch import nn, Tensor

from .op import *

class VAELatEmb_v2(nn.Module):
    def __init__(self, 
                 in_size: tuple[int] = (4, 25, 25), 
                 in_channels: int = 10, 
                 out_size: tuple[int] = (4, 25, 25),
                 out_channels: int = 10,
                 model_channels: int = 32,
                 embedding_input: int = 2,
                 embedding_channels: int = 256,
                 latent_channels: int = 1024,
                 num_res_blocks: int = (2, 2), 
                 dropout: int = 0, 
                 channel_mult: tuple[int] = (1, 2, 3, 4, 8), # 1 2 4 8
                 z_down: tuple[int] = (2, 8), # 1 2 4 8
                 z_up: tuple[int] = (2, 8), # 1 2 4 8
                 conv_resample: bool = True, 
                 ):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.dims = len(in_size)
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.embedding_input = embedding_input
        self.embedding_channels = embedding_channels
        self.latent_channels = latent_channels
        self.up_blocks = num_res_blocks[0]
        self.down_blocks = num_res_blocks[1]
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = torch.float
        self.input_transform = None
        self.output_transform = None
        
        self.embedding = nn.Sequential(
            nn.Linear(embedding_input, embedding_channels),
            nn.ReLU(True),
            nn.Linear(embedding_channels, embedding_channels),
            nn.ReLU(True),
            nn.Linear(embedding_channels, embedding_channels),
        )
        
        down_size = [tuple(in_size)] # [(4, 25, 25), (4, 13, 13), (2, 7, 7), (2, 4, 4), (1, 4, 4)]
        up_size = [tuple(out_size)]

        for i in range(len(channel_mult) - 1):
            dos = down_size[-1]
            ups = up_size[-1]
            
            if 2 ** i in z_down:
                dos = tuple(math.ceil(j / 2) for j in dos)
            else:
                dos = (dos[0], ) + tuple(math.ceil(j / 2) for j in dos[1:])
                
            if 2 ** i in z_up:
                ups = tuple(math.ceil(j / 2) for j in ups)
            else:
                ups = (ups[0], ) + tuple(math.ceil(j / 2) for j in ups[1:])
                
            down_size.append(dos)
            up_size.append(ups)
            
        up_size = up_size[-2::-1]
        down_ch = [int(channel_mult[i] * model_channels) for i in range(len(channel_mult))]
        up_ch = down_ch[::-1]
        
        self.inp = conv_nd(self.dims, in_channels, model_channels, 3, padding=1) # 1 -> 32
        
        ds = 1
        
        self.down = nn.ModuleList([])
        for level, (in_ch, out_ch) in enumerate(zip(down_ch[:-1], down_ch[1:])): # [(32,64), (64,128), (128,256), ...]
            for n in range(self.down_blocks):
                layer = TimestepEmbedSequential()
                layer.add_module(f"res{n}", ResBlock(in_ch, None, dropout = dropout, dims = self.dims))
                layer.add_module(f"attn{n}", AttentionBlock(in_ch, num_heads = 8, use_new_attention_order = True))
                self.down.append(layer)
            self.down.append(Downsample(in_ch, conv_resample, dims = self.dims, out_channels = out_ch, z_down = ds in z_down))
            ds *= 2
        
        in_ch = out_ch
            
        self.enc = nn.Sequential(
            conv_nd(self.dims, in_ch, 2 * latent_channels, 1),
            )
        
        self.dec_var = nn.Sequential(
            conv_nd(self.dims, latent_channels, in_ch, 1),
        )
        
        self.up = nn.ModuleList([])
        for level, (in_ch, out_ch, ups) in enumerate(zip(up_ch[:-1], up_ch[1:], up_size)):
            self.up.append(Upsample(in_ch, True, self.dims, out_ch, out_size = ups))
            ds //= 2
            for n in range(self.up_blocks):
                layer = TimestepEmbedSequential()
                layer.add_module(f"res{n}", ResBlock(out_ch, embedding_channels, dropout = dropout, out_channels = out_ch, dims = self.dims))
                layer.add_module(f"attn{n}", AttentionBlock(out_ch, num_heads = 8, use_new_attention_order = True))
                self.up.append(layer)
                
        self.out = TimestepEmbedSequential(
            ResBlock(out_ch, None, dropout, dims=self.dims, ),
            conv_nd(self.dims, out_ch, out_ch, 1),
            nn.ReLU(True),
            conv_nd(self.dims, out_ch, out_ch, 1),
            nn.ReLU(True),
            conv_nd(self.dims, out_ch, out_channels, 1),
        )
        
    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        out = mu + std * eps
        return out
            
    def forward(self, x: Tensor, emb: Tensor, encoder=False, decoder=False) -> Tensor:
        if encoder:
            return self._forward_encoder(x, emb, True)
        if decoder:
            return self._forward_decoder(x, emb)
        if self.input_transform is not None:
            x = self.input_transform(x)
        
        emb = self.embedding(emb)
        
        mu, logvar = self._forward_encoder(x, emb, False)
        
        x = self.sample(mu, logvar)
        
        x = self._forward_decoder(x, emb)
        
        return x, mu, logvar
    
    def _forward_encoder(self, x, emb, transform = True):
        if self.input_transform is not None and transform:
            x = self.input_transform(x)
            
        x = self.inp(x)
        
        for i, module in enumerate(self.down):
            x = module(x)
            
        mu, logvar = torch.split(self.enc(x), self.latent_channels, dim = 1)
        
        return mu, logvar
    
    def _forward_decoder(self, x, emb):
        x = self.dec_var(x)
        
        for i, module in enumerate(self.up):
            if isinstance(module, Upsample):
                x = module(x)
            else:
                x = module(x, emb)

        x = self.out(x)
        
        if self.output_transform is not None:
            x = self.output_transform(x)
        
        return x