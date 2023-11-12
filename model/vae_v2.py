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
        if embedding_input > 0:
            self.embedding_input = embedding_input
            self.embedding_channels = embedding_channels
            self.embedding = nn.Sequential(
                nn.Linear(embedding_input, embedding_channels),
                nn.ReLU(True),
                nn.Linear(embedding_channels, embedding_channels),
                nn.ReLU(True),
                nn.Linear(embedding_channels, embedding_channels),
            )
        else:
            self.embedding_input = None
            self.embedding_channels = None
            
        self.latent_channels = latent_channels
        self.up_blocks = num_res_blocks[0]
        self.down_blocks = num_res_blocks[1]
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = torch.float
        self.input_transform = None
        self.output_transform = None
        
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
                layer.add_module(f"res{n}", ResBlock(out_ch, self.embedding_channels, dropout = dropout, out_channels = out_ch, dims = self.dims))
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
            
    def forward(self, x: Tensor, emb: Tensor = None, encoder=False, decoder=False, resample = True) -> Tensor:
        if encoder:
            return self._forward_encoder(x, emb, True)
        if decoder:
            return self._forward_decoder(x, emb)
        if self.input_transform is not None:
            x = self.input_transform(x)
        
        if emb is not None:
            emb = self.embedding(emb)
        
        mu, logvar = self._forward_encoder(x, emb, False)
        
        if resample:
            x = self.sample(mu, logvar)
        else:
            x = mu
        
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
    
    def _forward_decoder(self, x, emb = None):
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
    
class vae_v3(nn.Module):
    def __init__(self, 
                 in_size: tuple[int] = (4, 25, 25), 
                 in_channels: int = 10,
                 out_size: tuple[int] = (16, 25, 25),
                 out_channels: tuple[int] | int = [1, 3, 6], 
                 out_conv_blocks = 2,
                 model_channels: int = 32, 
                 embedding_input: int = 0, # 0 for bulk water # 1 for cluster water
                 embedding_channels: int = 128,
                 num_res_blocks: int = (2, 2), 
                 attention_resolutions: tuple[int] = (4, 8, 16), 
                 latent_channels: int = 256,
                 dropout: int = 0.0, 
                 channel_mult: tuple[int] = (1, 2, 3, 4, 8), 
                 z_down: tuple[int] = (4, 8),
                 z_up: tuple[int] = (1, 2, 4, 8),
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
        self.latent_channels = latent_channels
        self.up_blocks = num_res_blocks[0]
        self.down_blocks = num_res_blocks[1]
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
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
        out_size = np.array(out_size)
        channel_mult = np.array(channel_mult)
        down_size = []
        up_size = []
        for i in range(len(channel_mult)): # 0 -> 1 -> 2 -> 3 -> 4 -> mid -> 3 -> 2 -> out
            down_size.append(in_size)
            up_size.append(out_size)
            ds = 2 ** i
            if ds in z_down:
                in_size = np.ceil(in_size / 2).astype(int)
            else:
                in_size = np.ceil(in_size / [1, 2, 2]).astype(int)
            if ds in z_up:
                out_size = np.ceil(out_size / 2).astype(int)
            else:
                out_size = np.ceil(out_size / [1, 2, 2]).astype(int)
        up_size = up_size[-1::-1]
        #print(down_size, up_size)
        
        down_ch = model_channels * channel_mult
        up_ch = np.flip(down_ch, axis = 0)[:len(up_size)+1]
        
        self._up_size = up_size
        self._down_size = down_size
        
        self.inp = conv_nd(self.dims, in_channels, model_channels, 3, padding=1) # 1 -> 32
        
        ds = 1
        self.enc = TimestepEmbedSequential()
        for level, (in_ch, out_ch) in enumerate(zip(down_ch[:-1], down_ch[1:])): # [(32,64), (64,128), (128,256), ...]
            for n in range(self.down_blocks):
                layer = TimestepEmbedSequential()
                layer.add_module(f"res", ResBlock(in_ch, self.embedding_channels, dropout = dropout, dims = self.dims))
                if ds in attention_resolutions and n < self.up_blocks - 1:
                    layer.add_module(f"attn", AttentionBlock(in_ch, num_heads = num_heads, use_new_attention_order = True))
                self.enc.add_module(f"layer{level}-{n}", layer)
            self.enc.add_module(f"down{level}", Downsample(in_ch, conv_resample, dims = self.dims, out_channels = out_ch, z_down = ds in z_down))
            ds *= 2
            
        in_ch = out_ch
        
        self.enc.add_module("noise_enc", conv_nd(self.dims, in_ch, 2 * latent_channels, 1))
        
        self.dec = TimestepEmbedSequential()
        
        self.dec.add_module("noise_dec", conv_nd(self.dims, latent_channels, in_ch, 1))
        
        for level, (in_ch, out_ch, ups) in enumerate(zip(up_ch[:-1], up_ch[1:], up_size[1:])):
            self.dec.add_module(f"up{level}",Upsample(in_ch, True, self.dims, out_ch, out_size = ups.tolist()))
            ds //= 2
            for n in range(self.up_blocks):
                layer = TimestepEmbedSequential()
                layer.add_module(f"res", ResBlock(out_ch, self.embedding_channels, dropout = dropout, out_channels = out_ch, dims = self.dims))
                if ds in attention_resolutions and n < self.up_blocks - 1:
                    layer.add_module(f"attn", AttentionBlock(out_ch, num_heads = num_heads, use_new_attention_order = True))
                self.dec.add_module(f"layer{level}-{n}", layer)
        
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

    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def sample(self, mu = None, logvar = None):
        if mu is None and logvar is None:
            shape = (1, self.latent_channels // 2, self._up_size[0])
            mu = torch.zeros(shape, dtype = self.dtype, device = self.device)
            logvar = torch.zeros(shape, dtype = self.dtype, device = self.device)
        elif mu is None:
            mu = torch.zeros_like(logvar)
        else:
            logvar = torch.zeros_like(mu)
            
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        out = mu + std * eps
        return out

    def encode(self, x, emb = None):
        if self._input_transform is not None:
            x = self._input_transform(x)
            
        x = self.inp(x)
        
        x = self.enc(x, emb)
        
        mu, logvar = torch.split(x, self.latent_channels, dim = 1)
        
        return mu, logvar
    
    def decode(self, x = None, emb = None):
        x = self.dec(x, emb)
                
        xs = []
        
        for i, module in enumerate(self.out):
            xs.append(module(x))
            
        x = torch.cat(xs, dim=1)
        
        if self._output_transform is not None:
            x = self._output_transform(x)
        
        return x
    
    def forward(self, x, emb = None) -> Tensor:
        mu, logvar = self.encode(x, emb)
        
        x = self.sample(mu, logvar)
        
        x = self.decode(x, emb)
        
        return x, mu, logvar
    
    def apply_transform(self, inp = None, out = None):
        if inp is not None:
            self._input_transform = inp
        if out is not None:
            self._output_transform = out