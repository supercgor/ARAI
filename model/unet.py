import torch
from torch import nn, Tensor

from .op import *

# ============================================================================
# UNet: Edited and adapted from github repo: https://github.com/AlexGraikos/diffusion_priors    
class unet_onehot(nn.Module):
    def __init__(self, 
                 image_size: tuple[int] = (10, 100, 100), 
                 in_channels: int = 1, 
                 model_channels: int = 32, 
                 latent_channels: int = 128,
                 out_channels: int = 6, 
                 num_res_blocks: int = 1, 
                 attention_resolutions: tuple[int] = (), 
                 dropout: int = 0, 
                 channel_mult: tuple[int] = (1, 2, 4, 8, 16), 
                 out_mult: int = 2,
                 z_down: tuple[int] = (1, 2, 4), 
                 conv_resample: bool = True, 
                 use_checkpoint: int = False, 
                 num_heads: int = 1, 
                 num_head_channels: int = -1, 
                 num_heads_upsample: int = -1, 
                 use_scale_shift_norm: bool = False, 
                 use_new_attention_order: bool = False,
                 use_vaeblock: bool = False,
                 ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_vaeblock = use_vaeblock
        self.dims = len(image_size)
        
        ds = 1
        ds_mult = [ds * (2**i) for i in range(len(channel_mult))]
        ds_size = [image_size] # [(10, 100, 100), (5, 50, 50), (3, 25, 25), (2, 13, 13), (2, 7, 7)]
        for i in range(len(channel_mult) - 1):
            size = ds_size[-1]
            if 2 ** i in z_down:
                size = tuple(math.ceil(j / 2) for j in size)
            else:
                size = (size[0], ) + tuple(math.ceil(j / 2) for j in size[1:])
            ds_size.append(size)
        ds_ch = [int(channel_mult[i] * model_channels) for i in range(len(channel_mult))]
        skip_chs = []
        
        self.inp = conv_nd(self.dims, in_channels, model_channels, 3, padding=1) # 1 -> 32
        
        self.down = nn.ModuleList([])
        for level, (in_ch, out_ch) in enumerate(zip(ds_ch[:-1], ds_ch[1:])): # [(32,64), (64,128), (128,256), ...]
            for n in range(num_res_blocks):
                layer = TimestepEmbedSequential()
                layer.add_module(f"res{n}", ResBlock(in_ch, None, dropout = dropout, dims = self.dims, use_checkpoint = use_checkpoint, use_scale_shift_norm = use_scale_shift_norm))
                if ds in attention_resolutions:
                    layer.add_module(f"attn{n}", AttentionBlock(in_ch, use_checkpoint = use_checkpoint, num_heads = num_heads, num_head_channels = num_head_channels, use_new_attention_order = use_new_attention_order))
                skip_chs.append(in_ch)
                self.down.append(layer)
            self.down.append(Downsample(in_ch, conv_resample, dims = self.dims, out_channels = out_ch, z_down = ds in z_down))
            ds *= 2
        
        in_ch = out_ch
        self.middle = TimestepEmbedSequential()
        self.middle.add_module("res0", ResBlock(in_ch, None, dropout, dims=self.dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm))
        if ds in attention_resolutions:
            self.middle.add_module("attn", AttentionBlock(in_ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
        if self.use_vaeblock:
            self.fc_mu = nn.Sequential(
                nn.Flatten(),
                nn.Linear(math.prod(ds_size[-1]) * in_ch, latent_channels),
            )
            self.fc_var = nn.Sequential(
                nn.Flatten(),
                nn.Linear(math.prod(ds_size[-1]) * in_ch, latent_channels),
            )

        self.up = nn.ModuleList([])
        for level, (in_ch, out_ch, size) in enumerate(zip(ds_ch[-1:0:-1], ds_ch[-2::-1], ds_size[-2::-1])):
            self.up.append(Upsample(in_ch, True, self.dims, out_ch, out_size = size))
            ds //= 2
            for n in range(num_res_blocks):
                layer = TimestepEmbedSequential()
                layer.add_module(f"res{n}", ResBlock(skip_chs.pop() + out_ch, None, dropout = dropout, out_channels = out_ch, dims = self.dims, use_checkpoint = use_checkpoint, use_scale_shift_norm = use_scale_shift_norm))
                if ds in attention_resolutions:
                    layer.add_module(f"attn{n}", AttentionBlock(out_ch, use_checkpoint = use_checkpoint, num_heads = num_heads_upsample, num_head_channels = num_head_channels, use_new_attention_order = use_new_attention_order))
                self.up.append(layer)
            if ds == out_mult:
                break
            
        self.out = TimestepEmbedSequential()
        self.out.add_module("res", ResBlock(out_ch, None, dropout, dims=self.dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm))
        self.out.add_module("conv1", conv_nd(self.dims, out_ch, out_ch, 1))
        self.out.add_module("act1", nn.ReLU(True))
        self.out.add_module("conv2", conv_nd(self.dims, out_ch, out_ch, 1))
        self.out.add_module("act2", nn.ReLU(True))
        self.out.add_module("conv3", conv_nd(self.dims, out_ch, out_channels, 1))
        

    def forward(self, x: Tensor, debug: bool = False) -> Tensor:
        """ Apply the model to an input.

        Args:
            x (Tensor): input tensor
            debug (bool, optional): return every layer's shape. Defaults to False.

        Shapes:
            x: :math:`(B, C, (D), H, W)`
            output: :math:`(B, C, (D), H, W)`
            
        Returns:
            Tensor: output tensor
        """
        xs = []
        shapes = []
        x = self.inp(x)
        for i, module in enumerate(self.down):
            x = module(x)
            if i % (self.num_res_blocks + 1) != self.num_res_blocks:
                shapes.append(x.shape)
                xs.append(x)
        if self.use_vaeblock:
            mu, logvar = self.fc_mu(x), self.fc_var(x)
        else:
            mu, logvar = None, None
        x = self.middle(x)
        shapes.append(x.shape)
        for i, module in enumerate(self.up):
            if i % (self.num_res_blocks + 1) != 0:
                x = torch.cat([x, xs.pop()], dim=1)
            x = module(x)
            shapes.append(x.shape)
        
        if debug:
            print("shapes", shapes)
        x = self.out(x)
        pos, typ = x[:, :3], x[:, 3:]
        return pos, typ, mu, logvar

class VAEunet(nn.Module):
    def __init__(self, 
                 image_size: tuple[int] = (4, 25, 25), 
                 in_channels: int = 10, 
                 out_channels: int = 10,
                 model_channels: int = 32,
                 embedding_input: int = 1,
                 embedding_channels: int = 128,
                 latent_channels: int = 128,
                 num_res_blocks: int = 2, 
                 attention_resolutions: tuple[int] = (), 
                 dropout: int = 0, 
                 channel_mult: tuple[int] = (1, 2, 3, 4), 
                 z_down: tuple[int] = (2, 8),
                 skip_mult: tuple[int] = (4, ),
                 conv_resample: bool = True, 
                 num_heads: int = 1, 
                 num_head_channels: int = -1, 
                 num_heads_upsample: int = -1, 
                 use_new_attention_order: bool = False,
                 ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.embedding_input = embedding_input
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.skip_mult = skip_mult
        self.dtype = torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.dims = len(image_size)
        self.embedding_channels = embedding_channels
        
        self.embedding = nn.Sequential(
            nn.Linear(embedding_input, embedding_channels),
            nn.SiLU(True),
            nn.Linear(embedding_channels, embedding_channels),
            nn.SiLU(True),
            nn.Linear(embedding_channels, embedding_channels),
        )
        
        ds = 1
        ds_mult = [ds * (2**i) for i in range(len(channel_mult))]
        ds_size = [tuple(image_size)] # [(4, 25, 25), (4, 13, 13), (2, 7, 7), (2, 4, 4), (1, 4, 4)]
        for i in range(len(channel_mult) - 1):
            size = ds_size[-1]
            if 2 ** i in z_down:
                size = tuple(math.ceil(j / 2) for j in size)
            else:
                size = (size[0], ) + tuple(math.ceil(j / 2) for j in size[1:])
            ds_size.append(size)
        ds_ch = [int(channel_mult[i] * model_channels) for i in range(len(channel_mult))]
        skip_chs = []
        
        self.inp = conv_nd(self.dims, in_channels, model_channels, 3, padding=1) # 1 -> 32
        
        self.down = nn.ModuleList([])
        for level, (in_ch, out_ch) in enumerate(zip(ds_ch[:-1], ds_ch[1:])): # [(32,64), (64,128), (128,256), ...]
            for n in range(num_res_blocks):
                layer = TimestepEmbedSequential()
                layer.add_module(f"res{n}", ResBlock(in_ch, None, dropout = dropout, dims = self.dims))
                if ds in attention_resolutions:
                    layer.add_module(f"attn{n}", AttentionBlock(in_ch, num_heads = num_heads, num_head_channels = num_head_channels, use_new_attention_order = use_new_attention_order))
                if ds in self.skip_mult:
                    skip_chs.append(in_ch)
                self.down.append(layer)
            self.down.append(Downsample(in_ch, conv_resample, dims = self.dims, out_channels = out_ch, z_down = ds in z_down))
            ds *= 2
        
        in_ch = out_ch
        
        self.enc_mu = nn.Sequential(
            nn.Flatten(),
            nn.Linear(math.prod(ds_size[-1]) * in_ch, latent_channels),
        )
        self.enc_var = nn.Sequential(
            nn.Flatten(),
            nn.Linear(math.prod(ds_size[-1]) * in_ch, latent_channels),
        )
        
        self.dec_var = nn.Sequential(
            nn.Linear(latent_channels, math.prod(ds_size[-1]) * in_ch),
            nn.Unflatten(1, (in_ch, *ds_size[-1])),
        )
        
        self.up = nn.ModuleList([])
        for level, (in_ch, out_ch, size) in enumerate(zip(ds_ch[-1:0:-1], ds_ch[-2::-1], ds_size[-2::-1])):
            self.up.append(Upsample(in_ch, True, self.dims, out_ch, out_size = size))
            ds //= 2
            for n in range(num_res_blocks):
                layer = TimestepEmbedSequential()
                if ds in self.skip_mult:
                    cat_ch = out_ch + skip_chs.pop()
                else:
                    cat_ch = out_ch
                layer.add_module(f"res{n}", ResBlock(cat_ch, None, dropout = dropout, out_channels = out_ch, dims = self.dims))
                if ds in attention_resolutions:
                    layer.add_module(f"attn{n}", AttentionBlock(out_ch, num_heads = num_heads_upsample, num_head_channels = num_head_channels, use_new_attention_order = use_new_attention_order))
                self.up.append(layer)
                
        self.out = TimestepEmbedSequential()
        self.out.add_module("res", ResBlock(out_ch, None, dropout, dims=self.dims, ))
        self.out.add_module("conv1", conv_nd(self.dims, out_ch, out_ch, 1))
        self.out.add_module("act1", nn.ReLU(True))
        self.out.add_module("conv2", conv_nd(self.dims, out_ch, out_ch, 1))
        self.out.add_module("act2", nn.ReLU(True))
        self.out.add_module("conv3", conv_nd(self.dims, out_ch, out_channels, 1))
        
    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        out = mu + std * eps
        return out
            
    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """ Apply the model to an input.

        Args:
            x (Tensor): input tensor
            debug (bool, optional): return every layer's shape. Defaults to False.

        Shapes:
            x: :math:`(B, C, (D), H, W)`
            output: :math:`(B, C, (D), H, W)`
            
        Returns:
            Tensor: output tensor
        """
        ds = 1
        xs = {}
        x = self.inp(x)
        emb = self.embedding(emb)
        
        for i, module in enumerate(self.down):
            if isinstance(module, Downsample):
                ds *= 2
                x = module(x)
            else:
                x = module(x, emb)
            if isinstance(module, TimestepEmbedSequential):
                if ds in self.skip_mult:
                    if ds not in xs:
                        xs[ds] = [x]
                    else:
                        xs[ds].append(x)
            
        mu, logvar = self.enc_mu(x), self.enc_var(x)
        
        x = self.sample(mu, logvar)
        x = self.dec_var(x)
        
        for i, module in enumerate(self.up):
            if isinstance(module, Upsample):
                ds //= 2
                x = module(x)
            else:
                if ds in self.skip_mult:
                    x = torch.cat([x, xs[ds].pop(0)], dim=1)
                x = module(x, emb)

        x = self.out(x)
        return x, mu, logvar

class VAELatEmb(nn.Module):
    def __init__(self, 
                 in_size: tuple[int] = (4, 25, 25), 
                 in_channels: int = 10, 
                 out_size: tuple[int] = (8, 25, 25),
                 out_channels: int = 10,
                 model_channels: int = 32,
                 embedding_input: int = 1,
                 latent_channels: int = 128,
                 num_res_blocks: int = (2, 2), 
                 dropout: int = 0, 
                 channel_mult: tuple[int] = (1, 2, 3, 4), 
                 z_down: tuple[int] = (2, 8),
                 z_up: tuple[int] = (2, 4, 8),
                 conv_resample: bool = True, 

                 ):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.dims = len(in_size)
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.embedding_input = embedding_input
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
            nn.Linear(embedding_input, latent_channels),
            nn.SiLU(True),
            nn.Linear(latent_channels, latent_channels),
            nn.SiLU(True),
            nn.Linear(latent_channels, latent_channels),
        )
        
        down_size = [tuple(in_size)] # [(4, 25, 25), (4, 13, 13), (2, 7, 7), (2, 4, 4), (1, 4, 4)]
        up_size = [tuple(out_size)] # [(8, 25, 25), (8, 13, 13), (4, 7, 7), (2, 4, 4), (1, 4, 4)]

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
                layer.add_module(f"res{n}", ResBlock(out_ch, None, dropout = dropout, out_channels = out_ch, dims = self.dims))
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
            
    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        if self.input_transform is not None:
            x = self.input_transform(x)
        
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
        emb = self.embedding(emb)[..., None, None, None]
        x = self.dec_var(x + emb)
        
        for i, module in enumerate(self.up):
            x = module(x)

        x = self.out(x)
        
        if self.output_transform is not None:
            x = self.output_transform(x)
        
        return x