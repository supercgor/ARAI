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
