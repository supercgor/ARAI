import torch
from torch import nn
from .op import styleBlock, MapStyle3d
from functools import partial

class StyleGAN3D(nn.Module):
    def __init__(self, 
                 out_size: tuple = (4, 32, 32), 
                 latent_dim: int = 512, 
                 gan_elem: int = 1,
                 network_capacity: int = 16, 
                 attn_layers: list = [], 
                 ):
        super().__init__()
        self.out_size = out_size
        self.latent_dim = latent_dim
        self.gan_elem = gan_elem
        self.layer_up = [False, (1, 2, 2), (1, 2, 2), (1, 2, 2), False] # ((4,4,4) -> (4 4 4) -> (4 8 8) -> (4 16 16) -> (4 32 32) --attn-> (4 32 32) -> out
        self.layer_channels = [network_capacity, network_capacity * 2, network_capacity * 4, network_capacity * 8] # 16 16 32 64 128, don't be too large

        init_channels = self.layer_channels[0]
        self.layer_channels = [init_channels, *self.layer_channels]

        in_out_pairs = zip(self.layer_channels[:-1], self.layer_channels[1:])

        self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4, 4)))

        self.initial_conv = nn.Conv3d(init_channels, init_channels, 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            attn_fn = nn.Identity() # attn_and_ff(in_chan) if ind in attn_layers else 

            self.attns.append(attn_fn)

            self.blocks.append(styleBlock(in_chan, out_chan, 4, latent_dim, upsample_lat = self.layer_up[ind], upsample_out = self.layer_up[ind + 1]))

        self.map = MapStyle3d(latent_dim)

    def forward(self, 
                features: list,             # The feature given by UNet 
                input_noise: torch.Tensor   # The shape is (B out_Z out_X out_Y 1)
                ):
        features = torch.cat(list(map(self.map, features)), dim = 1).permute((1, 0, 2))
        B = input_noise.shape[0]
        
        x = self.initial_block.expand(B, -1, -1, -1, -1)
        x = self.initial_conv(x)

        out = None
        
        for S, block, attn in zip(features, self.blocks, self.attns):
            x = attn(x)
            x, out = block(x, S, input_noise, out)

        return out