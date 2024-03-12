import torch
from torch import nn
from torch.nn import functional as F

from ..op import GatedResBlock, Downsample, conv_nd, AttentionBlock, avg_adt_pool_nd, max_adt_pool_nd, Upsample, TimestepEmbedSequential
from .transform import inp_transform, out_transform

class ConditioalVAE2(nn.Module):
    def __init__(self, in_channel, model_channel, out_channel = None, latent_channel = None, conditonal_channel = None, dropout = 0.0):
        super().__init__()
        self.in_ch   = in_channel
        self.out_ch  = out_channel or in_channel
        self.ch      = model_channel
        self.z_ch    = latent_channel or model_channel
        self.c_ch    = conditonal_channel
        self.dp      = dropout
        
        self.conf_weight = 1.0
        self.offset_weight = 0.25
        self.rot_weight = 0.0
        self.vae_weight = 1.0
        self.pos_weight = 10.0
        
        self.inp_transform = inp_transform
        self.out_transform = out_transform
        
        self.condional_layer = nn.Sequential(
            conv_nd(3,      1 * self.in_ch, self.ch, 1),
            GatedResBlock(  1 * self.ch, dropout=self.dp, out_channels= self.ch, dims=3),
            Downsample(     1 * self.ch, True, 3, self.ch, z_down=True),
            
            GatedResBlock(  1 * self.ch, dropout=self.dp, out_channels= self.ch, dims=3),
            Downsample(     1 * self.ch, True, 3, 2 * self.ch, z_down=True),
            
            GatedResBlock(  2 * self.ch, dropout=self.dp, out_channels=2 * self.ch, dims=3),
            AttentionBlock( 2 * self.ch, num_heads = 8, use_new_attention_order = True),
            Downsample(     2 * self.ch, True, 3, 4 * self.ch, z_down=False),
            
            GatedResBlock(  4 * self.ch, dropout=self.dp, out_channels=4 * self.ch, dims=3),
            AttentionBlock( 4 * self.ch, num_heads = 8, use_new_attention_order = True),
            GatedResBlock(  4 * self.ch, dropout=self.dp, out_channels=4 * self.ch, dims=3),
            
            avg_adt_pool_nd(3, (1, 1, 1)),
            conv_nd(3,      4 * self.ch, self.z_ch, 1),
        )
                
        self.vae_enocder = TimestepEmbedSequential(
            conv_nd(3,      1 * self.in_ch, self.ch, 1),
            GatedResBlock(  1 * self.ch, dropout=self.dp, out_channels= self.ch, dims=3),
            Downsample(     1 * self.ch, True, 3, self.ch, z_down=True),
            
            GatedResBlock(  1 * self.ch, dropout=self.dp, out_channels= self.ch, dims=3),
            Downsample(     1 * self.ch, True, 3, 2 * self.ch, z_down=True),
            
            GatedResBlock(  2 * self.ch, dropout=self.dp, out_channels=2 * self.ch, dims=3),
            AttentionBlock( 2 * self.ch, num_heads = 8, use_new_attention_order = True),
            Downsample(     2 * self.ch, True, 3, 4 * self.ch, z_down=False),
            
            GatedResBlock(  4 * self.ch, dropout=self.dp, out_channels=4 * self.ch, dims=3),
            AttentionBlock( 4 * self.ch, num_heads = 8, use_new_attention_order = True),
            GatedResBlock(  4 * self.ch, dropout=self.dp, out_channels=4 * self.ch, dims=3),
            
            avg_adt_pool_nd(3, (1, 1, 1)),
            conv_nd(3,      4 * self.ch, 2 * self.z_ch, 1),
        )
        
        self.dec_mu = nn.Linear(self.z_ch, 4 * self.ch * 3 * 4 * 4)
        
        self.vae_decoder = TimestepEmbedSequential(
            
            
            Upsample(       4 * self.ch, False, 3, 4 * self.ch, out_size = (3, 4, 4)),
            GatedResBlock(  4 * self.ch, dropout=self.dp, out_channels=4 * self.ch, dims=3),
            AttentionBlock( 4 * self.ch, num_heads = 8, use_new_attention_order = True),
            
            Upsample(       4 * self.ch, True, 3, 2 * self.ch, out_size = (3, 7, 7)),
            GatedResBlock(  2 * self.ch, dropout=self.dp, out_channels=2 * self.ch, dims=3),
            AttentionBlock( 2 * self.ch, num_heads = 8, use_new_attention_order = True),
            
            Upsample(       2 * self.ch, True, 3, 1 * self.ch, out_size = (6, 13, 13)),
            GatedResBlock(  1 * self.ch, dropout=self.dp, out_channels=1 * self.ch, dims=3),
            
            Upsample(       1 * self.ch, True, 3, 1 * self.ch, out_size = (6, 25, 25)),
            GatedResBlock(  1 * self.ch, dropout=self.dp, out_channels=1 * self.ch, dims=3),
        )
        
        self.regressor = nn.Sequential(
            conv_nd(3,      1 * self.ch, 1 * self.ch, 1),
            nn.SiLU(),
            conv_nd(3,      1 * self.ch, 1 * self.out_ch, 1),
        )
        
    def compile_loss(self, conf_weight = None, offset_weight = None, rot_weight = None, vae_weight = None, pos_weight = None):
        self.conf_weight = conf_weight or self.conf_weight
        self.offset_weight = offset_weight or self.offset_weight
        self.rot_weight = rot_weight or self.rot_weight
        self.vae_weight = vae_weight or self.vae_weight
        self.pos_weight = pos_weight or self.pos_weight
        
    def apply_transform(self, inp = None, out = None):
        if inp is not None:
            self.inp_transform = inp
        if out is not None:
            self.out_transform = out
    
    def forward(self, x, c = None):
        if self.inp_transform is not None:
            x = self.inp_transform(x)
            if c is not None:
                c = self.inp_transform(c)
                
        if c is None:
            c = torch.zeros(x.shape[0], 2 * self.z_ch, dtype = x.dtype, device = x.device)
        else:
            c = self.condional_layer(c).flatten(1)
        
        x = self.vae_enocder(x).flatten(1)
        mu, sigma = torch.split(x, self.z_ch, dim=1)
        x = mu + torch.randn_like(sigma) * torch.exp(0.5 * sigma)
        x = self.dec_mu(x).reshape(-1, 4 * self.ch, 3, 4, 4)
        x = self.vae_decoder(x)
        x = self.regressor(x)
        
        if self.out_transform is not None:
            x = self.out_transform(x)
        
        return x, mu - c, sigma
        
    def compute_loss(self, input, target, mu, sigma):
        kl_loss    = torch.mean(-0.5 * (1 + sigma - mu.pow(2) - sigma.exp()).flatten(1).sum(1) / input[0,...,0].numel())
        
        pos_weight = torch.tensor([self.pos_weight], dtype=input.dtype, device=input.device)
        
        mask = target[...,0] > 0.5

        conditonal_loss = F.binary_cross_entropy_with_logits(input[...,0], target[...,0], pos_weight=pos_weight)
        offset_loss     = F.binary_cross_entropy(input[...,1:4], target[...,1:4], reduction='none')[mask].mean() - F.binary_cross_entropy(target[...,1:4], target[...,1:4], reduction='none')[mask].mean()
        rotational_loss = F.l1_loss(input[...,4:], target[...,4:], reduction='none')[mask].mean()
        
        loss = self.vae_weight    * kl_loss + self.conf_weight   * conditonal_loss + self.offset_weight * offset_loss + self.rot_weight * rotational_loss
               
        return loss, {"vae": kl_loss.item(), "conf": conditonal_loss.item(), "offset": offset_loss.item(), "rot": rotational_loss.item()}
    
    def complex_relative_square_error(self, input: torch.Tensor, target: torch.Tensor, batched: bool = True, channel_first: bool = True) -> torch.Tensor:
        if input.dtype == torch.complex64 or input.dtype == torch.complex32:
            input = torch.stack([input.real, input.imag], dim=-1)
            target = torch.stack([target.real, target.imag], dim=-1)
        elif channel_first:
            input = input.transpose(1, -1)
            target = target.transpose(1, -1)
            
        assert input.shape[-1] == 2, "The last dimension (real, complex) of input should be 2, but got {input.shape[-1]}"
        assert target.shape[-1] == 2, "The last dimension (real, complex) of target should be 2, but got {target.shape[-1]}"
        
        input = input.flatten(int(batched), -2)
        target = target.flatten(int(batched), -2)

        inp_norm = torch.norm(((input - target) ** 2).sum(-1) ** 0.5, p = 2, dim = 1)
        tgt_norm = torch.norm(((target - target.mean(-2, keepdim=True))**2).sum(-1) ** 0.5, p = 2, dim = 1)
        
        return 0.1 * (inp_norm / tgt_norm).mean()
    
    def conditional_sample(self, c, condition_only = False):
        if self.inp_transform is not None:
            c = self.inp_transform(c)

        c = self.condional_layer(c).flatten(1)
        
        if condition_only:
            x = torch.zeros(c.shape[0], self.z_ch, dtype = c.dtype, device = c.device)
        else:
            x = torch.randn(c.shape[0], self.z_ch, dtype = c.dtype, device = c.device)
            
        x = x + c
        x = self.dec_mu(x).reshape(-1, 4 * self.ch, 3, 4, 4)
        x = self.vae_decoder(x)
        x = self.regressor(x)
        
        if self.out_transform is not None:
            x = self.out_transform(x)
            
        return x
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    