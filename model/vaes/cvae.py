import torch
from torch import nn
from torch.nn import functional as F

from ..op import GatedResBlock, Downsample, conv_nd, AttentionBlock, avg_adt_pool_nd, Upsample

class ConditioalVAE3(nn.Module):
    def __init__(self, in_channel, model_channel, out_channel = None, latent_channel = None, dropout = 0.0):
        super().__init__()
        self.in_ch   = in_channel
        self.out_ch  = out_channel or in_channel
        self.ch      = model_channel
        self.z_ch    = latent_channel or model_channel
        self.dp      = dropout
                
        self.condition_encoder = nn.Sequential(
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
        )
        
        self.condition_mid = conv_nd(3,      4 * self.ch, self.z_ch, 1)        
        self.condition_bot = nn.Sequential(
            avg_adt_pool_nd(3, (1, 1, 1)),
            nn.Flatten(1),
            nn.Linear(4 * self.ch, self.z_ch),
        )
        
        self.vae_encoder_mid = nn.Sequential(
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
            
            conv_nd(3,      4 * self.ch, 2 * self.z_ch, 1),
        )
        
        self.vae_encoder_bot = nn.Sequential(
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
            nn.Flatten(1),
            nn.Linear(4 * self.ch, 2 * self.z_ch),
        )
        
        # 1 * 4 * 4
        self.vae_decoder_mid = nn.Sequential(
            conv_nd(3,      self.z_ch, 4 * self.ch, 1),
            GatedResBlock(  4 * self.ch, dropout=self.dp, out_channels=4 * self.ch, dims=3),
            AttentionBlock( 4 * self.ch, num_heads = 8, use_new_attention_order = True),
            
            Upsample(       4 * self.ch, True, 3, 2 * self.ch, out_size = (1, 7, 7)),
            GatedResBlock(  2 * self.ch, dropout=self.dp, out_channels=2 * self.ch, dims=3),
            AttentionBlock( 2 * self.ch, num_heads = 8, use_new_attention_order = True),
            
            Upsample(       2 * self.ch, True, 3, 1 * self.ch, out_size = (2, 13, 13)),
            GatedResBlock(  1 * self.ch, dropout=self.dp, out_channels=1 * self.ch, dims=3),
            
            Upsample(       1 * self.ch, True, 3, 1 * self.ch, out_size = (2, 25, 25)),
            GatedResBlock(  1 * self.ch, dropout=self.dp, out_channels=1 * self.ch, dims=3),
            
            nn.SiLU(),
            conv_nd(3,      1 * self.ch, 1 * self.out_ch, 1),
            nn.Sigmoid()
        )
        
        # 1 -> 4 * 25 * 25
        self.vae_decoder_bot = nn.Sequential(
            nn.Linear(self.z_ch, 4 * self.ch * 2 * 4 * 4),
            nn.Unflatten(1, (4 * self.ch, 2, 4, 4)),
            
            GatedResBlock(  4 * self.ch, dropout=self.dp, out_channels=4 * self.ch, dims=3),
            AttentionBlock( 4 * self.ch, num_heads = 8, use_new_attention_order = True),
            
            Upsample(       4 * self.ch, True, 3, 2 * self.ch, out_size = (2, 7, 7)),
            GatedResBlock(  2 * self.ch, dropout=self.dp, out_channels=2 * self.ch, dims=3),
            AttentionBlock( 2 * self.ch, num_heads = 8, use_new_attention_order = True),
            
            Upsample(       2 * self.ch, True, 3, 1 * self.ch, out_size = (4, 13, 13)),
            GatedResBlock(  1 * self.ch, dropout=self.dp, out_channels=1 * self.ch, dims=3),
            
            Upsample(       1 * self.ch, True, 3, 1 * self.ch, out_size = (4, 25, 25)),
            GatedResBlock(  1 * self.ch, dropout=self.dp, out_channels=1 * self.ch, dims=3),
            
            nn.SiLU(),
            conv_nd(3,      1 * self.ch, 1 * self.out_ch, 1),
            nn.Sigmoid()
        )
        
    def compile_loss(self, conf_weight = None, offset_weight = None, vae_weight = None, pos_weight = None):
        self.conf_weight = conf_weight or self.conf_weight
        self.offset_weight = offset_weight or self.offset_weight
        self.vae_weight = vae_weight or self.vae_weight
        pos_weight = pos_weight or self.pos_weight
        self.register_buffer('pos_weight', torch.tensor([pos_weight], dtype = torch.float))
        
    def forward(self, input):
        input = self.inp_transform(input)
        
        Z = input.shape[2]
        if Z % 4 != 0:
            raise ValueError(f"Z dimension must be divisible by 4, got {Z}")
        cond, mid, bot = torch.split(input, [Z // 4, Z // 4, Z // 2], dim=2)
        
        cond_enc = self.condition_encoder(cond)
        
        cond_mid = self.condition_mid(cond_enc)
        cond_bot = self.condition_bot(cond_enc)
        
        z_mid = self.vae_encoder_mid(mid)
        z_bot = self.vae_encoder_bot(bot)
        
        inp_mid = self.reparameterize(z_mid)
        inp_bot = self.reparameterize(z_bot)
        
        inp_mid = self.vae_decoder_mid(inp_mid)
        inp_bot = self.vae_decoder_bot(inp_bot)
        
        input = torch.cat([cond, inp_mid, inp_bot], dim=2)
        input = self.out_transform(input)
        
        return input, (z_mid, z_bot), (cond_mid, cond_bot)
        
    def compute_loss(self, input, target, latent, condition):
        Z = input.shape[3]
        input = input[..., Z//4:, :]
        target = target[..., Z//4:, :]
        
        nelem = target[0,...,0].numel()
        
        # no reweighting
        mask = target[...,(0,)] > 0.5
        
        # reweighting, optional
        mask = input[..., (0,)].clamp(min=0.5).detach() * mask
        
        valid_num = mask.flatten(1).sum(1)
        input_fea, target_fea = input[...,1:4], target[...,1:4]
        
        confidence_loss = F.binary_cross_entropy_with_logits(input[...,0].logit(eps = 1E-6), target[...,0], pos_weight=self.pos_weight, reduction='mean')
        offset_loss = (F.mse_loss(input_fea, target_fea, reduction='none') * mask).flatten(1).sum(1) / valid_num
        offset_loss = offset_loss.mean()
        
        kls = []
        for lat, con in zip(latent, condition):
            mu, logvar = torch.chunk(lat, 2, dim=1)
            kl = -0.5 * (1 + logvar - (mu - con).pow(2) - logvar.exp())
            kl = kl.flatten(1).sum(1) / nelem
            kls.append(kl.mean())
        

        total_loss = confidence_loss * self.conf_weight + offset_loss * self.offset_weight + sum(kls) * self.vae_weight
               
        return total_loss, {'vae': sum(kls), 'conf': confidence_loss, 'offset': offset_loss}
    
    def inp_transform(self, input):
        # B X Y Z C -> B C Z X Y
        input = input.permute(0, 4, 3, 1, 2)
        return input
    
    def out_transform(self, input):
        # B C Z X Y -> B X Y Z C
        input = input.permute(0, 3, 4, 2, 1)
        return input
    
    def reparameterize(self, input):
        mu, logvar = torch.chunk(input, 2, dim=1)
        std = torch.exp(0.5 * logvar)   
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def conditional_sample(self, input, condition_only = False):
        cond = self.inp_transform(input)

        cond_enc = self.condition_encoder(cond)
        
        cond_mid = self.condition_mid(cond_enc)
        cond_bot = self.condition_bot(cond_enc)
        
        if not condition_only:
            cond_mid += torch.randn_like(cond_mid)
            cond_bot += torch.randn_like(cond_bot)
        
        inp_mid = self.vae_decoder_mid(cond_mid)
        inp_bot = self.vae_decoder_bot(cond_bot)
        
        input = torch.cat([cond, inp_mid, inp_bot], dim=2)
        
        input = self.out_transform(input)
            
        return input
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    