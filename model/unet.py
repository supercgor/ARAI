import os
import torch
from torch import nn
import torch.nn.functional as F
import einops
from torch import nn, Tensor
import numpy as np

from .utils import basicModel
from .op import *

# ============================================================================
# UNet: Edited and adapted from github repo: https://github.com/AlexGraikos/diffusion_priors


class UNetModel(basicModel):
    """
        The full UNet model with attention and timestep embedding.

        Could used for 1D, 2D, or 3D data.

        Args:
            image_size (tuple[int]): input of the image size. format as: tuple[(D), H, W]
            in_channels (int): channels in the input Tensor.
            model_channels (int): base channel count for the model.
            out_channels (int): channels in the output Tensor.
            num_res_blocks (int): number of residual blocks per downsample.
            attention_resolutions (tuple[int], optional): a collection of downsample rates at which attention will take place. May be a set, list, or tuple. For example, if this contains 4, then at 4x downsampling, attention will be used. Defaults to none.
            dropout (float, optional): the dropout probability. Defaults to 0.
            channel_mult (tuple[int], optional): channel multiplier for each level of the UNet. Defaults to (1, 2, 4, 8).
            z_down (tuple[int], optional): a collection of downsamples of z-stride equals to 2. Defaults to (1, 2, 4).
            conv_resample (bool, optional): if True, use learned convolutions for upsampling and downsampling.. Defaults to True.
            dims (int, optional): determines if the signal is 1D, 2D, or 3D. Defaults to 2.
            num_classes (_type_, optional): if specified (as an int), then this model will be class-conditional with `num_classes` classes.. Defaults to None.
            use_checkpoint (bool, optional): use gradient checkpointing to reduce memory usage, not very useful because it is extremely slow. Defaults to False.
            num_heads (int, optional): the number of attention heads in each attention layer. Defaults to 1.
            num_head_channels (int, optional): if specified, ignore num_heads and instead use a fixed channel width per attention head. Defaults to -1.
            num_heads_upsample (int, optional): _description_. Defaults to -1.
            use_scale_shift_norm (bool, optional): use a FiLM-like conditioning mechanism. Defaults to False.
            resblock_updown (bool, optional): use residual blocks for up/downsampling. Defaults to False.
            use_new_attention_order (bool, optional): use a different attention pattern for potentially increased efficiency. Defaults to False.
            time_embed (None | int, optional): use time embeding to condition output. Defaults to 4.
            reference_channels (None | int, optional): use reference tensor to condition output. Defaults to None.
            reference_mode (str, optional): use concat or add to condition output. Defaults to "concat".
            
        Returns:
            Tensor: the output Tensor.
        """

    def __init__(self, image_size: tuple[int], in_channels: int, model_channels: int, out_channels: int, num_res_blocks: int, attention_resolutions: tuple[int] = tuple(), dropout: int = 0, channel_mult: tuple[int] = (1, 2, 4, 8), z_down: tuple[int] = (1, 2, 4), conv_resample: bool = True, dims: int = 2, num_classes=None, use_checkpoint=False, num_heads=1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, time_embed: None | int = 4, reference_channels: None | int = None, reference_mode: str = "concat"):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.ref_channels = reference_channels
        self.ref_mode = reference_mode
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.time_embed = time_embed
        
        if self.time_embed is not None:
            time_embed_dim = model_channels * self.time_embed
            self.time_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        else:
            time_embed_dim = None

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)

        # Turn input channels to model channels, if reference channels are used, add a MixBlock.

        input_block = TimestepEmbedSequential()
        input_block.add_module("conv", conv_nd(
            dims, in_channels, ch, 3, padding=1))
        if reference_channels is not None:
            input_block.add_module("mix", MixBlock(
                ch, reference_channels, ch, mode=self.ref_mode, dims=dims))
        self.input_blocks = nn.ModuleList([input_block])

        # the start of downsampling

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for n in range(num_res_blocks):
                layers = TimestepEmbedSequential()
                layers.add_module(f"res{n}", ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm))

                if ds in attention_resolutions:
                    layers.add_module(f"attn{n}", AttentionBlock(
                        ch,
                        use_checkpoint=use_checkpoint,
                        num_heads=num_heads,
                        num_head_channels=num_head_channels,
                        use_new_attention_order=use_new_attention_order))

                self.input_blocks.append(layers)
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = int(model_channels * channel_mult[level + 1])
                layer = TimestepEmbedSequential()
                if resblock_updown:
                    layer.add_module("down_res", ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                        z_down=ds in z_down))
                else:
                    layer.add_module("down_conv",
                                     Downsample(ch, conv_resample, dims=dims, out_channels=out_ch, z_down=ds in z_down))

                self.input_blocks.append(layer)
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # bottom layer of the network ( R - A )

        self.middle_block = TimestepEmbedSequential()
        self.middle_block.add_module("res0", ResBlock(
            ch,
            time_embed_dim,
            dropout,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm))

        if ds in attention_resolutions:
            self.middle_block.add_module("attn", AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order))
        self._feature_size += ch

        # the start of upsampling
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = TimestepEmbedSequential()
                layers.add_module(f"res{i}", ResBlock(
                    ch + ich,
                    time_embed_dim,
                    dropout,
                    out_channels=ch,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm))
                if ds in attention_resolutions:
                    layers.add_module(f"attn{i}", AttentionBlock(
                        ch,
                        use_checkpoint=use_checkpoint,
                        num_heads=num_heads_upsample,
                        num_head_channels=num_head_channels,
                        use_new_attention_order=use_new_attention_order))

                if level and i == num_res_blocks:
                    out_ch = int(model_channels * channel_mult[level - 1])
                    if resblock_updown:
                        layers.add_module("up", ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            z_down=(ds//2) in z_down))
                    else:
                        layers.add_module("up", Upsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch, z_down=(ds//2) in z_down))
                    ds //= 2
                    ch = out_ch
                self.output_blocks.append(layers)
                self._feature_size += ch

        # output convolution

        self.out = nn.Sequential()
        self.out.add_module("norm", normalization(ch))
        self.out.add_module("act", nn.SiLU())
        self.out.add_module("conv", conv_nd(dims, ch, out_channels, 3, padding=1))

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor | None = None, y: torch.Tensor | None = None, ref: torch.Tensor | None = None, debug: bool = False) -> torch.Tensor:
        """ Apply the model to an input.

        Args:
            x (torch.Tensor): input tensor
            timesteps (torch.Tensor | None): time tensor. Defaults to None.
            y (torch.Tensor | None): label tensor. Defaults to None.
            ref (torch.Tensor | None): reference tensor, the shape shape as x, channel can be different if mode is "concat". Defaults to None.
            debug (bool, optional): return every layer's shape. Defaults to False.

        Shapes:
            x: :math:`(B, C, (D), H, W)`

            timestep: :math:`(B)`
            
            y: :math:`(B)`
            
            ref: :math:`(B, C', (D), H, W)`
            
            output: :math:`(B, C, (D), H, W)`
            
        Returns:
            torch.Tensor: output tensor
        """
        assert (y is not None) == (self.num_classes is not None), "must specify y if and only if the model is class-conditional"
        
        shapes, hs, emb = [x.shape], [], None
        if self.time_embed is not None:
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

            if self.num_classes is not None:
                assert y.shape == (x.shape[0],)
                emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, ref)
            shapes.append(h.shape)
            hs.append(h)

        h = self.middle_block(h, emb)
        shapes.append(h.shape)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
            shapes.append(h.shape)
        if debug:
            print("shapes", shapes)
        return self.out(h)


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(
            low_res, (new_height, new_width), mode="bilinear")
        x = torch.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)
    
class unet_onehot(nn.Module):
    def __init__(self, 
                 image_size: tuple[int] = (10, 100, 100), 
                 in_channels: int = 1, 
                 model_channels: int = 32, 
                 latent_channels: int = 128,
                 out_channels: int = 6, 
                 num_res_blocks: int = 2, 
                 attention_resolutions: tuple[int] = (), 
                 dropout: int = 0, 
                 channel_mult: tuple[int] = (1, 2, 4, 8, 16), 
                 out_mult: int = 2,
                 z_down: tuple[int] = (1, 2, 4), 
                 conv_resample: bool = True, 
                 num_classes: int = None, 
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
        self.num_classes = num_classes
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
            self.up.append(Upsample_conv(size, in_ch, out_ch))
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
            
        return self.out(x), mu, logvar
    
class Upsample_conv(nn.Module):
    def __init__(self, size, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(size)
        self.conv = conv_nd(len(size), in_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x