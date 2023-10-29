
from typing import Literal
from torch import nn
from .op import conv_nd, avg_adt_pool_nd

class n_layer_disc(nn.Module):
    def __init__(self, in_channels, model_channels, channels_mult, z_down, dim = 3, norm: Literal["layer", "spectral", "group", "batch"] = "batch"):
        super().__init__()
        self.disc = nn.Sequential()
        self._input_transform = None
        self._output_transform = None
        model_channels = [model_channels * c for c in channels_mult]
        ch = [in_channels, *model_channels]
        
        for i, (in_ch, out_ch) in enumerate(zip(ch[:-1], ch[1:])):
            ds = 2 ** i
            layer = nn.Sequential()
            if norm == "spectral":
                if ds in z_down:
                    layer.add_module(f"sn_conv{i}", nn.utils.spectral_norm(conv_nd(dim, in_ch, out_ch, 4, 2, 1)))
                else:
                    layer.add_module(f"sn_conv{i}", nn.utils.spectral_norm(conv_nd(dim, in_ch, out_ch, 4, 1, 1)))
            elif norm == "batch":
                if ds in z_down:
                    layer.add_module(f"conv{i}", conv_nd(dim, in_ch, out_ch, 4, 2, 1))
                else:
                    layer.add_module(f"conv{i}", conv_nd(dim, in_ch, out_ch, 4, 1, 1))
                layer.add_module(f"bn{i}", nn.BatchNorm3d(out_ch))
                    
            layer.add_module(f"act{i}", nn.LeakyReLU(0.2, True))
            self.disc.add_module(f"layer{i}", layer)
        
        self.disc.add_module(f"conv", conv_nd(dim, ch[-1], 1, 3, 1, 1))
        
        self.out = nn.Identity()
    
    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)
            
        for m in self.disc:
            x = m(x)
        x = self.out(x).flatten(1)
        
        if self._output_transform is not None:
            x = self._output_transform(x)
        return x
    
    
    def apply_transform(self, inp = None, out = None):
        if inp is not None:
            self._input_transform = inp
        if out is not None:
            self._output_transform = out