import torch
from torch import nn
from .basic import basicModel

class NLayerDiscriminator(basicModel):
    """Defines a PatchGAN discriminator"""

    def __init__(self, in_channels = 128, hidden_channels=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        self.inp = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True))
        
        self.layer = nn.Sequential()
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.layer.extend([
                nn.Conv2d(hidden_channels * nf_mult_prev, hidden_channels * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_channels * nf_mult),
                nn.LeakyReLU(0.2, True)
            ])
            
        self.out = nn.Sequential()
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.out.extend([
            nn.Conv2d(hidden_channels * nf_mult_prev, hidden_channels * nf_mult, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels * nf_mult),
            nn.LeakyReLU(0.2, True)
        ])

        self.out.append(nn.Conv2d(hidden_channels * nf_mult, 1, kernel_size=4, stride=1, padding=1)) # output 1 channel prediction map

    def forward(self, x):
        """Standard forward."""
        x = self.inp(x)
        x = self.layer(x)
        x = self.out(x)
        return x.mean(dim = [1,2,3])
    
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def load_pretrained_layer(self, pre_dict_path):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())
        print(pre_dict_path)
        pretrained_state_dict = torch.load(pre_dict_path)
        pretrained_param_names = list(pretrained_state_dict.keys())
        match_list = []
        for i, param in enumerate(pretrained_param_names):
            if i == len(param_names):
                break
            if param == param_names[i]:
                match_list.append(param)
                state_dict[param] = pretrained_state_dict[param]
            else:
                break
        self.load_state_dict(state_dict)
        return match_list
    
if __name__ == "__main__":
    net = NLayerDiscriminator()
    inp = torch.rand((4,1,32,32))
    print(net(inp).shape)