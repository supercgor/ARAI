import torch.autograd

from network.unet_part import *

class UNet3D(nn.Module):
    def __init__(self, n_channels, channels, output_z):
        super(UNet3D, self).__init__()
        self.output_z = output_z
        self.inc = DoubleConv(in_channels=n_channels, out_channels=channels, encoder=True)
        self.down1 = Down(channels, 2 * channels, all_dim=0)
        self.down2 = Down(2 * channels, 4 * channels, all_dim=0)
        self.down3 = Down(4 * channels, 8 * channels)
        self.down4 = Down(8 * channels, 8 * channels)
        self.up1 = Up(16 * channels, 4 * channels)
        self.up2 = Up(8 * channels, 2 * channels)
        self.up3 = Up(4 * channels, channels)
        self.up4 = Up(2 * channels, channels)
        self.out = Out(in_channels=channels)

    def forward(self, x):    # (batch_size, 1, 10, 128, 128)
        x1 = self.inc(x)     # (batch_size, 32, 10, 128, 128)
        x2 = self.down1(x1)  # (batch_size, 64, 10, 64, 64)
        x3 = self.down2(x2)  # (batch_size, 128, 10, 32, 32)
        x4 = self.down3(x3)  # (batch_size, 256, 5, 16, 16)
        x5 = self.down4(x4)  # (batch_size, 256, 2, 8, 8)
        x = self.up1(x4, x5) # (batch_size, 256+256 -> 128, 5, 16, 16)
        x = self.up2(x3, x)  # (batch_size, 64, 10, 32, 32)
        x = self.up3(x2, x)  # (batch_size, 32, 10, 64, 64)
        x = self.up4(x1, x)  # (batch_size, 128, 5, 16, 16)
        x = F.interpolate(x, (self.output_z, x.shape[-2], x.shape[-1]), mode='trilinear', align_corners=True)
        # (batch_size, 32, output_z, 128, 128)
        x = self.out(x)  # (batch_size, 8, 9, 32, 32)
        x = x.permute([0, 3, 4, 2, 1]).contiguous()
        # (batch_size, 32, 32, output_z, 8)
        shape = x.shape
        x = x.view(-1, 4)
        x[..., :3] = torch.sigmoid(x[..., :3])
        x = x.view(shape)
        return x

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


class TransUNet3D(nn.Module):
    def __init__(self, n_channels, channels, output_z):
        super(transUNet3D, self).__init__()
        self.output_z = output_z
        self.inc = DoubleConv(in_channels=n_channels, out_channels=channels, encoder=True)
        self.down1 = Down(channels, 2 * channels, all_dim=0)
        self.down2 = Down(2 * channels, 4 * channels, all_dim=0)
        self.down3 = Down(4 * channels, 8 * channels)
        self.down4 = Down(8 * channels, 8 * channels)
        self.vit = ViT(8 * channels, 8 * channels)
        self.up1 = Up(16 * channels, 4 * channels)
        self.up2 = Up(8 * channels, 2 * channels)
        self.up3 = Up(4 * channels, channels)
        self.up4 = Up(2 * channels, channels)
        self.out = Out(in_channels=channels)

    def forward(self, x):    # (batch_size, 1, 10, 128, 128)
        x1 = self.inc(x)     # (batch_size, 32, 10, 128, 128)
        x2 = self.down1(x1)  # (batch_size, 64, 10, 64, 64)
        x3 = self.down2(x2)  # (batch_size, 128, 10, 32, 32)
        x4 = self.down3(x3)  # (batch_size, 256, 5, 16, 16)
        x5 = self.down4(x4)  # (batch_size, 256, 2, 8, 8)
        x5 = self.vit(x5)
        x = self.up1(x4, x5)
        x = self.up2(x3, x)  # (batch_size, 64, 5, 32, 32)
        x = self.up3(x2, x)
        x = self.up4(x1, x)  # (batch_size, 32, 5, 128, 128)
        x = F.interpolate(x, (self.output_z, x.shape[-2], x.shape[-1]), mode='trilinear', align_corners=True)
        # (batch_size, 32, output_z, 128, 128)
        x = self.out(x)  # (batch_size, 8, 9, 32, 32)
        x = x.permute([0, 3, 4, 2, 1]).contiguous()
        # (batch_size, 32, 32, output_z, 8)
        shape = x.shape
        x = x.view(-1, 4)
        x[..., :3] = torch.sigmoid(x[..., :3])
        x = x.view(shape)
        return x

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

model = {
    "UNet3D": UNet3D,
    "TransUNet3D": TransUNet3D
}


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet3D(1, 32, 4)
    model.weight_init()
    model.to(device)
    inputs = torch.rand((1, 1, 10, 128, 128)).to(device)
    print(model(inputs).shape)
