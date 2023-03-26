import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .basic import model_structure, SingleConv
from .parts import Fire

class SqueezeNet(nn.Module):

    def __init__(self, in_channels, sample_size, sample_duration, version=1.1, num_classes=600):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        if version == 1.0:
            ini = SingleConv(in_channels= in_channels, out_channels= 96, kernel_size= 7, order = "cbr",stride = (1,2,2), padding=(3,3,3))
            self.features = nn.Sequential(
                ini,
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(in_channels = 96, squeeze_channels = 16, expand1x1_channels = 64, expand3x3_channels = 64),
                Fire(in_channels = 128, squeeze_channels = 16, expand1x1_channels = 64, expand3x3_channels = 64, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(in_channels = 128, squeeze_channels = 32, expand1x1_channels = 128, expand3x3_channels = 128),
                Fire(in_channels = 256, squeeze_channels = 32, expand1x1_channels = 128, expand3x3_channels = 128, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(in_channels = 256, squeeze_channels = 48, expand1x1_channels = 192, expand3x3_channels = 192),
                Fire(in_channels = 384, squeeze_channels = 48, expand1x1_channels = 192, expand3x3_channels = 192, use_bypass=True),
                Fire(in_channels = 384, squeeze_channels = 64, expand1x1_channels = 256, expand3x3_channels = 256),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(in_channels = 512, squeeze_channels = 64, expand1x1_channels = 256, expand3x3_channels = 256, use_bypass=True),
            )
        if version == 1.1:
            ini = SingleConv(in_channels= in_channels, out_channels= 64, kernel_size= 3, order = "cbr",stride = (1,2,2), padding=(1,1,1))
            self.features = nn.Sequential(
                ini,
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(in_channels = 64, squeeze_channels = 16, expand1x1_channels = 64, expand3x3_channels = 64),
                Fire(in_channels = 128, squeeze_channels = 16, expand1x1_channels = 64, expand3x3_channels = 64, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(in_channels = 128, squeeze_channels = 32, expand1x1_channels = 128, expand3x3_channels = 128),
                Fire(in_channels = 256, squeeze_channels = 32, expand1x1_channels = 128, expand3x3_channels = 128, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(in_channels = 256, squeeze_channels = 48, expand1x1_channels = 192, expand3x3_channels = 192),
                Fire(in_channels = 384, squeeze_channels = 48, expand1x1_channels = 192, expand3x3_channels = 192, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(in_channels = 384, squeeze_channels = 64, expand1x1_channels = 256, expand3x3_channels = 256),
                Fire(in_channels = 512, squeeze_channels = 64, expand1x1_channels = 256, expand3x3_channels = 256, use_bypass=True),
            )
        # Final convolution is initialized differently form the rest
        
        final_conv = SingleConv(in_channels= 512, out_channels= self.num_classes, kernel_size= 1, order = "cr")
        softmax = nn.Softmax(dim=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.AvgPool3d((last_duration, last_size, last_size), stride=1),
            softmax
        )

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)

if __name__ == "__main__":
    # sample_size: img_size, sample_duration: Z
    model = SqueezeNet(version=1.1, in_channels = 64, sample_size = 16, sample_duration = 16, num_classes=600)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    model_structure(model)
    # (B, channels, Z, X, Y)
    input_var = torch.randn(8, 64, 16, 16, 16)
    output = model(input_var)
    print(output.shape)
