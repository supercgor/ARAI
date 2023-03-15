from torch import nn
from .unet3d_model import TransUNet3D
from .squeezenet import SqueezeNet
from .basic import ReverseLayerF

class DANNUNet(nn.Module):
    def __init__(self,
                 img_channels = 1, 
                 hidden_channels = 32, 
                 inp_size = (16,128,128), 
                 out_size = (4, 32, 32)):
        super(DANNUNet, self).__init__()
        self.unet = TransUNet3D(img_channels, hidden_channels, inp_size, out_size, out_feature = True)
        # torch.Size([batch_size, 64, Z, 32, 32])
        self.class_net = SqueezeNet(64, inp_size[1] // 4, inp_size[0], num_classes = 2)
    
    def forward(self, x, alpha):
        out, fea = self.unet(x)
        reverse_feature = ReverseLayerF.apply(fea, alpha)
        domain_output = self.class_net(reverse_feature)

        return out, domain_output