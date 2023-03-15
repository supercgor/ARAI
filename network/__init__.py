from .unet3d_model import UNet3D, TransUNet3D
from .squeezenet import SqueezeNet

model = {
    "UNet3D" : UNet3D,
    "TransUNet3D" : TransUNet3D,
    "SqueezeNet": SqueezeNet,
}