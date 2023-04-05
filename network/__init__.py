from .unet3d_model import UNet3D, TransUNet3D
from .squeezenet import SqueezeNet3d
from .NLayerNN import NLayerDiscriminator

model = {
    "UNet3D" : UNet3D,
    "TransUNet3D" : TransUNet3D,
    "SqueezeNet3d": SqueezeNet3d,
    "NLNN": NLayerDiscriminator
}