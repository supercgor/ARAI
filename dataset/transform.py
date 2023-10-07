import torch
from torch import nn, Tensor
import torchvision
from typing import Literal

def resize(x: Tensor, size: tuple[int] = (100, 100)) -> Tensor:
    return nn.functional.interpolate(x, size, mode='bilinear', align_corners=False)

class Resize(nn.Module):
    def __init__(self, size: tuple[int] = (100, 100)):
        super().__init__()
        self.size = size
        
    def forward(self, x: Tensor) -> Tensor:
        return resize(x, self.size)

@torch.jit.script
def cutout(x: Tensor, N: int = 4, scale: tuple[float, float] = (0.02, 0.98), size: float = 0.02, layerwise: bool = True) -> Tensor:
    """
    Cutout augmentation. Randomly cut out a rectangle from the image. The operation is in-place.

    Args:
        x (Tensor): _description_
        N (int, optional): _description_. Defaults to 4.
        scale (tuple[float, float], optional): _description_. Defaults to (0.02, 0.98).
        size (float, optional): _description_. Defaults to 0.02.
        layerwise (bool, optional): _description_. Defaults to True.

    Returns:
        Tensor: _description_
    """
    _, D, H, W = x.shape
    HW = torch.as_tensor([H, W], device=x.device, dtype=x.dtype)
    if not layerwise:
        D = 1
    x = x.permute(1, 2, 3, 0) # D, H, W, C
    usearea = torch.randn(D, N, device = x.device, dtype= x.dtype).abs() * size
    pos = torch.rand(D, N, 2, device = x.device, dtype= x.dtype)
    sizex = torch.randn(D, N, device = x.device, dtype= x.dtype).abs() * usearea.sqrt()
    sizey = usearea / sizex
    size_hw = torch.stack([sizex, sizey], dim=-1).clip(*scale)
    start_points = ((pos - size_hw / 2).clamp(0, 1) * HW).long()
    end_points = ((pos + size_hw / 2).clamp(0, 1) * HW).long()
    
    if layerwise:
        for i in range(D):
            xi, stp, enp = x[i], start_points[i], end_points[i]
            xmean = xi.mean()
            for j in range(N):
                xi[stp[j,0]:enp[j,0], stp[j,1]:enp[j,1]] = xmean
    else:
        xi, stp, enp = x, start_points[0], end_points[0]
        xmean = xi.mean()
        for j in range(N):
            xi[:, stp[j,0]:enp[j,0], stp[j,1]:enp[j,1]] = xmean
    x = x.permute(3, 0, 1, 2)
    return x

@torch.jit.script
def noisy(x: Tensor, intensity: float = 0.1, noisy_mode: list[int] = (0, 1, 2), noisy_type: list[int] = (0, 1), layerwise: bool = True) -> Tensor:
    """
    Add noise to the image. The operation is in-place.
    
    Args:
        x (Tensor): input tensor, shape (C, D, H, W)
        intensity (float): noise intensity. Defaults to 0.1.
        noisy_mode (tuple[int]): noise mode, 0: None, 1: add, 2: times. Defaults to (0, 1, 2).
        noisy_type (tuple[int]): noise type, 0: normal, 1: uniform. Defaults to (0, 1).
        layerwise (bool): whether to apply the noise according to the layers. Defaults to True.
    Returns:
        Tensor: output tensor, shape (C, D, H, W)
    """
    C, D, H, W = x.shape
    if not layerwise:
        D = 1
    x = x.permute(1, 2, 3, 0) # D, H, W, C
    n = torch.empty(H, W, C, device=x.device, dtype=x.dtype)
    noisy_modes = torch.randint(0, len(noisy_mode), (D,), device=x.device)
    noisy_types = torch.randint(0, len(noisy_type), (D,), device=x.device)
    if layerwise:
        for i in range(D):
            noisy, mode = noisy_type[noisy_types[i]], noisy_mode[noisy_modes[i]]
            if mode == 0:
                continue
            
            if noisy == 0:
                n.normal_(0.0, intensity)
            elif noisy == 1:
                n.uniform_(-intensity, intensity)
            
            if mode == 1:
                x[i] += n
            elif mode == 2:
                x[i] *= 1 + n
    else:
        noisy, mode = noisy_type[noisy_types[0]], noisy_mode[noisy_modes[0]]
        if mode != 0:
            if noisy == 0:
                n.normal_(0.0, intensity)
            elif noisy == 1:
                n.uniform_(-intensity, intensity)
                
            if mode == 1:
                x += n
            elif mode == 2:
                x *= 1 + n
                
    x.clamp_(0, 1)
    x = x.permute(3, 0, 1, 2)
    return x

class Cutout(nn.Module):
    def __init__(self, N: int = 4, scale: tuple[float, float] = (0.02, 0.98), size: float = 0.02, layerwise: bool = True):
        super().__init__()
        self.N = N
        self.scale = scale
        self.size = size
        self.layerwise = layerwise
        
    def forward(self, x: Tensor) -> Tensor:
        return cutout(x, self.N, self.scale, self.size, self.layerwise)

class Noisy(nn.Module):
    def __init__(self, intensity: float = 0.1, noisy_mode: list[int] = (0, 1, 2), noisy_type: list[int] = (0, 1), layerwise: bool = True):
        super().__init__()
        self.intensity = intensity
        self.noisy_mode = noisy_mode
        self.noisy_type = noisy_type
        self.layerwise = layerwise
        
    def forward(self, x: Tensor) -> Tensor:
        return noisy(x, self.intensity, self.noisy_mode, self.noisy_type, self.layerwise)

class ColorJitter(nn.Module):
    def __init__(self, brightness: float | tuple[float, float] = 0.1, 
                contrast: float | tuple[float, float] = 0.1, 
                saturation: float | tuple[float, float] = 0.1,
                hue: float | tuple[float, float] = 0.1):
        super().__init__()
        self.jt = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(1, 0, 2, 3)
        x = self.jt(x)
        x = x.permute(1, 0, 2, 3)
        return x.clamp_(0, 1)

class Blur(nn.Module):
    def __init__(self, ksize=5, sigma: float = 2.0):
        super().__init__()
        self.bur = torchvision.transforms.GaussianBlur(ksize, (sigma * 0.5, sigma * 1.5))
        
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(1, 0, 2, 3)
        x = self.bur(x)
        x = x.permute(1, 0, 2, 3)
        return x

def pixelshift(x: Tensor, max_shift=(0.1, 0.1), fill: Literal['none', 'mean'] | float = "none", ref: int = 3, layerwise: bool = False) -> Tensor:
    C, D, H, W = x.shape
    x = x.permute(1, 0, 2, 3)
    max_shift = torch.as_tensor(max_shift, device=x.device, dtype=x.dtype)

    if layerwise:
        maxshift = torch.rand(D, 2, device=x.device, dtype=x.dtype) * max_shift * 2 - max_shift
        shift = (maxshift * torch.as_tensor([H, W], device=x.device, dtype=x.dtype)).long()
    else:
        maxshift = torch.rand(2, device=x.device, dtype=x.dtype) * max_shift * 2 - max_shift
        maxshift = maxshift / (D - 1)
        shift = torch.arange(D, device=x.device, dtype=x.dtype) - ref
        shift = shift[:, None] * maxshift[None, :] # D, 2
        shift = (shift * torch.as_tensor([H, W], device=x.device, dtype=x.dtype)).long()
    for i in range(D):
        if i == ref:
            continue
        x[i] = torch.roll(x[i], tuple(shift[i]), (-2, -1))
        if fill != "none":
            if fill == "mean":
                fill = x[i].mean()
            if shift[i, 0] > 0:
                x[i, :, :shift[i, 0], :] = fill
            elif shift[i, 0] < 0:
                x[i, :, shift[i, 0]:, :] = fill
            if shift[i, 1] > 0:
                x[i, :, :, :shift[i, 1]] = fill
            elif shift[i, 1] < 0:
                x[i, :, :, shift[i, 1]:] = fill
    x = x.permute(1, 0, 2, 3)
    return x
    
class PixelShift(nn.Module):
    def __init__(self, max_shift=(0.1, 0.1), fill: Literal['none', 'mean'] | float = "none", ref: int = 3, layerwise: bool = False):
        super().__init__()
        self.max_shift = max_shift
        self.fill = fill
        self.ref = ref
        self.layerwise = layerwise

    def forward(self, x: Tensor) -> Tensor:  # shape (B ,C, X, Y)
        return pixelshift(x, self.max_shift, self.fill, self.ref, self.layerwise)
    
# class domainTransfer(nn.Module):
#     def __init__(self, module, offset = 0.0):
#         super().__init__()
#         self.module = module
#         self.offset = offset
    
#     def forward(self, x):
#         x = x.unsqueeze(0).transpose(1, 2)
#         alpha = (1 - torch.randn((1, 1, 1, 1, 1), device=x.device).abs()) + self.offset
#         alpha.clamp_(0, 1)
#         x = self.module(x).sigmoid() * alpha + x * (1 - alpha)
#         x = x.transpose(1, 2).squeeze(0)
#         return x