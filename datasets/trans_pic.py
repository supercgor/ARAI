import numpy as np
import cv2
import torch
from torch import nn
import torchvision.transforms as tf
import random
from torchvision.utils import make_grid, save_image

class PixelShift(nn.Module):
    def __init__(self, max_shift=(5, 5), fill=..., ref=5):
        super().__init__()
        self.max_shift = max_shift
        self.fill = fill
        self.ref = ref

    def forward(self, x):  # shape (B ,C, X, Y)
        for i in range(0, x.shape[0]):
            if i == self.ref:
                continue
            shift = (random.randint(-self.max_shift[0], self.max_shift[0]),
                     random.randint(-self.max_shift[1], self.max_shift[1]))
            x[i] = torch.roll(x[i], shift, (1, 2))
            if self.fill is not None:
                fill = self._fill(x[i])
                if shift[0] > 0:
                    x[i, :, :shift[0], :] = fill
                elif shift[0] < 0:
                    x[i, :, shift[0]:, :] = fill
                if shift[1] > 0:
                    x[i, :, :, :shift[1]] = fill
                elif shift[1] < 0:
                    x[i, :, :, shift[1]:] = fill
        return x

    def _fill(self, y):
        if self.fill is ...:
            return 0
        elif self.fill == "mean":
            return y.mean().item()


class CutOut(nn.Module):
    def __init__(self, max_n=4, scale=(0.1, 0.9), ratio=0.02):
        super().__init__()
        self.max_n = max_n
        self.ratio = ratio
        self.scale = scale

    def forward(self, x):
        B, C, X, Y = x.shape
        area = X * Y * 0.1
        for i in range(B):
            for _ in range(random.randint(0, self.max_n)):
                use_area = random.randint(int(area * 0.1), int(area))
                pos = (random.randint(0, X), random.randint(0, Y))
                if random.randbytes(1):
                    rd_x = random.randint(
                        int(X * self.scale[0]), int(X * self.scale[1]))
                    rd_y = use_area // rd_x
                else:
                    rd_y = random.randin(
                        int(Y * self.scale[0]), int(Y * self.scale[1]))
                    rd_x = use_area // rd_y
                value = x[i].mean() * random.uniform(0.5, 1.5)
                value.clip_(0, 1)
                x[i, :, pos[0]-rd_x//2: pos[0]+rd_x//2,
                    pos[1] - rd_y//2:pos[1]+rd_y//2] = value
        return x


class Noisy(nn.Module):
    def __init__(self, intensity=0.1, mode=["add", "times", None], noisy=["uniform", "normal"], add_factor=1):
        super().__init__()
        self.int = intensity
        self.noisy = noisy
        self.mode = mode
        self.add_factor = add_factor

    def forward(self, x):
        for i in range(x.shape[0]):
            noisy, mode = random.choice(self.noisy), random.choice(self.mode)
            if mode is None:
                continue
            else:
                noise = torch.FloatTensor(x.shape[1:])
                if noisy == "normal":
                    noise.normal_(0, self.int)
                elif noisy == "uniform":
                    noise.uniform_(-self.int, self.int)
                if mode == "add":
                    x[i].add_(noise * self.add_factor)
                elif mode == "times":
                    x[i].mul_(1 + noise)
        x.clip_(0, 1)
        return x


class ColorJitter(nn.Module):
    def __init__(self, B=0.3, C=0.3, S=0.0):
        super().__init__()
        self.J = tf.ColorJitter(brightness=B, contrast=C, saturation=S)

    def forward(self, x):
        for i in range(x.shape[0]):
            x[i] = self.J(x[i])
        return x

class Blur(nn.Module):
    def __init__(self, ksize=5, sigma=2.0):
        super().__init__()
        self.ksize = ksize
        self.sigma = sigma
        self.G = tf.GaussianBlur(ksize, sigma)

    def forward(self, x):
        for i in range(x.shape[0]):
            x[i] = self.G(x[i])
        return x

stan_T = tf.Compose([
        PixelShift(fill=None),
        CutOut(),
        Blur(),
        tf.RandomApply([ColorJitter()], p = 0.5),
        Noisy()
        ])

if __name__ == '__main__':
    a = torch.load("/home/supercgor/gitfile/ARAI/datasets/data/bulkice/datapack/T160_1.npz")
    r = [0,0,0,1,1,2,2,3,3,4,4,5,5,7,7,8]
    IMGS = a['imgs'][r]
    
    stan_T(IMGS)
    
    save_image(make_grid(IMGS), "test.png")
