import random
import math
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

import cv2
from typing import Tuple
from collections import OrderedDict
from typing import Dict, List
from einops import rearrange

def read_file(file_name):
    data = []
    with open(file_name) as fr:
        for line in fr:
            fn = line.strip().replace('\t', ' ')
            fn2 = fn.split(" ")
            if fn2[0] != '':
                data.append(fn2[0])
    return data


class indexGen():
    @classmethod
    def get(cls, use_len: int | None = 10,
            out_len:int = 16,
            max_index: int = 20,
            split_border: None | Tuple[int] = None,
            split_ratio: Tuple[int] = (0.4, 0.3, 0.3),
            rand: bool = False):
        split = split_border or []
        split = [0, *split, max_index]
        ratio = split_ratio[:len(split) - 1]
        use_len = use_len or out_len
        use_num = cls.weighted_split(use_len, ratio)
        out_num = cls.weighted_split(out_len, ratio)
        use_indices = []
        for lower, upper, use, out in zip(split[:-1], split[1:], use_num, out_num):
            try:
                indices = sorted(random.sample(range(lower, upper), k=min((upper - lower), use)))
                use_indices.extend(indices[i] for i in cls.gives_indices(min((upper - lower), use), out, rand=rand))
            except:
                print(lower, upper, use, out, split, ratio, out_num)
        return use_indices

    @staticmethod
    def weighted_split(nums, ratio, mode=math.ceil):
        if len(ratio) <= 1:
            return nums,
        else:
            out = mode(nums * ratio[0]/sum(ratio))
            return out, *indexGen.weighted_split(nums - out, ratio=ratio[1:], mode=mode)

    @staticmethod
    def gives_indices(upper, select_num, offset=0, rand=False):
        assert upper > 0, f"Input is not valid: select_num = {select_num}, upper = {upper}"
        if rand:
            
            sp = random.sample(range(upper), k=(select_num % upper))
            return [i + offset for i in range(upper) for _ in range(select_num // upper + (i in sp))]
        else:
            return [i + offset for i in range(upper) for _ in range((select_num // upper + ((select_num % upper) > i)))]


def read_pic(path: str,
             indices: list,
             img_size: tuple = (128, 128)
             ) -> torch.Tensor:

    IMG = []
    last = None

    for i in indices:
        filename = f'{i}.png'
        img_path = os.path.join(path, filename)
        if i == last:
            pass
        elif os.path.exists(img_path):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = np.flip(img.T, axis=1)
            if img.shape != img_size and img_size is not None:
                img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
            img = img[None, ...]
            last = i
        out = img.copy()
        IMG.append(out)
    IMG = np.stack(IMG) / 255
    IMG = torch.from_numpy(IMG).float()
    return IMG

class poscar():
    def __init__(self, 
                 path, 
                 lattice=(25, 25, 3), 
                 out_size=(32, 32, 4), 
                 elem=("O", "H"), 
                 cutoff=OrderedDict(O=2.2, H=0.8)
                 ):
        self._lattice = np.asarray(lattice)
        self._out_size = np.asarray(out_size)
        self._elem = elem
        self._cutoff = cutoff
        self.zoom = [i/j for i, j in zip(lattice, out_size)]

    @classmethod
    def load(cls, 
             name,      # The name of the file, should end with .npy or .poscar
             *args, 
             **kwargs
             ):
        
        if ".npy" in name:
            return cls._load_npy(name, *args, **kwargs)
        
        elif ".poscar" in name:
            return cls._load_poscar(name, *args, **kwargs)
        
        else:
            raise TypeError(f"There is no way to load {name}!")
    
    @classmethod
    def _load_npy(cls, name):
        f = torch.load(name) # ( B C Z X Y)
        return f

    @classmethod
    def _load_poscar(cls, name):
        with open(name, 'r') as f:
            f.readline()
            scale = float(f.readline().strip())
            x = float(f.readline().strip().split(" ")[0])
            y = float(f.readline().strip().split(" ")[1])
            z = float(f.readline().strip().split(" ")[2])
            real_size = (z, x, y)
            elem = OrderedDict((i,int(j)) for i,j in zip(f.readline()[1:-1].strip().split(" "),f.readline().strip().split(" ")))
            f.readline()
            f.readline()
            pos = OrderedDict((e, []) for e in elem.keys())
            for e in elem:
                for i in range(elem[e]):
                    X,Y,Z = map(float,f.readline().strip().split(" ")[0:3])
                    pos[e].append([Z * z, X * x, Y * y])
                pos[e] = torch.tensor(pos[e])
                pos[e][...,0].clip_(0,z - 1E-5)
                pos[e][...,1].clip_(0,x - 1E-5)
                pos[e][...,2].clip_(0,y - 1E-5)
        return {"scale": scale, "real_size": real_size, "elem": elem, "pos": pos}
    
    @classmethod
    def pos2box(cls, points_dict, real_size = (3, 25, 25), out_size = (8, 32, 32), order = ("O", "H")):
        scale = torch.tensor([i/j for i,j in zip(out_size, real_size)])
        OUT = torch.FloatTensor(*out_size, len(points_dict), 4).zero_()
        for i, e in enumerate(order):
            POS = points_dict[e] * scale
            IND = POS.floor().int()
            offset = (POS - IND)
            ONE = torch.ones((offset.shape[0], 1))
            offset = torch.cat((ONE, offset), dim = 1)
            OUT[IND[...,0],IND[...,1],IND[...,2], i] = offset
        return OUT
    
    @classmethod
    def pos2boxncls(cls, points_dict, real_size = (3, 25, 25), out_size = (8, 32, 32), order = ("O", "H")):
        scale = torch.tensor([i/j for i,j in zip(out_size, real_size)])
        OFFSET = torch.FloatTensor(*out_size, 3).zero_()
        CLS = torch.LongTensor(*out_size).fill_(len(order))
        for i, e in enumerate(order):
            POS = points_dict[e] * scale
            IND = POS.floor().int()
            OFFSET[IND[...,0],IND[...,1],IND[...,2]] = (POS- IND)
            CLS[IND[...,0],IND[...,1],IND[...,2]] = i
        return OFFSET.permute(3, 0, 1, 2), CLS
    
    @classmethod
    def boxncls2pos(cls, OFFSET: torch.Tensor, CLS: torch.Tensor, real_size = (3, 25, 25), order = ("O", "H"), nms = True, sort = True) -> Dict[str, torch.Tensor]:
        """
        OFFSET: (3, Z, X, Y)
        CLS: (Z, X, Y) or (n, Z, X, Y)
        """
        if CLS.dim() == 3: # n
            conf, CLS =  torch.ones_like(CLS), CLS# (Z, X, Y)
            sort = False
        else:
            conf, CLS = CLS.max(dim = 0).values, CLS.argmax(dim = 0) # (Z, X, Y)
        Z, X, Y = CLS.shape

        scale = torch.tensor([i/j for i,j in zip(real_size, (Z, X, Y))], device = CLS.device)
        pos = OrderedDict()
        for i, e in enumerate(order):
            IND = CLS == i
            POS = (OFFSET.permute(1,2,3,0)[IND] + IND.nonzero()) * scale
            CONF = conf[IND]
            
            if sort:
                sort_ind = torch.argsort(CONF, descending=True)
                POS = POS[sort_ind]
            
            pos[e] = POS
        if nms:
            pos = cls.nms(pos)
        return pos
    
    @classmethod
    def box2pos(cls, 
                box, 
                real_size = (3, 25, 25), 
                order = ("O", "H"), 
                threshold = 0.7, 
                nms = True,
                sort = True,
                format = "DHWEC") -> Dict[str, torch.Tensor]:
        
        threshold = 1 / (1 + math.exp(-threshold))
        
        newformat = "E D H W C"
        format, newformat = " ".join(format), " ".join(newformat)
        
        box = rearrange(box, f"{format} -> {newformat}", C = 4)
        E, D, H, W, C = box.shape
        scale = torch.tensor([i/j for i,j in zip(real_size, (D, H, W))], device= box.device)
        pos = OrderedDict()
        for i, e in enumerate(order):
            pd_cls = box[i,...,0]
            mask = pd_cls > threshold
            pd_cls = pd_cls[mask]
            pos[e] = torch.nonzero(mask) + box[i,...,1:][mask]
            pos[e] = pos[e] * scale
            if sort:
                sort_ind = torch.argsort(pd_cls, descending=True)
                pos[e] = pos[e][sort_ind]
        if nms:
            pos = cls.nms(pos)
        return pos
    
    def save(self, name, pos):
        output = ""
        output += f"{' '.join(self.elem)}\n"
        output += f"{1:3.1f}" + "\n"
        output += f"\t{self.lattice[0]:.8f} {0:.8f} {0:.8f}\n"
        output += f"\t{0:.8f} {self.lattice[1]:.8f} {0:.8f}\n"
        output += f"\t{0:.8f} {0:.8f} {self.lattice[2]:.8f}\n"
        output += f"\t{' '.join([str(ele) for ele in pos])}\n"
        output += f"\t{' '.join([str(pos[ele].shape[0]) for ele in pos])}\n"
        output += f"Selective dynamics\n"
        output += f"Direct\n"
        for ele in pos:
            p = pos[ele]
            for a in p:
                output += f" {a[0]/self.lattice[0]:.8f} {a[1]/self.lattice[1]:.8f} {a[2]/self.lattice[2]:.8f} T T T\n"

        path = f"{self.path}/result/{self.model_name}"
        if not os.path.exists(path):
            os.mkdir(path)

        with open(f"{path}/{name}", 'w') as f:
            f.write(output)
        return

    def save4npy(self, name, pred, NMS=True, conf=0.7):
        return self.save(name, self.npy2pos(pred, NMS=NMS, conf=conf))
    
    @classmethod
    def nms(cls, pd_pos: Dict[str, torch.Tensor], radius: Dict[str, float] = {"O":0.8, "H":0.6}):
        for e in pd_pos.keys():
            pos = pd_pos[e]
            if pos.nelement() == 0:
                continue
            if e == "O":
                cutoff = radius[e] * 1.8
            else:
                cutoff = radius[e] * 1.8
            DIS = torch.cdist(pos, pos)
            DIS = DIS < cutoff
            DIS = (torch.triu(DIS, diagonal= 1)).float()
            restrain_tensor = DIS.sum(0)
            restrain_tensor -= ((restrain_tensor != 0).float() @ DIS)
            SELECT = restrain_tensor == 0
            pd_pos[e] = pos[SELECT]
            
        return pd_pos
    
    @classmethod
    def pos2poscar(cls, 
                   path,                    # the path to save poscar
                   points_dict,             # the orderdict of the elems : {"O": N *　(Z,X,Y)}
                   real_size = (3, 25, 25)  # the real size of the box
                   ):
        output = ""
        output += f"{' '.join(points_dict.keys())}\n"
        output += f"{1:3.1f}" + "\n"
        output += f"\t{real_size[1]:.8f} {0:.8f} {0:.8f}\n"
        output += f"\t{0:.8f} {real_size[2]:.8f} {0:.8f}\n"
        output += f"\t{0:.8f} {0:.8f} {real_size[0]:.8f}\n"
        output += f"\t{' '.join(points_dict.keys())}\n"
        output += f"\t{' '.join(str(len(i)) for i in points_dict.values())}\n"
        output += f"Selective dynamics\n"
        output += f"Direct\n"
        for ele in points_dict:
            p = points_dict[ele]
            for a in p:
                output += f" {a[1]/real_size[1]:.8f} {a[2]/real_size[2]:.8f} {a[0]/real_size[0]:.8f} T T T\n"

        with open(path, 'w') as f:
            f.write(output)
            
        return
    
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
        self.J = transforms.ColorJitter(brightness=B, contrast=C, saturation=S)

    def forward(self, x):
        for i in range(x.shape[0]):
            x[i] = self.J(x[i])
        return x

class Blur(nn.Module):
    def __init__(self, ksize=5, sigma=2.0):
        super().__init__()
        self.ksize = ksize
        self.sigma = sigma
        self.G = transforms.GaussianBlur(ksize, sigma)

    def forward(self, x):
        for i in range(x.shape[0]):
            x[i] = self.G(x[i])
        return x