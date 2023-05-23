import numpy as np
import torch
import os
from collections import OrderedDict
from typing import Dict, List
from einops import rearrange

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
    def load(self, 
             name,      # The name of the file, should end with .npy or .poscar
             *args, 
             **kwargs
             ):
        
        if ".npy" in name:
            return self._load_npy(name, *args, **kwargs)
        
        elif ".poscar" in name:
            return self._load_poscar(name, *args, **kwargs)
        
        else:
            raise TypeError(f"There is no way to load {name}!")
    
    @classmethod
    def _load_npy(self, name):
        f = torch.load(name) # ( B C Z X Y)
        return f

    @classmethod
    def _load_poscar(self, name):
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
    
    #TODO
    @classmethod
    def pos2apprbox(self, points_dict, real_size = (3, 25, 25), out_size = (32, 128, 128), order = ("O", "H", None)):
        scale = torch.tensor([i/j for i,j in zip(out_size, real_size)])
        OUT = torch.Sparse.FloatTensor(*out_size, len(order)).zero_()
        
    @classmethod
    def box2pos(cls, box, real_size = (3, 25, 25), order = ("O", "H"), threshold = 0.7, nms = True,sort = True) -> Dict[str, torch.Tensor]:
        D, H, W, E, C = box.shape
        scale = torch.tensor([i/j for i,j in zip(real_size, (D, H, W))], device= box.device)
        box = rearrange(box, 'd h w e c -> e d h w c')
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
    def pos2poscar(self, 
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