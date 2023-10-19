import h5py
import einops
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from typing import Callable

from utils import poscar, xyz, functional

class AFMDataset(Dataset):
    def __init__(self, path: str, useLabel: bool = True, useZ: int = 10, transform: Callable[[torch.Tensor], torch.Tensor] = None):
        self._path = path
        with h5py.File(path, 'r') as f:
            self._keys = list(f.keys())
            self._len = len(self._keys)
        self._useLabel = useLabel
        self._useZ = useZ
        self._transform = transform
        
    def __getitem__(self, index: int):
        file_name = self._keys[index]
        hfile = h5py.File(self._path, 'r')
        data = hfile[file_name]
        totalZ = data['afm'].shape[1]
        if self._useZ > totalZ:
            afm =data['afm'][()]
            ind = z_sampler(self._useZ, totalZ, is_rand=True)
            afm = afm[:, ind]
        else:
            ind = z_sampler(self._useZ, totalZ, is_rand=True)
            afm = data['afm'][:, ind]
            
        afm = torch.as_tensor(afm, dtype=torch.float)
        afm = einops.rearrange(afm, f"{' '.join(data['afm'].attrs['format'])}->C D H W")
        if self._transform is not None:
            afm = self._transform(afm)
        if self._useLabel:
            label = data['label'][()]
            
            labeltype, labelpos = poscar.pos2boxncls(label, data['label'].attrs['ion_num'])

            labeltype = torch.as_tensor(labeltype, dtype=torch.long)
            labelpos = torch.as_tensor(labelpos, dtype=torch.float)
            hfile.close()
            return file_name, afm, labeltype, labelpos
        else:
            hfile.close()
            return file_name, afm
        
    def __len__(self):
        return len(self._keys)

class AFMGenDataset(Dataset):
    def __init__(self, path: str, transform: Callable[[torch.Tensor], torch.Tensor] = None):
        self._path = path
        with h5py.File(path, 'r') as f:
            self._keys = list(f.keys())
            self._len = len(self._keys)
        self._transform = transform
        ang = 104.52 / 180 * np.pi
        initdir = np.array([
        [ 0.         , 0.         , 0.9572    ],
        [ 0.9266272  , 0.         ,-0.23998721],
        [ 0.         , 0.88696756 , 0.        ]
        ])
        self._invref = np.linalg.inv(initdir)
        self._ang = ang
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, index: int):
        # out: C Z X Y (cond, x, y, z, r1, r2, r3, r4, r5, r6)
        file_name = self._keys[index]
        hfile = h5py.File(self._path, 'r')
        data = hfile[file_name]
        atom4a = data['box_4a'][()]
        atom8a = data['box_8a'][()]
        temp = data.attrs['temp'] / 270
        box4a = np.zeros((25, 25, 4, 10))
        box8a = np.zeros((25, 25, 8, 10))
        for pos in atom4a:
            Oind = np.floor(pos[0])
            Ooff = pos[0] - Oind
            Oind = Oind.astype(np.int_)
            R = functional.getWaterRotate(pos.copy(), self._invref)
            R = R.flatten()[:6]
            box4a[Oind[0], Oind[1], Oind[2]] = np.concatenate([[1], Ooff, R])
        for pos in atom8a:
            Oind = np.floor(pos[0])
            Ooff = pos[0] - Oind
            Oind = Oind.astype(np.int_)
            R = functional.getWaterRotate(pos.copy(), self._invref)
            box8a[Oind[0], Oind[1], Oind[2]] = np.concatenate([[1], Ooff, R.flatten()[:6]])
            
        box8a = box8a[:,:,4:]
        
        box4a = torch.as_tensor(box4a, dtype=torch.float).permute(3, 2, 0, 1)
        box8a = torch.as_tensor(box8a, dtype=torch.float).permute(2, 0, 1, 3)
        
        # print(pos)
        # _, tg_pos, tg_R = functional.box2orgvec(box8a.detach().cpu().permute(1,2,0,3), 0.5, 1.0, (25.0,25.0,4.0), False, False)
        # print(R)
        # print(tg_R[-1])
        # print(tg_pos[-1])
        hfile.close()
        return file_name, box4a, box8a, torch.as_tensor([temp], dtype=torch.float)
    
class AFMGen8ADataset(Dataset):
    """
    Output is X Y Z C format

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, path: str, transform: Callable[[torch.Tensor], torch.Tensor] = None):
        self._path = path
        with h5py.File(path, 'r') as f:
            self._keys = list(f.keys())
            self._len = len(self._keys)
        self._transform = transform
        ang = 104.52 / 180 * np.pi
        initdir = np.array([
        [ 0.         , 0.         , 0.9572    ],
        [ 0.9266272  , 0.         ,-0.23998721],
        [ 0.         , 0.88696756 , 0.        ]
        ])
        self._invref = np.linalg.inv(initdir)
        self._ang = ang
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, index: int):
        # out: C Z X Y (cond, x, y, z, r1, r2, r3, r4, r5, r6)
        file_name = self._keys[index]
        hfile = h5py.File(self._path, 'r')
        data = hfile[file_name]
        atom4a = data['box_4a'][()]
        atom8a = data['box_8a'][()]
        temp = data.attrs['temp'] / 270
        box4a = np.zeros((25, 25, 4, 10))
        box8a = np.zeros((25, 25, 8, 10))
        for pos in atom4a:
            Oind = np.floor(pos[0])
            Ooff = pos[0] - Oind
            Oind = Oind.astype(np.int_)
            R = functional.getWaterRotate(pos.copy(), self._invref)
            R = R.flatten()[:6]
            box4a[Oind[0], Oind[1], Oind[2]] = np.concatenate([[1], Ooff, R])
            
        for pos in atom8a:
            Oind = np.floor(pos[0])
            Ooff = pos[0] - Oind
            Oind = Oind.astype(np.int_)
            R = functional.getWaterRotate(pos.copy(), self._invref)
            box8a[Oind[0], Oind[1], Oind[2]] = np.concatenate([[1], Ooff, R.flatten()[:6]])
                    
        box4a = torch.as_tensor(box4a, dtype=torch.float)
        box8a = torch.as_tensor(box8a, dtype=torch.float)
        
        hfile.close()
        return file_name, box4a, box8a, torch.as_tensor([temp], dtype=torch.float)

def z_sampler(use: int, total: int, is_rand: bool = False) -> tuple[int]:
    if is_rand:
        sp = random.sample(range(total), k=(use % total))
        return [i for i in range(total) for _ in range(use // total + (i in sp))]
    else:
        return [i for i in range(total) for _ in range((use // total + ((use % total) > i)))]
    
