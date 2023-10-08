import h5py
import einops
import random

import torch
from torch.utils.data import Dataset
from typing import Callable

from utils import poscar

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
        data = h5py.File(self._path, 'r')[file_name]
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
            return file_name, afm, labeltype, labelpos
        else:
            return file_name, afm
        
    def __len__(self):
        return len(self._keys)

def z_sampler(use: int, total: int, is_rand: bool = False) -> tuple[int]:
    if is_rand:
        sp = random.sample(range(total), k=(use % total))
        return [i for i in range(total) for _ in range(use // total + (i in sp))]
    else:
        return [i for i in range(total) for _ in range((use // total + ((use % total) > i)))]