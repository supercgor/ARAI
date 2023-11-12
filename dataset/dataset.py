import h5py
import einops
import random
import numpy as np
import os

import torch
from torch.utils.data import Dataset
from typing import Callable

from utils import library, poscar, xyz
from .sampler import z_sampler, layerz_sampler
from functools import partial

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

class AFMDataset_V2(Dataset):
    def __init__(self, path: str, useLabel = True, useEmb = True, label_size= [25, 25, 3], useZ = 10, transform = [], key_filter = None, sampler = None):
        self._path = path
        with h5py.File(path, 'r') as f:
            self._keys = list(f.keys())
            if key_filter is not None:
                self._keys = list(filter(key_filter, self._keys))
            self._len = len(self._keys)
        self._useLabel = useLabel
        self._label_size = torch.tensor(label_size)
        self._useEmb = useEmb
        self._useZ = useZ
        self._transform = transform
        self.sampler = layerz_sampler if sampler is None else sampler
        
    def __getitem__(self, index: int):
        file_name = self._keys[index]
        out = [file_name]
        hfile = h5py.File(self._path, 'r')
        data = hfile[file_name]
        totalZ = data['img'].shape[1]
        afm =data['img'][()]
        ind = self.sampler(self._useZ, totalZ, is_rand=True)
        afm = afm[:, ind]
            
        afm = torch.as_tensor(afm, dtype=torch.float)    
        if self._transform != []:
            afm = self._transform[0](afm)
        out.append(afm)
        
        if self._useEmb:
            temp = data.attrs['temp'] / 270
            sys = 1 if data.attrs['system'] == 'bulk' else 0
            emb = torch.as_tensor([temp, sys], dtype=torch.float)
            out.append(emb)
        
        if self._useLabel:
            water = data['pos'][()]
            water = torch.as_tensor(water, dtype=torch.float)
            if self._transform != []:
                water = self._transform[1](water)
            water = library.decodeWater(water)
            water[:,:3] /= torch.tensor(data.attrs['real_size'])
            water[:,:3].clamp_max_(1 - 1E-7).clamp_min_(0)
            waterind = torch.floor(water[:,:3] * self._label_size).long()
            water[:,:3] = (water[:, :3] * self._label_size) - waterind
            water = torch.cat([torch.ones((water.shape[0],1)), water], axis=1)
            box = torch.zeros((*self._label_size, 10))
            box[tuple(waterind.T)] = water
            box = torch.as_tensor(box, dtype=torch.float)
            out.append(box)
            out.append(0)
            
        hfile.close()
        return out
        
    def __len__(self):
        return len(self._keys)

class Point_Grid_Dataset_hdf(Dataset):
    def __init__(self, path: str, layer_size = [25, 25, 4], label_size = [25, 25, 16], key_filter = None, flipz = None, noise_position = 0.0):
        self._path = path
        with h5py.File(path, 'r') as f:
            self._keys = list(f.keys())
            if key_filter is not None:
                self._keys = list(filter(key_filter, self._keys))
            self._len = len(self._keys)
        if label_size is not None:
            self._label_size = torch.tensor(label_size)
        else:
            self._label_size = None
        
        self._layer_size = torch.tensor(layer_size)
        
        self._noise_position = noise_position
        self._flipz = flipz
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, index: int):
        file_name = self._keys[index]
        out = [file_name]
        hfile = h5py.File(self._path, 'r')
        data = hfile[file_name]

        if 'pos' in data.keys():
            water = data['pos'][()]
        elif 'label' in data.keys():
            water = data['label'][()]
        else:
            raise KeyError(f'No position data found, Found keys: {data.keys()}')
        water = torch.as_tensor(water, dtype=torch.float)

        if self._flipz is not None:
            water[:, 2::3] = self._flipz - water[:, 2::3]
        water = library.decodeWater(water)

        water = water + torch.rand_like(water) * self._noise_position
        
        inp_water = water[water[:, 2] < 4.0].clone()
        
        inp_water[:,:3] /= self._layer_size
        inp_water[:,:3].clamp_max_(1 - 1E-7).clamp_min_(0)
        inp_water_ind = torch.floor(inp_water[:,:3] * self._layer_size).long()
        inp_water[:, :3] = (inp_water[:, :3] * self._layer_size) - inp_water_ind
        inp_water = torch.cat([torch.ones((inp_water.shape[0],1)), inp_water], axis=1)
        inp_box = torch.zeros((*self._layer_size, 10))
        inp_box[tuple(inp_water_ind.T)] = inp_water
        inp_box = inp_box.contiguous()
        out.append(inp_box)
        
        if self._label_size is not None:
            out_water = water.clone()
            out_water[:,:3] /= self._label_size
            out_water[:,:3].clamp_max_(1 - 1E-7).clamp_min_(0)
            out_water_ind = torch.floor(out_water[:,:3] * self._label_size).long()
            out_water[:, :3] = (out_water[:, :3] * self._label_size) - out_water_ind
            out_water = torch.cat([torch.ones((out_water.shape[0],1)), out_water], axis=1)
            out_box = torch.zeros((*self._label_size, 10))
            out_box[tuple(out_water_ind.T)] = out_water
            out_box = out_box.contiguous()
            out.append(out_box)
            
        hfile.close()
        
        return out
    
class Point_Grid_Dataset_folder(Dataset):
    def __init__(self, path: str, layer_size = [25, 25, 4], label_size = [25, 25, 16], key_filter = None, flipz = None, noise_position = 0.0):
        self._path = path
        self._keys = os.listdir(self._path)
        self._keys = list(filter(lambda x: '.xyz' in x or '.poscar' in x, self._keys))
        self._len = len(self._keys)
        if label_size is not None:
            self._label_size = torch.tensor(label_size)
        else:
            self._label_size = None
        
        self._layer_size = torch.tensor(layer_size)
        
        self._noise_position = noise_position
        self._flipz = flipz
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, index: int):
        file_name = self._keys[index]
        out = [file_name]
        if file_name.endswith('.xyz'):
            _, water, _, _ = xyz.read(os.path.join(self._path, file_name))
            water = torch.as_tensor(water, dtype=torch.float)

        elif file_name.endswith('.poscar'):
            dic = poscar.load(os.path.join(self._path, file_name))
            ion_pos, ion_num = dic['pos'] * np.diag(dic['lattice']), dic['ion_num']
            ion_pos = torch.as_tensor(ion_pos, dtype=torch.float)
            ion_pos = torch.split(ion_pos, ion_num)
            water = library.group_as_water(ion_pos[0], ion_pos[1])

        if self._flipz is not None:
            water[:, 2::3] = self._flipz - water[:, 2::3]
        water = library.decodeWater(water)

        water = water + torch.rand_like(water) * self._noise_position
        
        inp_water = water[water[:, 2] < 4.0].clone()
        
        inp_water[:,:3] /= self._layer_size
        inp_water[:,:3].clamp_max_(1 - 1E-7).clamp_min_(0)
        inp_water_ind = torch.floor(inp_water[:,:3] * self._layer_size).long()
        inp_water[:, :3] = (inp_water[:, :3] * self._layer_size) - inp_water_ind
        inp_water = torch.cat([torch.ones((inp_water.shape[0],1)), inp_water], axis=1)
        inp_box = torch.zeros((*self._layer_size, 10))
        inp_box[tuple(inp_water_ind.T)] = inp_water
        inp_box = inp_box.contiguous()
        out.append(inp_box)
        
        if self._label_size is not None:
            out_water = water.clone()
            out_water[:,:3] /= self._label_size
            out_water[:,:3].clamp_max_(1 - 1E-7).clamp_min_(0)
            out_water_ind = torch.floor(out_water[:,:3] * self._label_size).long()
            out_water[:, :3] = (out_water[:, :3] * self._label_size) - out_water_ind
            out_water = torch.cat([torch.ones((out_water.shape[0],1)), out_water], axis=1)
            out_box = torch.zeros((*self._label_size, 10))
            out_box[tuple(out_water_ind.T)] = out_water
            out_box = out_box.contiguous()
            out.append(out_box)
            
        return out