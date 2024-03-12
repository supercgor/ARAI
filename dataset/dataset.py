import h5py
import einops
import random
from ase import Atoms
from ase.io import read
import numpy as np
import os

import torch
from torch.utils.data import Dataset
from typing import Callable

from utils import lib, poscar
from utils.lib import encodewater, group_as_water, vec2box
from fileio.asehdf import load_by_name
from .sampler import z_sampler, layerz_sampler

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
            water = lib.encodewater(water)
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

class PointGridDataset(Dataset):
    def __init__(self, 
                 path: str, 
                 grid_size = [25, 25, 8], 
                 reflect = [False, False, False],
                 position_offset = [0.0, 0.0, 1.3],
                 position_expand = [1.0, 1.0, 1.0],
                 ignore_H = False,
                 random_transform = False,
                 random_position_offset = [0.0, 0.0, 1.0],
                 random_position_noise = [0.0, 0.0, 0.0],
                 random_rotation = True,
                 random_remove_ratio = 0.4,
                 random_extra_noise_for_first_layer = [0.1, 0.1, 0.0],
                 random_z_compress_for_first_layer = [0.7, 1.0],
                 ):
        
        if path.endswith('.hdf5'):
            self.mode = 'hdf'
            with h5py.File(path, 'r') as f:
                self.keys = list(f.keys())
        else:
            self.mode = 'folder'
            self.keys = [os.path.join(p, f) for p, _, files in os.walk(path) for f in files]
            self.keys = list(filter(lambda x: x.endswith('xyz') or x.endswith('poscar'), self.keys))
            
        self.path = path
        self.grid_size = grid_size
        self.reflect = reflect
        self.position_offset = position_offset
        self.position_expand = position_expand
        self.ignore_H = ignore_H
        self.random_transform = random_transform
        self.random_kwargs = {
            'position_offset': random_position_offset,
            'position_noise': random_position_noise,
            'rotation': random_rotation,
            'remove_ratio': random_remove_ratio,
            'extra_noise_for_first_layer': random_extra_noise_for_first_layer,
            'z_compress_for_first_layer': random_z_compress_for_first_layer
        }
    
    def filter_key_(self, key_filter):
        self.keys = list(filter(key_filter, self.keys))
    
    def __len__(self):
        return len(self.keys)
    
    def _read(self, index: int) -> tuple[str, Atoms]:
        if self.mode == 'hdf':
            with h5py.File(self.path, 'r') as hfile:
                filename = self.keys[index]
                atoms = load_by_name(hfile, filename)
        else:
            path, filename = os.path.split(self.keys[index])
            atoms = read(os.path.join(path, filename))
            if filename.endswith('.xyz'):
                atoms.set_cell(np.diag([25, 25, 16]))
        return filename, atoms
    
    def _getitemraw(self, index: int) -> tuple[str, np.ndarray, np.ndarray]:
        filename, atoms = self._read(index)
        
        cell = np.diag(atoms.get_cell())
        
        if self.reflect[0]:
            atoms.positions[:, 0] = cell[0] - atoms.positions[:, 0]
        if self.reflect[1]:
            atoms.positions[:, 1] = cell[1] - atoms.positions[:, 1]
        if self.reflect[2]:
            atoms.positions[:, 2] = cell[2] - atoms.positions[:, 2]
        
        for i, expand in enumerate(self.position_expand):
            atoms.positions[:, i] = (atoms.positions[:, i] - cell[i] / 2) * expand + cell[i] / 2
        
        if self.ignore_H:
            atoms = atoms.get_positions()
        else:
            atoms = group_as_water(atoms.positions[atoms.symbols == 'O'], atoms.positions[atoms.symbols == 'H'])
            atoms = encodewater(atoms)
            
        atoms[:, :3] += self.position_offset
        
        if self.random_transform:
            # rotation
            if 'rotation' in self.random_kwargs:
                if self.random_kwargs['rotation']:
                    if random.getrandbits(1):
                        atoms[:, 0] = cell[0] - atoms[:, 0]
                    if random.getrandbits(1):
                        atoms[:, 1] = cell[1] - atoms[:, 1]
                    if random.getrandbits(1):
                        atoms[:, (0, 1)] = atoms[:, (1, 0)]
            # offset z
            if 'position_offset' in self.random_kwargs:
                atoms[:, :3] += np.random.uniform(-1, 1, size=(1, 3)) * self.random_kwargs['position_offset']
            
            # noise position
            if 'position_noise' in self.random_kwargs:
                atoms[:, :3] += np.random.randn(len(atoms), 3) * self.random_kwargs['position_noise']
            
            atoms = atoms[np.all(atoms[:, :3] < cell, axis = 1) & np.all(atoms[:, :3] > 0, axis = 1)]
            
            # extra noise for first layer
            if 'extra_noise_for_first_layer' in self.random_kwargs:
                mask = atoms[:, 2] <= 4
                num_top = np.sum(mask)
                atoms[mask, :3] += np.random.randn(num_top, 3) * self.random_kwargs['extra_noise_for_first_layer']
            if 'z_compress_for_first_layer' in self.random_kwargs:
                mask = atoms[:, 2] <= 4
                atoms_ref = atoms[mask, 2].max()
                low, high = self.random_kwargs['z_compress_for_first_layer']
                atoms[mask, 2] = (atoms[mask, 2] - atoms_ref) * np.random.uniform(low, high) + atoms_ref
            
            # remove atoms
            if 'remove_ratio' in self.random_kwargs:
                mask = atoms[:,2] <= 4
                num_top = mask.sum()
                num_remove = int(num_top * np.random.uniform(0, self.random_kwargs['remove_ratio']))
                idx = np.random.permutation(mask.nonzero()[0])[: num_remove]
                atoms = np.delete(atoms, idx, axis=0)

        # normalize
        atoms = atoms[(np.all(atoms[:,:3] < cell, axis=1)) & (np.all(atoms[:,:3] > 0, axis=1))]
        return filename, atoms, cell
    
    def __getitem__(self, index: int):
        filename, atoms, cell = self._getitemraw(index)
        pos = atoms.copy()
        pos[:, :3] /= cell
        pos = torch.as_tensor(pos, dtype=torch.float)
        
        grid = vec2box(pos[:,:3], pos[:, 3:], self.grid_size)
        atoms = Atoms(symbols="O" * len(atoms), positions = atoms[:, :3], cell=cell)
        # convert to tensor
        
        return filename, grid, atoms
    
def collate_fn(batch):
    filenames, grids, atoms = zip(*batch)
    return filenames, torch.stack(grids), atoms