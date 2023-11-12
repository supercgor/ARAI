import h5py
import einops
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from typing import Callable

from utils.library import decodeWater

class ZVarAFM(Dataset):
    def __init__(self, path: str, box_size: list[str], layer_thickness = 4.0, layer_start = -2.0, mode="random", zinp = 0, transform: Callable[[torch.Tensor], torch.Tensor] = None):
        self._path = path
        with h5py.File(path, 'r') as f:
            self._keys = list(f.keys())
            self._len = len(self._keys)
        self._box_size = box_size
        self._transform = transform
        self._layer_thickness = layer_thickness
        self._layer_start = layer_start
        self.zinp = zinp
        self._mode = mode
    
    def __len__(self):
        return self._len
    
    def get(self, index, zinp = None):
        # out: C Z X Y (cond, x, y, z, r1, r2, r3, r4, r5, r6)
        file_name = self._keys[index]
        hfile = h5py.File(self._path, 'r')
        data = hfile[file_name]
        water = data['pos'][()].reshape(-1, 3, 3)
        temp = data.attrs['temp'] / 270
        Z = data.attrs['real_size'][2]
        real_size = [*data.attrs['real_size']]
        #Z  =data['pos'].attrs['size'][2] # 16
        #real_size = [*data['pos'].attrs['size']]
        real_size[-1] = self._layer_thickness
        if zinp is None:
            zinp = np.random.uniform(self._layer_start, Z - 2 * self._layer_thickness)
        zout = zinp + self._layer_thickness
        inpbox = np.zeros((*self._box_size, 10))
        outbox = np.zeros((*self._box_size, 10))
        
        inppos = water - [0, 0, zinp]
        inppos = inppos.reshape(-1, 9)
        inppos = inppos[(inppos[:,2] >= 0) & (inppos[:,2] < self._layer_thickness)]
        outpos = water - [0,    0, zout]
        outpos = outpos.reshape(-1, 9)
        outpos = outpos[(outpos[:,2] >= 0) & (outpos[:,2] < self._layer_thickness)]
        
        inppos = decodeWater(inppos) # N * 9
        outpos = decodeWater(outpos) # M * 9
        
        inppos[:,:3] = inppos[:,:3] * self._box_size / real_size
        outpos[:,:3] = outpos[:,:3] * self._box_size / real_size
        
        inppos[:,2] = np.clip(inppos[:,2], 0, self._box_size[2] - 1E-4)
        outpos[:,2] = np.clip(outpos[:,2], 0, self._box_size[2] - 1E-4)
        
        inpind = np.floor(inppos[:,:3]).astype(int)
        outind = np.floor(outpos[:,:3]).astype(int)
        
        inppos[:,:3] -= inpind
        outpos[:,:3] -= outind
        
        inpbox[inpind[:,0], inpind[:,1], inpind[:,2]] = np.concatenate([np.ones((inppos.shape[0], 1)), inppos], axis = -1)
        outbox[outind[:,0], outind[:,1], outind[:,2]] = np.concatenate([np.ones((outpos.shape[0], 1)), outpos], axis = -1)
            
        inpbox = torch.as_tensor(inpbox, dtype=torch.float) # .permute(3, 2, 0, 1)
        outbox = torch.as_tensor(outbox, dtype=torch.float) # .permute(2, 0, 1, 3)
        Zemb = self.emb_z(zinp)
        #np.clip(x-6, 0, None)%4/10 + np.clip(x,0,6)/10, after 10, each 4 is a period.
        emb = torch.as_tensor([temp, Zemb], dtype=torch.float)       
        hfile.close()
        return file_name, inpbox, outbox, emb
    
    def emb_z(self, zinp):
        period_upbound = self._layer_thickness - self._layer_start
        period = self._layer_thickness
        zinp = zinp - self._layer_start
        return np.clip(zinp - period_upbound, 0, None) % period / (period_upbound + period) + np.clip(zinp,0,period_upbound)/(period_upbound + period)
    
    def next_emb(self, emb):
        period_upbound = self._layer_thickness - self._layer_start
        period = self._layer_thickness
        temp = emb[...,0]
        Zemb = emb[...,1]
        Zemb = torch.where(Zemb > period_upbound/(period_upbound + period), Zemb, Zemb + period/(period_upbound + period))
        return torch.as_tensor([temp, Zemb], dtype=torch.float)
    
    def __getitem__(self, index: int):
        if self._mode == "random":
            return self.get(index, None)
        else:
            return self.get(index, self.zinp)
        