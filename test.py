from utils.library import encodeWater, box2orgvec
from dataset import AFMDataset_V2
import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py
from torchvision.utils import make_grid
import utils
from torch.utils.data import Dataset
path = "../data/ice_16A/ice_16A_test.hdf5"

class AFMDataset(Dataset):
    def __init__(self, path: str, layer_size = [25, 25, 4], label_size= [25, 25, 16], key_filter = None):
        self._path = path
        with h5py.File(path, 'r') as f:
            self._keys = list(f.keys())
            if key_filter is not None:
                self._keys = list(filter(key_filter, self._keys))
            self._len = len(self._keys)
        self._label_size = torch.tensor(label_size)
        self._layer_size = torch.tensor(layer_size)
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, index: int):
        file_name = self._keys[index]
        out = [file_name]
        hfile = h5py.File(self._path, 'r')
        data = hfile[file_name]

        water = data['pos'][()]
        water = torch.as_tensor(water, dtype=torch.float)

        water = utils.library.decodeWater(water)
        
        water = water + torch.rand_like(water) * 0.01
        
        inp_water = water[water[:, 2] < 4.0]
        out_water = water.clone()
        inp_water[:,:3] /= self._layer_size
        out_water[:,:3] /= self._label_size
        inp_water[:,:3].clamp_max_(1 - 1E-7).clamp_min_(0)
        out_water[:,:3].clamp_max_(1 - 1E-7).clamp_min_(0)
        inp_water_ind = torch.floor(inp_water[:,:3] * self._layer_size).long()
        out_water_ind = torch.floor(out_water[:,:3] * self._label_size).long()
        inp_water[:, :3] = (inp_water[:, :3] * self._layer_size) - inp_water_ind
        out_water[:, :3] = (out_water[:, :3] * self._label_size) - out_water_ind
        print(inp_water_ind.shape, out_water_ind.shape)
        inp_water = torch.cat([torch.ones((inp_water.shape[0],1)), inp_water], axis=1)
        out_water = torch.cat([torch.ones((out_water.shape[0],1)), out_water], axis=1)
        inp_box = torch.zeros((*self._layer_size, 10))
        out_box = torch.zeros((*self._label_size, 10))
        inp_box[tuple(inp_water_ind.T)] = inp_water
        out_box[tuple(out_water_ind.T)] = out_water
        inp_box = inp_box.contiguous()
        out_box = out_box.contiguous()
        hfile.close()
        return file_name, inp_box, out_box
    
dts = AFMDataset(path)

filename, inp, out = dts[-1] 

#inp = make_grid(inp.permute(1,0,2,3))
#plt.imshow(inp.permute(1,2,0))
print(out.shape)
print(out[...,0].nonzero())
_, pos, rot = box2orgvec(inp, 0.5, 2.0, [25.0,25.0,4.0], sort = False, nms = True)
rot = rot.reshape(-1, 9)[:, :6]
make = torch.cat([pos, rot], dim = -1)
make = encodeWater(make)
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')

ax.title.set_text('inpbox')
# 取出矩陣的每一列的數據
ax.scatter(make[...,0], make[...,1], make[...,2], c='r', marker='o')
ax.scatter(make[...,3], make[...,4], make[...,5], c='b', marker='o')
ax.scatter(make[...,6], make[...,7], make[...,8], c='b', marker='o')
ax.set_aspect('equal', adjustable='box')

plt.show()