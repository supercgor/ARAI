import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from fileio.asehdf import load_by_name
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from utils.metrics import rdf
from h5py import File
import numpy as np
from fileio.asehdf import load_by_name

class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        with h5py.File(self.path, 'r') as f:
            self.keys = list(f.keys())
        
        self.gr_ref =  np.array([0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 
                                0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 
                                0.000, 0.000, 0.135, 2.576, 6.314, 4.258, 1.486, 0.336, 0.078, 0.017, 0.003, 
                                0.003, 0.009, 0.023, 0.056, 0.130, 0.295, 0.581, 1.059, 1.718, 2.299, 2.754, 
                                2.522, 1.503, 0.677, 0.365, 0.282, 0.229, 0.179, 0.142, 0.092, 0.044])
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        k = self.keys[idx]
        with h5py.File(self.path, 'r') as f:
            atoms = load_by_name(f, k)
            atoms = atoms[atoms.symbols == 'O']
            atoms = atoms[atoms.positions[:,2] > 4]
            atoms.positions[:,2] -= 4
        gr, _ = rdf(atoms.positions, dr=0.1, dims=[25.0, 25.0, 12.0])
        return np.sqrt(((gr - self.gr_ref) ** 2).mean())

all_dist = []
path = ['../data/ice_16A_R_hup_low_T_train.hdf5', '../data/ice_16A_R_hup_low_T_test.hdf5']

for p in path:
    dts = MyDataset(p)
    dtl = DataLoader(dts, batch_size=18, num_workers=6, multiprocessing_context='fork')
    for gr in tqdm(dtl):
        all_dist.append(gr)

all_dist = np.concatenate(all_dist, axis=0)
plt.hist(all_dist, bins=100, density=True)
plt.show()