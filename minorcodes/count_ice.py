
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import h5py
from fileio.asehdf import load_by_name
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from utils.metrics import ice_rule_counter
import numpy as np
from fileio.asehdf import load_by_name

class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        with h5py.File(self.path, 'r') as f:
            self.keys = list(f.keys())
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        k = self.keys[idx]
        with h5py.File(self.path, 'r') as f:
            atoms = load_by_name(f, k)
            atoms = atoms[atoms.symbols == 'O']
        return ice_rule_counter(atoms)
        

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