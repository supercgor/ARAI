import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import h5py
import numpy as np
import tqdm
import torch
from utils import lib, xyz

def collate_fn(batch):
    batch = zip(*batch)
    return batch

class DataReader(torch.utils.data.Dataset):
    def __init__(self, data_labels):
        self._data_labels = data_labels
        
    def __len__(self):
        return len(self._data_labels)
    
    def __getitem__(self, index):
        path: str = self._data_labels[index]
        types, molecules, charges, ids = xyz.read(path)
        molecules = np.asarray(molecules)
        molecules[...,2] *= -1
        name = path.split("/")[-1].replace(".xyz", "")
        offset = list(map(int,name.split("_")[-3:]))
        system = "dposit" if "d" in name else "hup"
        temp = int(name.split("_")[-5].replace("T", ""))
        return  name, np.asarray(molecules), {"system": system, "temp": temp, "X": offset[0], "Y": offset[1], "Z": offset[2]}
            
if __name__ == '__main__':
    input_dir: str = "/Users/supercgor/Documents/data/ice_16A"
    output_path: str = f"{input_dir}/ice_16A.hdf5"
    afm_dirnames = []
    for system in ["dposit", "hup"]:
        for temp in os.listdir(f"{input_dir}/{system}"):
            if temp.startswith("."):
                continue
            for name in os.listdir(f"{input_dir}/{system}/{temp}"):
                if name.endswith(".xyz") and not name.startswith("."):
                    afm_dirnames.append(f"{input_dir}/{system}/{temp}/{name}")
    afm_dirnames = [i for i in afm_dirnames if not i.startswith('.') and i[:2] != '._' and 'txt' not in i]
    afm_dirnames.sort()
    print(afm_dirnames)
    print("Enter the ratio of train/test (e.g. '0.8,0.2')")
    inp = "0.8,0.2".split(",")
    inp = [float(i) for i in inp]
    inp = [i / sum(inp) for i in inp]
    test_num = int(len(afm_dirnames) * inp[1])
    train_num = len(afm_dirnames) - test_num
    np.random.shuffle(afm_dirnames)
    train_names = afm_dirnames[:train_num]
    train_names = train_names
    train_names.sort()
    test_names = afm_dirnames[train_num:]
    test_names = test_names
    test_names.sort()
    if os.path.exists(f"{output_path[:-5]}_train.hdf5") or os.path.exists(f"{output_path[:-5]}_test.hdf5"):
        print(f"Warning: Train/Test already exists, confirm to overwrite? (y/n)")
        inp = input().lower().strip()
        if inp not in ['y', 'yes', 'ok', "true", "1", "t"]:
            exit(0)
        else:
            if os.path.exists(f"{output_path[:-5]}_train.hdf5"):
                os.remove(f"{output_path[:-5]}_train.hdf5")
            if os.path.exists(f"{output_path[:-5]}_test.hdf5"):
                os.remove(f"{output_path[:-5]}_test.hdf5")
        
    with h5py.File(f"{output_path[:-5]}_train.hdf5", 'w') as h5file:
        dataReader = DataReader(train_names)
        dataLoader = torch.utils.data.DataLoader(dataReader, batch_size=3, shuffle=False, num_workers=6, collate_fn=collate_fn)
        print("Start processing training dataset...")
        for names, poses, attrs in tqdm.tqdm(dataLoader):
            for name, pos, attr in zip(names, poses, attrs):
                group = h5file.create_group(name)
                gp_4a = group.create_dataset('pos', data=pos)
                gp_4a.attrs['size'] = (25.0, 25.0, 16.0)
                gp_4a.attrs['shape'] = pos.shape
                for k, v in attr.items():
                    group.attrs[k] = v
                    
        print(f"Successfully saved to '{output_path[:-5]}_train.hdf5'.")
        print("Done!")
    with h5py.File(f"{output_path[:-5]}_test.hdf5", 'w') as h5file:
        print("Start processing testing dataset...")
        dataReader = DataReader(test_names)
        dataLoader = torch.utils.data.DataLoader(dataReader, batch_size=3, shuffle=False, num_workers=6, collate_fn=collate_fn)
        for names, poses, attrs in tqdm.tqdm(dataLoader):
            for name, pos, attr in zip(names, poses, attrs):
                group = h5file.create_group(name)
                gp_4a = group.create_dataset('pos', data=pos)
                gp_4a.attrs['size'] = (25.0, 25.0, 16.0)
                gp_4a.attrs['shape'] = pos.shape
                for k, v in attr.items():
                    group.attrs[k] = v
                    
        print(f"Successfully saved to '{output_path[:-5]}_test.hdf5'.")
        print("Done!")
        