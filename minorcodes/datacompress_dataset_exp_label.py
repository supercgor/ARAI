import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import re
import h5py
import numpy as np
import tqdm
import torch
from utils import lib, xyz, poscar

def collate_fn(batch):
    batch = zip(*batch)
    return batch

class DataReader(torch.utils.data.Dataset):
    def __init__(self, data_labels):
        self._data_labels = data_labels
        
    def __len__(self):
        return len(self._data_labels)
    
    def __getitem__(self, index):
        path, name = self._data_labels[index]
        # images
        card = poscar.load(f"{path}/{name}")
        real = np.diag(card['lattice']).copy()
        dis = card['pos'] * real
        split_ = card['ion_num']
        O, H = np.split(dis, np.cumsum(split_)[:-1], axis=0)
        water = lib.group_as_water(torch.from_numpy(O), torch.from_numpy(H))
        water = water.numpy()
        water[...,(2,5,8)] *= -1
        water[...,(2,5,8)] += 4
        real[2] = 4
        # water = water[water[...,2]>1] - [0, 0, 1] * 3
        # #water = water[water[...,2]>6] - [0, 0, 6, 0, 0, 6, 0, 0, 6]
        dic = {
            "temp": 160,
            "real_size": real.tolist(),
            "size": real.tolist(),
        }

        return  name, water, dic
            
if __name__ == '__main__':
    input_dir: str = ["../data/bulkexp/result/unet_tune_v1_repair"]
    output_path: str = f"../data/bulkexp/result/unet_tune_v1_repair/unet_tune_v1_repair-HDA.hdf5"
    all_files = [] # (path, name) path + afm + name or path + label + name.poscar
    for i in input_dir:
        print(f"Checking {i}...")
        afms = os.listdir(i)
        all_files += [(i, name) for name in afms]
    all_files: list[str]
    all_files = list(filter(lambda x: not x[1].startswith(".") and not x[1].endswith(".txt"), all_files))
    all_files = list(filter(lambda x: x[1].endswith(".poscar"), all_files))
    all_files.sort(key=lambda x: x[1])
    
    print(f"Total {len(all_files)} files.")
    train_names = all_files
    train_names.sort()
    
    if os.path.exists(f"{output_path[:-5]}.hdf5"):
        print(f"Warning: Train/Test already exists, confirm to overwrite? (y/n)")
        inp = input().lower().strip()
        if inp not in ['y', 'yes', 'ok', "true", "1", "t"]:
            exit(0)
        else:
            if os.path.exists(f"{output_path[:-5]}.hdf5"):
                os.remove(f"{output_path[:-5]}.hdf5")

    with h5py.File(f"{output_path[:-5]}.hdf5", 'w') as h5file:
        dataReader = DataReader(train_names)
        dataLoader = torch.utils.data.DataLoader(dataReader, batch_size=1, shuffle=False, num_workers=6, collate_fn=collate_fn)
        print("Start processing training dataset...")
        for names, labels, attrs in tqdm.tqdm(dataLoader):
            for name, label, attr in zip(names, labels, attrs):
                group = h5file.create_group(name)
                label_data = group.create_dataset('pos', data=label)
                label_data.attrs['shape'] = label.shape
                for k, v in attr.items():
                    group.attrs[k] = v
                    
        print(f"Successfully saved to '{output_path[:-5]}.hdf5'.")
        print("Done!")
