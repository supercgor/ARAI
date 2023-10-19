import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import h5py
import numpy as np
import tqdm
import torch
from utils import xyz, functional

def collate_fn(batch):
    batch = zip(*batch)
    return batch

class DataReader(torch.utils.data.Dataset):
    def __init__(self, data_labels):
        self._data_labels = data_labels
        
    def __len__(self):
        return len(self._data_labels)
    
    def __getitem__(self, index):
        name_4a: str = self._data_labels[index]
        name_8a = name_4a.replace("4A.xyz", "8A.xyz")
        name = name_4a[:-7].split("/")[-1]
        types, molecules4a, charges, ids = xyz.read(name_4a)
        offset = list(map(int,name_4a.split("_")[-4:-1]))
        # box4a = np.zeros((25, 25, 4, 10))
        # mask = np.ones((25, 25, 8), dtype=np.int_)
        # mask[:,:,:5] = 0
        # ang = 104.52 / 180 * np.pi
        # v = np.array([0, 0, 1])
        # u = np.array([np.sin(ang), 0, np.cos(ang)])
        # r = np.cross(v, u)
        # ref = np.asarray([v, u, r])
        # invref = np.linalg.inv(ref)
        # for pos in molecules:
        #     Oind = np.floor(pos[0])
        #     Ooff = pos[0] - Oind
        #     Oind = Oind.astype(np.int_)
        #     R = functional.getWaterRotate(pos.copy(), invref)
        #     R = R.flatten()[:6]
        #     box4a[Oind[0], Oind[1], Oind[2]] = np.concatenate([[1], Ooff, R])
        
        # box8a = np.zeros((25, 25, 8, 10))
        types, molecules8a, charges, ids = xyz.read(name_8a)
        # for pos in molecules:
        #     Oind = np.floor(pos[0])
        #     Ooff = pos[0] - Oind
        #     Oind = Oind.astype(np.int_)
        #     R = functional.getWaterRotate(pos.copy(), invref)
        #     R = R.flatten()[:6]
        #     box8a[Oind[0], Oind[1], Oind[2]] = np.concatenate([[1], Ooff, R])
        system = name_4a.split("/")[-3]
        temp = int(name_4a.split("/")[-2][1:])
        return  name, np.asarray(molecules4a), np.asarray(molecules8a), {"system": system, "temp": temp}
            

if __name__ == '__main__':
    input_dir: str = "/Volumes/HEIU/data/ice_8_4A"
    output_path: str = f"{input_dir}/ice_8_4A_small.hdf5"
    afm_dirnames = []
    for system in ["dposit", "hup"]:
        for temp in os.listdir(f"{input_dir}/{system}"):
            if temp.startswith("."):
                continue
            for name in os.listdir(f"{input_dir}/{system}/{temp}"):
                if name.endswith(".xyz") and not name.startswith(".") and "8A" not in name:
                    afm_dirnames.append(f"{input_dir}/{system}/{temp}/{name}")
    afm_dirnames = [i for i in afm_dirnames if not i.startswith('.') and i[:2] != '._' and 'txt' not in i]
    afm_dirnames.sort()    
    print("Enter the ratio of train/test (e.g. '0.85,0.15')")
    inp = "0.85,0.15".split(",")
    inp = [float(i) for i in inp]
    inp = [i / sum(inp) for i in inp]
    test_num = int(len(afm_dirnames) * inp[1])
    train_num = len(afm_dirnames) - test_num
    np.random.shuffle(afm_dirnames)
    train_names = afm_dirnames[:train_num]
    train_names = train_names[:100]
    train_names.sort()
    test_names = afm_dirnames[train_num:]
    test_names = test_names[:50]
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
        for names, box4as, box8as, attrs in tqdm.tqdm(dataLoader):
            for name, box4a, box8a, attr in zip(names, box4as, box8as, attrs):
                group = h5file.create_group(name)
                gp_4a = group.create_dataset('box_4a', data=box4a)
                gp_4a.attrs['size'] = (25.0, 25.0, 4.0)
                gp_4a.attrs['shape'] = box4a.shape
                gp_8a = group.create_dataset('box_8a', data=box8a)
                gp_8a.attrs['size'] = (25.0, 25.0, 8.0)
                gp_8a.attrs['shape'] = box8a.shape
                for k, v in attr.items():
                    group.attrs[k] = v
                    
        print(f"Successfully saved to '{output_path[:-5]}_train.hdf5'.")
        print("Done!")
    with h5py.File(f"{output_path[:-5]}_test.hdf5", 'w') as h5file:
        print("Start processing testing dataset...")
        dataReader = DataReader(test_names)
        dataLoader = torch.utils.data.DataLoader(dataReader, batch_size=3, shuffle=False, num_workers=6, collate_fn=collate_fn)
        for names, box4as, box8as, attrs in tqdm.tqdm(dataLoader):
            for name, box4a, box8a, attr in zip(names, box4as, box8as, attrs):
                group = h5file.create_group(name)
                gp_4a = group.create_dataset('box_4a', data=box4a)
                gp_4a.attrs['size'] = (25.0, 25.0, 4.0)
                gp_4a.attrs['shape'] = box4a.shape
                gp_8a = group.create_dataset('box_8a', data=box8a)
                gp_8a.attrs['size'] = (25.0, 25.0, 8.0)
                gp_8a.attrs['shape'] = box8a.shape
                for k, v in attr.items():
                    group.attrs[k] = v
                    
        print(f"Successfully saved to '{output_path[:-5]}_test.hdf5'.")
        print("Done!")
        