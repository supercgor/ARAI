import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import re
import h5py
import numpy as np
import tqdm
import torch
from utils import library, xyz, poscar

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
        afms_names = os.listdir(f"{path}/afm/{name}")
        afms_names = [i[:-4] for i in afms_names if i.endswith('.png') and not i.startswith('.')]
        afms_names.sort(key=lambda x: int(x))
        imgs = []
        for afm_name in afms_names:
            img = cv2.imread(f"{path}/afm/{name}/{afm_name}.png", cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 100))
            imgs.append(img.T[:,::-1])
            
        imgs = np.stack(imgs, axis = 0)
        imgs = imgs[None, ...]
        imgs = imgs / 256
        imgs = imgs.astype(np.float32)
        # poscar labels 
        card = poscar.load(f"{path}/label/{name}.poscar")
        real = np.diag(card['lattice']).copy()
        dis = card['pos'] * real
        split_ = card['ion_num']
        O, H = np.split(dis, np.cumsum(split_)[:-1], axis=0)
        water = library.group_as_water(torch.from_numpy(O), torch.from_numpy(H))
        water = water.numpy()
        if "afm3d" in path:
            real[2] -= 6
            water = water[water[...,2]>6] - [0, 0, 6, 0, 0, 6, 0, 0, 6]
            
        dic = {
            "system": "cluster" if "surface" in path else "bulk",
            "temp": int(re.search(r"T(\d+)", name).group(1)),
            "real_size": real.tolist(),
        }

        return  name, imgs, water, dic
            
if __name__ == '__main__':
    input_dir: str = [#"/gpfs/share/home/2000012508/ML2023/data/bulkice", # bulk
                      #"/gpfs/share/home/2000012508/ML2023/data/bulkiceHup", # bulk
                      #"/gpfs/share/home/2000012508/ML2023/data/surface", # cluster
                      "/gpfs/share/home/2000012508/ML2023/data/afm3d" # bulk
                      ]
    output_path: str = f"/gpfs/share/home/2000012508/Documents/data/union-water-data.hdf5"
    all_files = [] # (path, name) path + afm + name or path + label + name.poscar
    for i in input_dir:
        print(f"Checking {i}...")
        afms = os.listdir(f"{i}/afm")
        labels = os.listdir(f"{i}/label")
        missing_poscar = list(filter(lambda x: f"{x}.poscar" not in labels, afms))
        missing_afm = list(filter(lambda x: x.replace(".poscar", "") not in afms, labels))
        if missing_poscar:
            print(f"Warning: {len(missing_poscar)} poscar files are missing.")
            print(missing_poscar)
        if missing_afm:
            print(f"Warning: {len(missing_afm)} afm files are missing.")
            print(missing_afm)
        if missing_poscar or missing_afm:
            if input("Continue? (y/n)").lower().strip() not in ['y', 'yes', 'ok', "true", "1", "t"]:
                exit(0)
        for missing in missing_poscar:
            afms.remove(missing)
        for missing in missing_afm:
            labels.remove(missing)
        all_files += [(i, name) for name in afms]
    all_files: list[str]
    all_files = list(filter(lambda x: not x[1].startswith(".") and not x[1].endswith(".txt"), all_files))
    all_files.sort(key=lambda x: x[1])
    
    print(f"Total {len(all_files)} files.")
    print("Enter the ratio of train/test (e.g. '0.85,0.15')")
    inp = input().split(",")
    inp = [float(i) for i in inp]
    inp = [i / sum(inp) for i in inp]
    test_num = int(len(all_files) * inp[1])
    train_num = len(all_files) - test_num
    
    np.random.shuffle(all_files)
    train_names = all_files[:train_num]
    train_names = train_names
    train_names.sort()
    test_names = all_files[train_num:]
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
        dataLoader = torch.utils.data.DataLoader(dataReader, batch_size=1, shuffle=False, num_workers=6, collate_fn=collate_fn)
        print("Start processing training dataset...")
        for names, imges, labels, attrs in tqdm.tqdm(dataLoader):
            for name, img, label, attr in zip(names, imges, labels, attrs):
                group = h5file.create_group(name)
                img_data = group.create_dataset('img', data=img)
                img_data.attrs['shape'] = img.shape
                label_data = group.create_dataset('pos', data=label)
                label_data.attrs['shape'] = label.shape
                for k, v in attr.items():
                    group.attrs[k] = v
                    
        print(f"Successfully saved to '{output_path[:-5]}_train.hdf5'.")
        print("Done!")
    with h5py.File(f"{output_path[:-5]}_test.hdf5", 'w') as h5file:
        print("Start processing testing dataset...")
        dataReader = DataReader(test_names)
        dataLoader = torch.utils.data.DataLoader(dataReader, batch_size=3, shuffle=False, num_workers=6, collate_fn=collate_fn)
        for names, imges, labels, attrs in tqdm.tqdm(dataLoader):
            for name, img, label, attr in zip(names, imges, labels, attrs):
                group = h5file.create_group(name)
                img_data = group.create_dataset('img', data=img)
                img_data.attrs['shape'] = img.shape
                label_data = group.create_dataset('pos', data=label)
                label_data.attrs['shape'] = label.shape
                for k, v in attr.items():
                    group.attrs[k] = v
                    
        print(f"Successfully saved to '{output_path[:-5]}_test.hdf5'.")
        print("Done!")
        