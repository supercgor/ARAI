import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import h5py
import numpy as np
import tqdm
import torch

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
        real = [25, 25, 3]

        dic = {
            "real_size": real,
        }

        return  name, imgs, dic
            
if __name__ == '__main__':
    input_dir: str = ["/Volumes/HEIU/data/middle", 
                      "/Volumes/HEIU/data/bulkexp",
                      "/Volumes/HEIU/data/ice_cluster", 
                      ]
    output_path: str = f"/Volumes/HEIU/data/exp-middle-bulkexp-cluster-data.hdf5"
    all_files = [] # (path, name) path + afm + name or path + label + name.poscar
    for i in input_dir:
        print(f"Checking {i}...")
        afms = os.listdir(f"{i}/afm")
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
        for names, imges, attrs in tqdm.tqdm(dataLoader):
            for name, img, attr in zip(names, imges, attrs):
                group = h5file.create_group(name)
                img_data = group.create_dataset('img', data=img)
                img_data.attrs['shape'] = img.shape
                for k, v in attr.items():
                    group.attrs[k] = v
                    
        print(f"Successfully saved to '{output_path[:-5]}_train.hdf5'.")
        print("Done!")
    with h5py.File(f"{output_path[:-5]}_test.hdf5", 'w') as h5file:
        print("Start processing testing dataset...")
        dataReader = DataReader(test_names)
        dataLoader = torch.utils.data.DataLoader(dataReader, batch_size=3, shuffle=False, num_workers=6, collate_fn=collate_fn)
        for names, imges, attrs in tqdm.tqdm(dataLoader):
            for name, img, attr in zip(names, imges, attrs):
                group = h5file.create_group(name)
                img_data = group.create_dataset('img', data=img)
                img_data.attrs['shape'] = img.shape
                for k, v in attr.items():
                    group.attrs[k] = v
                    
        print(f"Successfully saved to '{output_path[:-5]}_test.hdf5'.")
        print("Done!")
        