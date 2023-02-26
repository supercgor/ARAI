from itertools import accumulate

import os
import torch
import cv2
import numpy as np

from torch.utils.data.dataset import Dataset
from utils.trans_pic import pixel_shift, add_noise, cutout, add_normal_noise
from utils.basic import read_POSCAR, read_file
from utils.poscar import generate_target


def read_pic(path, N=10, all_N=20, img_size=(256, 256),
             transform=False, shift_size=5, c=0.2, rec_size=500, pre=False):
    # add all the transform to the picture
    if pre:
        file_index = np.arange(0, N)
    else:
        low = max(all_N // N - 1, 1)  # minimum of intervals
        n = np.random.randint(0, min(all_N - low * N, N - 1) + 1)  # the number of intervals to increase
        intervals = np.full(N - 1, low, dtype=np.int32)
        indices = np.random.choice(np.arange(N - 1), n, replace=False)  # indices of intervals to increase
        for i in indices:
            intervals[i] += 1
        start = np.random.randint(0, all_N - np.sum(intervals))
        file_index = start + np.array(tuple(accumulate([0, *intervals])))
    IMG = []
    nt = np.random.random()
    c_i = np.random.randint(0, 2, 1)[0]
    for i in file_index:
        filename = f'{i}.png'
        img_path = os.path.join(path, filename)
        if os.path.exists(img_path):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        else:
            if pre:
                break
            else:
                raise FileExistsError(f"No such file: {img_path}")
        img = np.flip(img.T, axis=1)
        if img.shape != img_size and img_size is not None:
            img = cv2.resize(img, tuple(img_size), interpolation=cv2.INTER_AREA)
        if transform:
            shift = np.random.randint(-1 * shift_size, shift_size, 2)
            img = pixel_shift(img, shift)
            if c_i == 0:
                img = add_noise(img, c=nt * c)
            elif c_i == 1:
                img = add_normal_noise(img, c=nt * (c - 0.05), mode=1)
            else:
                raise ValueError(f"Invalid c_i:{c_i}")
            img = cutout(img, rec_size=rec_size)
        IMG.append(img)
    IMG = np.expand_dims(np.array(IMG, dtype=np.float32), axis=0) / 255
    IMG = torch.from_numpy(IMG)
    return IMG


# build the AFM dataset class for read data and label
class AFMDataset(Dataset):
    def __init__(self, root_path, ele_name, file_list, label, img_size, transform, max_Z, Z):
        self.root = root_path
        self.ele_name = ele_name
        self.filenames = read_file(os.path.join(root_path,file_list))
        self.label = label
        self.max_Z = max_Z
        self.Z = Z
        self.transform = transform
        self.img_size = (img_size, img_size)

    def __getitem__(self, index):
        filename = self.filenames[index]
        data_path = os.path.join(self.root,'data',filename)
        data = read_pic(data_path, N=self.max_Z, img_size=self.img_size, transform=self.transform)
        poscar_path = os.path.join(self.root,self.label,f"{filename}.poscar")
        info, positions = read_POSCAR(poscar_path)
        target = torch.from_numpy(generate_target(info, positions, self.ele_name, self.Z)).float()
        return data, target, filename, str(poscar_path)

    def __len__(self):
        return len(self.filenames)


# build the AFM dataset class without label for prediction only
class AFMPredictDataset(Dataset):
    def __init__(self, root_path, ele_name, file_list, img_size, max_Z, Z):
        self.root = root_path
        self.ele_name = ele_name
        self.filenames = read_file(os.path.join(root_path,file_list))
        self.max_Z = max_Z
        self.Z = Z
        self.img_size = (img_size, img_size)

    def __getitem__(self, index):
        filename = self.filenames[index]
        data_path = os.path.join(self.root , "data" , filename)
        data = read_pic(data_path, N=self.max_Z, img_size=self.img_size, transform=False, pre=True)
        return data, filename

    def __len__(self):
        return len(self.filenames)


def make_dataset(mode, cfg):
    if mode == "train":
        train_dataset = AFMDataset(
            cfg.DATA.TRAIN_PATH, 
            cfg.DATA.ELE_NAME, 
            file_list=cfg.DATA.TRAIN_FILE_LIST,
            label=cfg.DATA.LABEL_PATH,
            img_size=cfg.DATA.IMG_SIZE, transform=True, max_Z=cfg.DATA.MAX_Z,
            Z=cfg.DATA.Z)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.DATA.BATCH_SIZE,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
            shuffle = True)
        
        return train_loader

    elif mode == "valid":
        valid_dataset = AFMDataset(
            cfg.DATA.VAL_PATH, 
            cfg.DATA.ELE_NAME, 
            file_list=cfg.DATA.VAL_FILE_LIST, 
            label=cfg.DATA.LABEL_PATH,
            img_size=cfg.DATA.IMG_SIZE, 
            transform=True, 
            max_Z=cfg.DATA.MAX_Z,
            Z=cfg.DATA.Z)

        val_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=cfg.DATA.BATCH_SIZE,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY)
    
        return val_loader
    
    elif mode == "test":
        test_dataset = AFMDataset(
            cfg.DATA.TEST_PATH, 
            cfg.DATA.ELE_NAME, 
            file_list=cfg.DATA.TEST_FILE_LIST, 
            label=cfg.DATA.LABEL_PATH,
            img_size=cfg.DATA.IMG_SIZE, 
            transform=True,
            max_Z=cfg.DATA.MAX_Z,
            Z=cfg.DATA.Z)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.DATA.BATCH_SIZE,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY)

        return test_loader
    
    elif mode == "predict":
        predict_dataset = AFMPredictDataset(
            cfg.PRED.PATH, 
            cfg.DATA.ELE_NAME, 
            file_list=cfg.PRED.FILE_LIST, 
            img_size=cfg.DATA.IMG_SIZE, 
            max_Z=cfg.DATA.MAX_Z,
            Z=cfg.DATA.Z)

        predict_loader = torch.utils.data.DataLoader(
            predict_dataset,
            batch_size=1,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY)

        return predict_loader
    else:
        raise ValueError("The mode have to be 'train', 'valid', 'test', 'predict'")