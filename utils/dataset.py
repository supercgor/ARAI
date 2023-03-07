from itertools import accumulate

import os
import torch
import cv2
import numpy as np

from torch.utils.data.dataset import Dataset
from utils.trans_pic import *
from utils.basic import read_POSCAR, read_file
from utils.poscar import generate_target


def read_pic(path: str, 
             N: int | None = 10,
             out_N: int = 20,
             split_layer: bool = True,
             split_border = [0,6,12],
             img_size: tuple | None = None,
             transform: bool = False, 
             shift_size: int = 5, 
             c: float = 0.04, 
             rec_size: float = 0.1, 
             pre: bool = False) -> torch.Tensor:
    """Read .png or .jpg files from folder and return a torch tensor and it is unitized and has size ( N, Height, Width).

    Args:
        path (str): folder path e.g. '/home/supercgor/gitfile/data/bulkice/afm/T160_1'
        N (int, optional): The number of image will be used. Defaults to 10.
        out_N (int, optional): The number of output size, if N < out_N, some images will be duplicated.
        split_layer (bool, optional): To split the layer from different height
        img_size (tuple | None, optional): The Width and Height of the images. Defaults to the original size.
        transform (bool, optional): _description_. Defaults to False.
        shift_size (int, optional): _description_. Defaults to 5.
        c (float, optional): _description_. Defaults to 0.2.
        rec_size (int, optional): _description_. Defaults to 500.
        pre (bool, optional): _description_. Defaults to False.

    Raises:
        FileExistsError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # add all the transform to the picture
    all_N = len(os.listdir(path))
    if N is not None:
        if N > all_N:
            raise ValueError(f"Use too many images, there are only {all_N} but use {N}")
    else:
        N = all_N    
    
    if pre:
        # 把所有圖片都用了，不夠就隨機copy，直到夠20張為止
        use = np.arange(0, all_N)
        if all_N < out_N:
            use = np.concatenate([use, np.random.choice(use, out_N - all_N)])
        use = np.sort(use)
        
    if split_layer:
        NL = np.around(N * 0.4).astype(int)
        NM = np.around(N * 0.3).astype(int)
        NH = N - NL - NM
        
        outNL = np.around(out_N * 0.4).astype(int)
        outNM = np.around(out_N * 0.3).astype(int)
        outNH = out_N - outNL - outNM
        
        useL = list(range(split_border[0], split_border[1]))
        useM = list(range(split_border[1], split_border[2]))
        useH = list(range(split_border[2], all_N))
        
        if len(useL) > NL:
            useL = np.random.choice(useL, NL, replace = False)
        if len(useM) > NM:
            useM = np.random.choice(useM, NM, replace = False)
        if len(useH) > NH:
            useH = np.random.choice(useH, NH, replace = False)
        
        useL = np.concatenate([useL, np.random.choice(useL, outNL - len(useL))])
        useM = np.concatenate([useM, np.random.choice(useM, outNM - len(useM))])
        useH = np.concatenate([useH, np.random.choice(useH, outNH - len(useH))])
        
        use = np.sort(np.concatenate([useL, useM, useH]))
        
    else:
        low = max(all_N // N - 1, 1)  # minimum of intervals
        n = np.random.randint(0, min(all_N - low * N, N - 1) + 1)  # the number of intervals to increase
        intervals = np.full(N - 1, low, dtype=np.int32)
        indices = np.random.choice(np.arange(N - 1), n, replace=False)  # indices of intervals to increase
        for i in indices:
            intervals[i] += 1
        start = np.random.randint(0, all_N - np.sum(intervals))
        use = start + np.array(tuple(accumulate([0, *intervals])))
        
    IMG = []
    bri, con = np.random.uniform(0, 0.1, 2)
    con = con / 2
    
    for i in use:
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
            img = reduce_bnc(img, b=bri, c = con)
            img = cutout(img, rec_size=rec_size)
            img = add_noise(img, max_c = c)
        IMG.append(img)
    IMG = np.expand_dims(np.array(IMG, dtype=np.float32), axis=0) / 255
    IMG = torch.from_numpy(IMG)
    return IMG


# build the AFM dataset class for read data and label
class AFMDataset(Dataset):
    def __init__(self, root_path, ele_name, file_list, img_size, transform, N ,max_Z, Z):
        self.root = root_path
        self.ele_name = ele_name
        self.filenames = read_file(os.path.join(root_path,file_list))
        self.max_Z = max_Z
        self.N = N
        self.Z = Z
        self.transform = transform
        self.img_size = (img_size, img_size)

    def __getitem__(self, index):
        filename = self.filenames[index]
        data_path = os.path.join(self.root,'afm',filename)
        data = read_pic(data_path, N = self.N, img_size=self.img_size, transform=self.transform, out_N=10)
        poscar_path = os.path.join(self.root,'label',f"{filename}.poscar")
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
        data_path = os.path.join(self.root , "afm" , filename)
        data = read_pic(data_path, N=self.max_Z, img_size=self.img_size, transform=False, pre=True)
        return data, filename

    def __len__(self):
        return len(self.filenames)


def make_dataset(mode, cfg):
    if mode == "train":
        train_dataset = AFMDataset(
            f"{cfg.path.data_root}/{cfg.path.dataset}", 
            cfg.DATA.ELE_NAME, 
            file_list=cfg.path.train_filelist,
            img_size=cfg.DATA.IMG_SIZE, transform=True, max_Z=cfg.DATA.MAX_Z, N = cfg.DATA.N,
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
            f"{cfg.path.data_root}/{cfg.path.dataset}", 
            cfg.DATA.ELE_NAME, 
            file_list=cfg.path.valid_filelist,
            img_size=cfg.DATA.IMG_SIZE, 
            transform=True, 
            max_Z=cfg.DATA.MAX_Z, N = cfg.DATA.N,
            Z=cfg.DATA.Z)

        val_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=cfg.DATA.BATCH_SIZE,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY)
    
        return val_loader
    
    elif mode == "test":
        test_dataset = AFMDataset(
            f"{cfg.path.data_root}/{cfg.path.dataset}", 
            cfg.DATA.ELE_NAME, 
            file_list=cfg.path.test_filelist,
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
            f"{cfg.path.data_root}/{cfg.path.dataset}", 
            cfg.DATA.ELE_NAME, 
            file_list=cfg.path.pred_filelist,
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