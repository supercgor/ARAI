from itertools import accumulate

import os
import torch
import cv2
import numpy as np

from torch.utils.data.dataset import Dataset
from utils.trans_pic import *
from utils.tools import read_POSCAR, read_file
from utils.poscar import generate_target


def read_pic(path: str, 
             img_use: int | None = 10,
             model_inp_img: int = 20,
             split_layer: bool = True,
             split_border = [0,6,12],
             img_size: tuple | None = None,
             transform: bool = False, 
             shift_size: int = 5, 
             c: float = 0.04, 
             rec_size: float = 0.04, 
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
    if img_use is not None:
        if img_use > all_N:
            raise ValueError(f"Use too many images, there are only {all_N} but use {img_use}")
    else:
        img_use = all_N    
    
    if pre:
        # 把所有圖片都用了，不夠就隨機copy，直到夠20張為止
        use = np.arange(0, all_N)
        if all_N < model_inp_img:
            use = np.concatenate([use, np.random.choice(use, model_inp_img - all_N)])
        use = np.sort(use)
        
    if split_layer:
        NL = np.around(img_use * 0.4).astype(int)
        NM = np.around(img_use * 0.3).astype(int)
        NH = img_use - NL - NM
        
        outNL = np.around(model_inp_img * 0.4).astype(int)
        outNM = np.around(model_inp_img * 0.3).astype(int)
        outNH = model_inp_img - outNL - outNM
        
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
        low = max(all_N // img_use - 1, 1)  # minimum of intervals
        n = np.random.randint(0, min(all_N - low * img_use, img_use - 1) + 1)  # the number of intervals to increase
        intervals = np.full(img_use - 1, low, dtype=np.int32)
        indices = np.random.choice(np.arange(img_use - 1), n, replace=False)  # indices of intervals to increase
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
    def __init__(self, root_path, ele_name, file_list, img_size, transform, img_use ,model_inp_img, model_out):
        self.root = root_path
        self.ele_name = ele_name
        self.filenames = read_file(os.path.join(root_path,file_list))
        self.model_inp_img = model_inp_img
        self.img_use = img_use
        self.model_out = model_out
        self.transform = transform
        self.img_size = (img_size, img_size)

    def __getitem__(self, index):
        filename = self.filenames[index]
        data_path = os.path.join(self.root,'afm',filename)
        data = read_pic(data_path, 
                        img_use = self.img_use, 
                        img_size=self.img_size, 
                        transform=self.transform, 
                        model_inp_img=self.model_inp_img)
        poscar_path = os.path.join(self.root,'label',f"{filename}.poscar")
        info, positions = read_POSCAR(poscar_path)
        target = torch.from_numpy(generate_target(info, positions, self.ele_name, self.model_out)).float()
        return data, target, filename, str(poscar_path)

    def __len__(self):
        return len(self.filenames)


# build the AFM dataset class without label for prediction only
class AFMPredictDataset(Dataset):
    def __init__(self, root_path, ele_name, file_list, img_size, model_inp_img, model_out):
        self.root = root_path
        self.ele_name = ele_name
        self.filenames = read_file(os.path.join(root_path,file_list))
        self.model_inp_img = model_inp_img
        self.img_use = None
        self.model_out = model_out
        self.img_size = (img_size, img_size)

    def __getitem__(self, index):
        filename = self.filenames[index]
        data_path = os.path.join(self.root , "afm" , filename)
        data = read_pic(data_path, 
                        img_use=self.img_use, 
                        model_inp_img=self.model_inp_img, 
                        img_size=self.img_size, 
                        transform=False, 
                        pre=True)
        return data, filename

    def __len__(self):
        return len(self.filenames)


def make_dataset(mode, cfg):
    if mode == "train":
        train_dataset = AFMDataset(
            f"{cfg.path.data_root}/{cfg.data.dataset}", 
            cfg.data.elem_name, 
            file_list=cfg.data.train_filelist,
            img_size=cfg.data.img_size, 
            transform=True, 
            model_inp_img=cfg.model.input, 
            img_use = cfg.data.img_use,
            model_out=cfg.model.output)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.setting.batch_size,
            num_workers=cfg.setting.num_workers,
            pin_memory=cfg.setting.pin_memory,
            shuffle = True)
        
        return train_loader

    elif mode == "valid":
        valid_dataset = AFMDataset(
            f"{cfg.path.data_root}/{cfg.data.dataset}", 
            cfg.data.elem_name, 
            file_list=cfg.data.valid_filelist,
            img_size=cfg.data.img_size, 
            transform=True, 
            model_inp_img=cfg.model.input, 
            img_use = cfg.data.img_use,
            model_out=cfg.model.output)

        val_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=cfg.setting.batch_size,
            num_workers=cfg.setting.num_workers,
            pin_memory=cfg.setting.pin_memory)
    
        return val_loader
    
    elif mode == "test":
        test_dataset = AFMDataset(
            f"{cfg.path.data_root}/{cfg.data.dataset}", 
            cfg.data.elem_name, 
            file_list=cfg.data.test_filelist,
            img_size=cfg.data.img_size, 
            transform=True, 
            model_inp_img=cfg.model.input, 
            img_use = cfg.data.img_use,
            model_out=cfg.model.output)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.setting.batch_size,
            num_workers=cfg.setting.num_workers,
            pin_memory=cfg.setting.pin_memory)

        return test_loader
    
    elif mode == "predict":
        predict_dataset = AFMPredictDataset(
            f"{cfg.path.data_root}/{cfg.data.dataset}", 
            cfg.data.elem_name,
            file_list=cfg.data.pred_filelist,
            img_size=cfg.data.img_size, 
            transform=False, 
            model_inp_img=cfg.model.input, 
            img_use = None,
            model_out=cfg.model.output)

        predict_loader = torch.utils.data.DataLoader(
            predict_dataset,
            batch_size=1,
            num_workers=cfg.setting.num_workers,
            pin_memory=cfg.setting.pin_memory)

        return predict_loader
    else:
        raise ValueError("The mode have to be 'train', 'valid', 'test', 'predict'")