from itertools import accumulate

import os
import torch
import cv2
import numpy as np

from torch.utils.data.dataset import Dataset
from utils.trans_pic import *
from utils.tools import read_POSCAR, read_file
from utils.poscar import generate_target
import random

class get_random_index():
    def __init__(self,
                 use_len: int | None = 10,
                 out_len: int = 16,
                 max_index: int = 20,
                 split_border: tuple | None = None,
                 split_ratio=(0.4, 0.3, 0.3),
                 ):
        
        self.use_len = use_len
        self.out_len = out_len
        self.max_index = max_index
        self.split_border = split_border
        self.split_ratio = split_ratio
        
        self.re_calc()

    def get(self, max_index = None, split_border = None):
        if max_index is not None and max_index != self.max_index:
            if split_border is not None:
                self.split_border = split_border
            self.max_index = max_index
            self.re_calc()
        out = []
        for lst, usage, out_use in self.layer:
            if usage > len(lst):
                lay = lst
            else:
                lay = random.sample(lst, usage)
            lay = lay * (out_use // len(lay)) + random.sample(lay, out_use % len(lay))
            out += sorted(lay)
        return out
    
    def re_calc(self):
        use_len = self.use_len
        out_len = self.out_len
        if use_len is None or use_len > out_len:
            use_len = out_len

        if self.split_border is None:
            self.split_border = (0, self.max_index)
            self.split_ratio = (1.0,)
        else:
            self.split_border = (0, *self.split_border, self.max_index)
        
        layer_usage = []  # 決定每一層用多少張圖片
        layer_out = []
        layer = []  # (lower, upper, usage, 輸出數量)
        
        for ratio in self.split_ratio:
            layer_usage.append(
                int(np.around(ratio/sum(self.split_ratio) * use_len)))
            layer_out.append(int(np.around(ratio/sum(self.split_ratio) * out_len)))
            
        if sum(layer_usage) != out_len:
            layer_usage.pop()
            layer_usage.append(use_len - sum(layer_usage))
            layer_out.pop()
            layer_out.append(out_len - sum(layer_out))

        for i, (lower, usage, out) in enumerate(zip(self.split_border[:-1], layer_usage, layer_out)):
            upper = self.split_border[i+1]
            layer.append((list(range(lower, upper)), usage, out))

        self.layer = layer

def read_pic(path: str,
             indices: list,
             img_size: tuple = (128,128),
             transform: bool = False,
             shift_size: int = 5,
             c: float = 0.04,
             rec_size: float = 0.04,
             pre: bool = False) -> torch.Tensor:
    
    IMG = []
    last = None
    bri, con = np.random.uniform(0, 0.1, 2)
    con = con / 2

    for i in indices:
        filename = f'{i}.png'
        img_path = os.path.join(path, filename)
        if i == last:
           pass
        elif os.path.exists(img_path):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = np.flip(img.T, axis=1)
            if img.shape != img_size and img_size is not None:
                img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
            last = i
            
        out = img.copy()
        if transform:
            shift = np.random.randint(-1 * shift_size, shift_size, 2)
            out = pixel_shift(out, shift)
            out = reduce_bnc(out, b=bri, c=con)
            out = cutout(out, rec_size=rec_size)
            out = add_noise(out, max_c=c)
        IMG.append(out)
        
    IMG = np.expand_dims(np.array(IMG, dtype=np.float32), axis=0) / 255
    IMG = torch.from_numpy(IMG)
    return IMG

# build the AFM dataset class for read data and label
class AFMDataset(Dataset):
    def __init__(self, root_path, ele_name=("O", "H"), file_list="train.filelist", transform = None, img_use=10, model_inp=(16, 128, 128), model_out=(4, 32, 32)):
        self.root = root_path
        self.transform = transform
        self.ele_name = ele_name
        self.filenames = read_file(os.path.join(root_path, file_list))
        self.model_inp_img = model_inp[0]
        self.img_use = img_use
        self.model_out = model_out[0]
        self.img_size = model_inp[1:]
        max_index = len(os.listdir(f"{root_path}/afm/{self.filenames[0]}"))
        self.ind_gen = get_random_index(use_len=img_use, out_len = model_inp[0], max_index= max_index, split_border = (6,12))
        self.domain_target = torch.tensor([1., 0.], dtype = torch.float32)

    def __getitem__(self, index):
        filename = self.filenames[index]
        data_path = os.path.join(self.root, 'afm', filename)
        max_index = len(os.listdir(data_path))
        indices = self.ind_gen.get(max_index = max_index)
        data = read_pic(data_path,
                        indices,
                        img_size=self.img_size,
                        transform=self.transform)
        poscar_path = os.path.join(self.root, 'label', f"{filename}.poscar")
        info, positions = read_POSCAR(poscar_path)
        target = torch.from_numpy(generate_target(
            info, positions, self.ele_name, self.model_out)).float()
        return data, target, self.domain_target, filename

    def __len__(self):
        return len(self.filenames)


# build the AFM dataset class without label for prediction only
class AFMPredictDataset(Dataset):
    def __init__(self, root_path, ele_name=("O", "H"), file_list="pred.filelist", model_inp=(16, 128, 128), img_use=10, model_out=(4, 32, 32)):
        self.root = root_path
        self.ele_name = ele_name
        self.filenames = read_file(os.path.join(root_path, file_list))
        self.model_inp_img = model_inp[0]
        self.img_use = img_use
        self.model_out = model_out[0]
        self.img_size = model_inp[1:]
        max_index = len(os.listdir(f"{root_path}/afm/{self.filenames[0]}"))
        self.ind_gen = get_random_index(use_len=img_use, out_len = model_inp[0], max_index= max_index, split_border = (6,12))
        self.domain_target = torch.tensor([0., 1.], dtype = torch.float32)

    def __getitem__(self, index):
        filename = self.filenames[index]
        data_path = os.path.join(self.root, "afm", filename)
        max_index = len(os.listdir(data_path))
        indices = self.ind_gen.get(max_index = max_index)
        data = read_pic(data_path,
                        indices,
                        img_size=self.img_size,
                        transform=None)
        return data, self.domain_target, filename

    def __len__(self):
        return len(self.filenames)


def make_dataset(mode, cfg):
    if mode == "train":
        train_dataset = AFMDataset(
            f"{cfg.path.data_root}/{cfg.data.dataset}",
            cfg.data.elem_name,
            file_list=cfg.data.train_filelist,
            transform=True,
            img_use=cfg.data.img_use,
            model_inp=cfg.model.inp_size,
            model_out=cfg.model.out_size)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.setting.batch_size,
            num_workers=cfg.setting.num_workers,
            pin_memory=cfg.setting.pin_memory,
            shuffle=True)

        return train_loader

    elif mode == "valid":
        valid_dataset = AFMDataset(
            f"{cfg.path.data_root}/{cfg.data.dataset}",
            cfg.data.elem_name,
            file_list=cfg.data.valid_filelist,
            transform=True,
            img_use=cfg.data.img_use,
            model_inp=cfg.model.inp_size,
            model_out=cfg.model.out_size)

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
            transform=True,
            img_use=cfg.data.img_use,
            model_inp=cfg.model.inp_size,
            model_out=cfg.model.out_size)

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
            model_inp=cfg.model.inp_size,
            model_out=cfg.model.out_size)

        predict_loader = torch.utils.data.DataLoader(
            predict_dataset,
            batch_size=1,
            num_workers=cfg.setting.num_workers,
            pin_memory=cfg.setting.pin_memory)

        return predict_loader
    else:
        raise ValueError(
            "The mode have to be 'train', 'valid', 'test', 'predict'")

# class newAFMDataset(Dataset):
#     def __init__(self, root_path = "/home/supercgor/gitfile/data/bulkice", 
#                  ele_name=("O", "H"), 
#                  file_list="train.filelist", 
#                  transform = transforms.ToTensor(), 
#                  img_use=10, 
#                  model_inp=(16, 128, 128), 
#                  model_out=(4, 32, 32)):
#         self.root = root_path
#         self.transform = transform
#         self.ele_name = ele_name
#         self.filenames = read_file(os.path.join(root_path, file_list))
#         max_index = len(os.listdir(f"{root_path}/afm/{self.filenames[0]}"))
#         self.ind_gen = get_random_index(use_len=img_use, out_len = model_inp[0], max_index= max_index, split_border = (6,12))
#         self.model_inp_img = model_inp[0]
#         self.img_use = img_use
#         self.model_out = model_out[0]
#         self.img_size = model_inp[1:]

#     def __getitem__(self, index):
#         filename = self.filenames[index]
#         pics_path = f"{self.root}/afm/{filename}"
#         max_index = len(os.listdir(pics_path))
#         indices = self.ind_gen.get(max_index = max_index)
#         pics = new_read_pic(pics_path, indices)
#         for _ in range(len(pics)):
#             pic = pics.pop(0)
#             pics.append(self.transform(pic))
#         pics = torch.cat(pics, dim=0)
#         pics = torch.reshape(pics, (pics.shape[0], 1, *pics.shape[1:]))
#         poscar_path = os.path.join(self.root, 'label', f"{filename}.poscar")
#         info, positions = read_POSCAR(poscar_path)
#         target = torch.from_numpy(generate_target(
#             info, positions, self.ele_name, self.model_out)).float()
#         return pics, target, filename, str(poscar_path)

#     def __len__(self):
#         return len(self.filenames)


# def new_read_pic(path, indices):
#     last = None
#     out = []
#     for i in indices:
#         if i == last:
#            out.append(img.copy()) 
#         elif os.path.exists(f"{path}/{i}.png"):
#             img = cv2.imread(f"{path}/{i}.png", cv2.IMREAD_GRAYSCALE)
#             out.append(img)
#             last = i
#     return out