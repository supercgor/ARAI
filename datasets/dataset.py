from itertools import accumulate

import os
import torch
import cv2
import numpy as np

from torch.utils.data.dataset import Dataset
from utils.tools import read_POSCAR, read_file
from .trans_pic import *
from .poscar import generate_target
from . import split_layer
import random
import math


class indexGen():
    def __init__(self,
                 use_len: int | None = 10,
                 out_len: int = 16,
                 split_border: tuple = (),
                 split_ratio=(0.4, 0.3, 0.3),
                 rand=False,
                 ):

        self.split_border = split_border
        self.random = rand
        if use_len is None:
            use_len = out_len
        self.use_num = self.weighted_split(use_len, split_ratio)
        self.out_num = self.weighted_split(out_len, split_ratio)

    def get(self, path):
        max_index = len(os.listdir(path))
        split = [0, *self.split_border, max_index]
        for i in split_layer:
            if i in path:
                split = [0, *split_layer[i], max_index]
                break

        use_indices = []
        for i in range(len(split) - 1):
            lower = split[i]
            upper = split[i+1]
            use = self.use_num[i]
            out = self.out_num[i]
            indices = sorted(random.sample(
                range(lower, upper), k=min((upper - lower), use)))
            use_indices.extend(indices[i] for i in self.gives_indices(
                min((upper - lower), use), out, rand=self.random))
        return use_indices

    @staticmethod
    def weighted_split(nums, ratio, mode=math.ceil):
        if len(ratio) <= 1:
            return nums,
        else:
            out = mode(nums * ratio[0]/sum(ratio))
            return out, *indexGen.weighted_split(nums - out, ratio=ratio[1:], mode=mode)

    @staticmethod
    def gives_indices(upper, select_num, offset=0, rand=False):
        if rand:
            sp = random.sample(range(upper), k=(select_num % upper))
            return [i + offset for i in range(upper) for _ in range(select_num // upper + (i in sp))]
        else:
            return [i + offset for i in range(upper) for _ in range((select_num // upper + ((select_num % upper) > i)))]


def read_pic(path: str,
             indices: list,
             img_size: tuple = (128, 128),
             transform: bool = False,
             shift_size: int = 5,
             c: float = 0.04,
             rec_size: float = 0.04) -> torch.Tensor:

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
    def __init__(self, root_path, ele_name=("O", "H"), file_list="train.filelist", transform=True, img_use=10, model_inp=(16, 128, 128), model_out=(4, 32, 32)):
        self.root = root_path
        self.transform = transform
        self.ele_name = ele_name
        self.filenames = read_file(os.path.join(root_path, file_list))
        self.model_inp_img = model_inp[0]
        self.img_use = img_use
        self.model_out = model_out[0]
        self.img_size = model_inp[1:]
        self.ind_gen = indexGen(
            use_len=img_use, out_len=model_inp[0], rand=True)

    def __getitem__(self, index):
        filename = self.filenames[index]
        data_path = os.path.join(self.root, 'afm', filename)
        indices = self.ind_gen.get(data_path)
        data = read_pic(data_path,
                        indices,
                        img_size=self.img_size,
                        transform=self.transform)
        poscar_path = os.path.join(self.root, 'label', f"{filename}.poscar")
        info, positions = read_POSCAR(poscar_path)
        target = torch.from_numpy(generate_target(
            info, positions, self.ele_name, self.model_out)).float()
        return data, target, filename

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
        self.ind_gen = indexGen(
            use_len=img_use, out_len=model_inp[0], rand=False)

    def __getitem__(self, index):
        filename = self.filenames[index]
        data_path = os.path.join(self.root, "afm", filename)
        indices = self.ind_gen.get(data_path)
        data = read_pic(data_path,
                        indices,
                        img_size=self.img_size,
                        transform=False)
        return data, filename

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

    elif mode == "dann":
        domain_dataset = AFMPredictDataset(
            f"{cfg.path.data_root}/lots_exp/HDA",
            cfg.data.elem_name,
            file_list="domain1.filelist",
            model_inp=cfg.model.inp_size,
            model_out=cfg.model.out_size)

        domain_loader = torch.utils.data.DataLoader(
            domain_dataset,
            batch_size = cfg.setting.batch_size,
            num_workers = cfg.setting.num_workers,
            pin_memory = cfg.setting.pin_memory,
            shuffle=True)
        
        return domain_loader
    else:
        raise ValueError(
            "The mode have to be 'train', 'valid', 'test', 'predict', 'dann'")

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
