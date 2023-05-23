import os
import torch
import cv2
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Union, Optional, Callable

from torch.utils.data.dataset import Dataset
from .trans_pic import stan_T, random_T
from .poscar import poscar
from .tools import indexGen, read_file, read_pic



# build the AFM dataset class for read data and label

class AFMDataset(Dataset):
    def __init__(self, 
                 root_path: str, 
                 elements: Tuple[str] = ("O", "H"), 
                 file_list: str = "train.filelist", 
                 transform = stan_T, 
                 img_use: None | int = 10,
                 model_inp: Tuple[str] = (16, 128, 128),
                 model_out: Tuple[str] = (8, 32, 32),
                 preload: bool = False,
                 fill_none: None | int = None,
                 label: bool = True,
                 random_layer = True,
                 ):
        self.root = root_path
        self.transform = transform
        self.elements = elements
        self.filenames = read_file(f"{root_path}/{file_list}")
        self.preload = preload
        self.img_use = img_use
        self.in_size = model_inp
        self.out_size = model_out
        self.random_layer = random_layer
        self.fill_none = fill_none
        self.label = label

    def __getitem__(self, index):
        filename = self.filenames[index]
        if self.preload:
            data_path = f"{self.root}/datapack/{filename}.npz"
            data_package = torch.load(data_path)
            data = data_package['imgs']
            
            max_index = data.shape[0]
            if "split" in data_package:
                split = data_package["split"]
                if split[0] == 0 or split[1] >= max_index:
                    split = None
            else:
                split = [6,12]
            indices = indexGen.get(use_len = self.img_use,
                                   out_len = self.in_size[0], 
                                   max_index = max_index, 
                                   split_border = split,
                                   rand = self.random_layer)
            
            data = data[indices]
            if self.label:
                bbox = data_package['pos']

        else:
            data_path = f"{self.root}/afm/{filename}"
            poscar_path = f"{self.root}/label/{filename}.poscar"
            if self.label:
                data_package = poscar._load_poscar(poscar_path)
                bbox = data_package['pos']
            
            max_index = len(os.listdir(data_path))
            split = None
            indices = indexGen.get(use_len = self.img_use,
                                   out_len = self.in_size[0], 
                                   max_index = max_index, 
                                   split_border = split,
                                   rand = self.random_layer)
            
            data = read_pic(data_path, indices, img_size=self.in_size[1:])
            
        if self.transform is not None:
            data = self.transform(data)
            
        data = data.permute(1,0,2,3)
                        
        if self.label:
            # transform the pos to bbox
            pos = poscar.pos2box(bbox, real_size= data_package['real_size'], out_size = self.out_size, order = self.elements)
            return data, pos, filename
        else:
            # build the AFM dataset class without label for prediction only
            return data, filename

    def __len__(self):
        return len(self.filenames)

def make_dataset(mode, cfg):
    if mode == "train":
        train_dataset = AFMDataset(
            f"{cfg.path.data_root}/{cfg.data.dataset}",
            cfg.data.elem_name,
            file_list=cfg.data.train_filelist,
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
        domain_dataset = AFMDataset(
            f"{cfg.path.data_root}/lots_exp/HDA",
            cfg.data.elem_name,
            file_list="domain1.filelist",
            preload= True, 
            label = False,
            transform = random_T,
            img_use=None)

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