import os
import torch
import cv2
import numpy as np
from collections import OrderedDict

from . import split_layer
from torch.utils.data.dataset import Dataset
from .trans_pic import stan_T
from .poscar import poscar
from .tools import indexGen, read_file, read_pic



# build the AFM dataset class for read data and label

class AFMDataset(Dataset):
    def __init__(self, 
                 root_path, 
                 elements=("O", "H"), 
                 file_list="train.filelist", 
                 transform = stan_T, 
                 img_use=10, 
                 model_inp = (16, 128, 128),
                 model_out = (8, 32, 32),
                 preload=False,
                 fill_none: None | int = 300
                 ):
        self.root = root_path
        self.transform = transform
        self.elements = elements
        self.filenames = read_file(f"{root_path}/{file_list}")
        self.preload = preload
        self.img_use = img_use
        self.in_size = model_inp
        self.out_size = model_out
        self.ind_gen = indexGen(use_len=img_use, out_len=model_inp[0], split_border = [6, 12], split_ratio= (0.4,0.3,0.3), rand=True)
        self.fill_none = fill_none

    def __getitem__(self, index):
        filename = self.filenames[index]
        if self.preload:
            data_path = f"{self.root}/datapack/{filename}.npz"
            data_package = torch.load(data_path)
            data = data_package['imgs']
            
            max_index = data.shape[0]
            indices = self.ind_gen.get(max_index)
            
            data = data[indices]
            bbox = data_package['pos']

        else:
            data_path = f"{self.root}/afm/{filename}"
            poscar_path = f"{self.root}/label/{filename}.poscar"
            data_package = poscar._load_poscar(poscar_path)
            
            max_index = len(os.listdir(data_path))
            indices = self.ind_gen.get(max_index)
    
            data = read_pic(data_path, indices, img_size=self.in_size[1:])
            bbox = data_package['pos']
            
        # transform the pos to bbox
        pos = poscar.pos2box(bbox, real_size= data_package['real_size'], out_size = self.out_size, order = self.elements)
        
        if self.transform is not None:
            data = self.transform(data)
            
        data = data.permute(1,0,2,3)
                        
        return data, pos, filename

    def __len__(self):
        return len(self.filenames)

# build the AFM dataset class without label for prediction only
class AFMPredictDataset(Dataset):
    def __init__(self, root_path, elements=("O", "H"), file_list="pred.filelist", transform = stan_T, img_use=10, model_inp=(16, 128, 128), model_out=(4, 32, 32), preload=False):
        self.root = root_path
        self.transform = transform
        self.elements = elements
        self.filenames = read_file(f"{root_path}/{file_list}")
        self.preload = preload
        self.img_use = img_use
        self.in_size = model_inp
        self.out_size = model_out
        self.ind_gen = indexGen(use_len=img_use, out_len=model_inp[0], rand=True)

    def __getitem__(self, index):
        filename = self.filenames[index]
        if self.preload:
            data_path = f"{self.root}/datapack/{filename}.npz"
            data_package = torch.load(data_path)
            data = data_package['imgs']
            indices = self.ind_gen.get(data.shape[0])
            data = data[indices]
        else:
            data_path = f"{self.root}/afm/{filename}"
            max_index = len(os.listdir(data_path))
            indices = self.ind_gen.get(max_index)
            data = read_pic(data_path, indices, img_size=self.in_size[1:])
            
        # data = torch.permute(data, (1, 0, 2, 3))
        
        return data, filename

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