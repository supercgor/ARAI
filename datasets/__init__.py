import os
import torch
from typing import List, Tuple, Union, Optional, Callable

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from .utils import *

stan_T = transforms.Compose([
    PixelShift(fill=None),
    CutOut(),
    ColorJitter(),
    Noisy(),
    Blur(),
])

random_T = transforms.Compose([
    transforms.RandomApply([PixelShift(fill=None)], p=0.7),
    transforms.RandomApply([CutOut()], p=0.7),
    transforms.RandomApply([Blur()], p=0.7),
    transforms.RandomApply([ColorJitter()], p=0.3),
    transforms.RandomApply([Noisy()], p=0.7),
])


# build the AFM dataset class for read data and label

class AFMDataset(Dataset):
    def __init__(self,
                 root_path: str,
                 elements: Tuple[str] = ("O", "H"),
                 file_list: str = "train.filelist",
                 transform=stan_T,
                 img_use: None | int = 10,
                 model_inp: Tuple[str] = (16, 128, 128),
                 model_out: Tuple[str] = (4, 32, 32),
                 preload: bool = False,
                 fill_none: None | int = None,
                 label: bool = True,
                 label_format: str = "box",
                 random_layer=True,
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
        self.label_format = label_format
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
                split = [6, 12]
            indices = indexGen.get(use_len=self.img_use,
                                   out_len=self.in_size[0],
                                   max_index=max_index,
                                   split_border=split,
                                   rand=self.random_layer)

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
            indices = indexGen.get(use_len=self.img_use,
                                   out_len=self.in_size[0],
                                   max_index=max_index,
                                   split_border=split,
                                   rand=self.random_layer)

            data = read_pic(data_path, indices, img_size=self.in_size[1:])

        if self.transform is not None:
            data = self.transform(data)

        data = data.permute(1, 0, 2, 3)

        if self.label:
            # transform the pos to bbox
            if self.label_format == "box":
                pos = poscar.pos2box(bbox, real_size=data_package['real_size'], out_size=self.out_size, order=self.elements)
                # print(pos.shape)
                return data, pos, filename
            elif self.label_format == "boxncls":
                OFFSET, CLS = poscar.pos2boxncls(bbox, real_size=data_package['real_size'], out_size=self.out_size, order=self.elements)
                return data, (OFFSET, CLS), filename
        else:
            # build the AFM dataset class without label for prediction only
            return data, filename

    def __len__(self):
        return len(self.filenames)
    
def afterTransform(data, trans):
    # data: b, c, d, h, w
    after = []
    data = data.permute(0, 2, 1, 3, 4)
    for d in data:
        after.append(trans(d))
    after = torch.stack(after)
    after = after.permute(0, 2, 1, 3, 4)
    return after
    