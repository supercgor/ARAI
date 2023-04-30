import os
import json
import torch
import cv2
import time
import logging
import logging.handlers
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from collections import OrderedDict

def Loader(cfg, make_dir=True):
    # 這一部份會決定是否先產生一個新的文件夾
    load_name = cfg.model.checkpoint
    if load_name != "":
        if os.path.exists(f"{cfg.path.check_root}/{load_name}"):
            f = open(f"{cfg.path.check_root}/{load_name}/info.json")
        else:
            raise NameError(f"No model in {cfg.path.check_root} is called {load_name}")
    else:
        f = open(f"{cfg.path.check_root}/default_info.json")
    
    info_dict = json.load(f)
    f.close()
    
    if make_dir:
        i = 0
        while True:
            name = f"{time.strftime('%y%m%d-%H%M%S', time.localtime())}"
            if info_dict['best'] != "" and os.path.exists(f"{cfg.path.check_root}/{name}"):
                i += 1
            else:
                if not os.path.exists(f"{cfg.path.check_root}/{name}"):
                    os.mkdir(f"{cfg.path.check_root}/{name}")
                break
    else:
        name = load_name

    cfg.merge_from_dict(info_dict)

    with open(f"{cfg.path.check_root}/{name}/info.json", "w") as f:
        json.dump(info_dict, f, indent = 4)
    
    load_dir = f"{cfg.path.check_root}/{load_name}"
    work_dir = f"{cfg.path.check_root}/{name}"
        
    return load_dir, work_dir

class sampler():
    def __init__(self, name, path="/home/supercgor/gitfile/data"):
        self.abs_path = f"{path}/{name}"
        if not os.path.exists(self.abs_path):
            raise FileNotFoundError(f"Not such dataset in {self.abs_path}")
        self.datalist = os.listdir(f"{self.abs_path}/afm")

    def __getitem__(self, index):
        img_path = f"{self.abs_path}/afm/{self.datalist[index]}"
        pl = poscarLoader(f"{self.abs_path}/label")
        info, positions = pl.load(f"{self.datalist[index]}.poscar")
        images = []
        for path in sorted(os.listdir(img_path), key=lambda x: int(x.split(".")[0])):
            img = cv2.imread(f"{img_path}/{path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)

        return {"name": self.datalist[index], "info": info, "image": images, "position": positions}

    def get(self, name):
        index = self.datalist.index(name)
        return self.__getitem__(index)

    def get_npy(self, index):
        loc = f"{self.abs_path}/npy/{self.datalist[index]}.npy"
        pred = np.load(loc)
        return pred

    def __len__(self):
        return len(self.datalist)

    def __next__(self):
        for i in range(self.__len__):
            return self.__getitem__(i)