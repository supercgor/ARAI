import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import make_grid

import random
import numpy as np
from .metrics import *
import os
import logging
import logging.handlers
import tqdm
import einops

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)  

class Logger():
    def __init__(self, path, log_name="train.log", elem=("O", "H"), split=(0, 3)):
        self.logger = self.get_logger(path, log_name)
        self.elem = elem
        self.split = [f"{split[i]}-{split[i+1]}" for i in range(len(split)-1)]

    def info(self, *arg, **args):
        self.logger.info(*arg, **args)

    def epoch_info(self, epoch, train_dic, valid_dic = None):
        info = f"\nEpoch = {epoch}" + "\n"
        info += f"Max memory use = {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MB" + "\n"
        
        for dic_name, dic in zip(["Train", "Valid"], [train_dic, valid_dic]):
            if dic is None:
                continue
            info += f"{dic_name} INFO:"
            for name, value in dic.items():
                if name != "MET":
                    info += f" {name} = {value:6.3f}"
                    # value is ElemLayerMet
            info += "\n"
            if "MET" in dic:
                value = dic["MET"]    
                for e in value.elems:
                    info += f"{e}"
                    for l in value.split:
                        info += f"\t{l}"
                        for met_name, met_format, met in zip(value.met, value.format, value[e,l]):
                            if met_format == "sum":
                                info += f" {met_name} = {met:8.0f},"
                            else:
                                info += f" {met_name} = {met:6.4f}"
                        info += "\n"
        self.info(info)
        
    def get_logger(self, save_dir, log_name):
        logger_name = "Main"
        log_path = os.path.join(save_dir, log_name)
        # 记录器
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        # 处理器
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_path, when='D', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        console_handler = TqdmLoggingHandler()
        # console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # Color code
        color = {
            "GREEN" : '\x1b[32;20m',
            "BLUE" : '\x1b[34;20m',
            "GRAY" : '\x1b[49;90m',
            "RESET" : '\x1b[0m'
        }
        # 格式化器
        file_formatter = logging.Formatter(fmt='[{asctime} - {levelname}]: {message}', datefmt='%m/%d/%Y %H:%M:%S', style='{')
        
        console = 'BLUE{name}RESET - GRAY{asctime}RESET - GREEN{levelname}RESET: {message}'
        for c in color:
            console = console.replace(c, color[c])        
        console_formatter = logging.Formatter(fmt=console, datefmt='%H:%M:%S', style='{')
        # 给处理器设置格式
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        # 给记录器添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
def label2img(x: torch.Tensor,
              use_sigmoid: bool = True,
              out_size: Tuple[int] = (4,32,32), 
              format = "BZXYEC"):
    if "B" in format:
        x = x.select(dim = format.index("B"), index = 0)
        format = format.replace("B", "")
    if "C" in format:
        x = x.select(dim = format.index("C"), index = 0)
        format = format.replace("C", "")
    format = " ".join(format)
    if use_sigmoid:
        x = torch.sigmoid(x)
    else:
        x = x.clip(0,1)
    x = einops.rearrange(x, f"{format} -> E Z X Y")
    x = F.interpolate(x[None,...], size = out_size, mode = "trilinear")
    # combine first and second dims -> (1 x (E x Z) x X x Y)
    x = x.flatten(1,2)
    x = x.permute(1,0,2,3)
    x = make_grid(x, nrow = out_size[0])
    x = x[(0,), ...]
    return x

def bnc2img(x: torch.Tensor, out_size: Tuple[int] = (4, 32, 32), format = "BCZXY", order = ("O", "H")):
    """convert output class to image

    Args:
        x (torch.Tensor): shape: (B, C, Z, X, Y)
        out_size (Tuple[int], optional): _description_. Defaults to (4, 32, 32).
    """
    if "B" in format:
        x = x.select(dim = format.index("B"), index = 0)
        format = format.replace("B", "")
    if "C" in format:
        y = torch.stack([x.select(dim = format.index("C"), index = i) for i in range(len(order))])
        format = format.replace("C", "")
    else:
        y = torch.zeros_like(x, dtype = torch.float).unsqueeze(0).repeat(len(order), 1, 1, 1)
        for i, o in enumerate(order):
            y[i][x == i] = 1
            
    # y shape (len(order), Z, X, Y)
    y = F.interpolate(y.unsqueeze(0), size = out_size, mode = "trilinear")
    # combine first and second dims -> (1 x (E x Z) x X x Y)
    y = y.flatten(1,2)
    y = y.permute(1,0,2,3)
    y = make_grid(y, nrow = out_size[0])
    y = y[(0,), ...]
    return y
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    return

set_seed(0)