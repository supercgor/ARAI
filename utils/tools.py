#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filename: tools.py
# modified: 2022-11-06

import os
import torch
import random
import numpy as np
from optparse import OptionParser
from utils import __version__, __date__
from yacs.config import CfgNode as CN
from torch.autograd import Function
from torchvision.utils import make_grid
from collections.abc import Iterable

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    return


def model_structure(model):
    blank = ' '
    print('-' * 110)
    print('|' + ' ' * 31 + 'weight name' + ' ' * 10 + '|'
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|'
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 110)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) > 50:
            key = key.split(".")
            key = ".".join(i[:7] for i in key)
        if len(key) <= 50:
            key = key + (50 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 110)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(
        model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 110)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def absp(*paths):
    return os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), *paths)))

def condense(dic):
    output = {}
    keys = dic.keys()
    for key in keys:
        if isinstance(dic[key][0], torch.Tensor):
            output[key] = torch.stack(dic[key])
        elif isinstance(dic[key][0], dict):
            sub_dic = {}
            for subkey in dic[key][0].keys():
                sub_dic[subkey] = torch.stack(
                    [inner_dic[subkey] for inner_dic in dic[key]])
            output[key] = sub_dic
        else:
            raise TypeError(
                f"The value of {key} must be a list of tensor. Ensure all the values are list of dict or tensor.")
    return output


def read_file(file_name):
    data = []
    with open(file_name) as fr:
        for line in fr:
            fn = line.strip().replace('\t', ' ')
            fn2 = fn.split(" ")
            if fn2[0] != '':
                data.append(fn2[0])
    return data


def clean(line, splitter=' '):
    """
    clean the one line by splitter
    all the data need to do format convert
    ""splitter:: splitter in the line
    """
    data0 = []
    line = line.strip().replace('\t', ' ')
    list2 = line.split(splitter)
    for i in list2:
        if i != '':
            data0.append(i)
    temp = np.array(data0)
    return temp


def read_POSCAR(file_name):
    """
    read the POSCAR or CONTCAR of VASP FILE
    and return the data position
    """
    with open(file_name) as fr:
        comment = fr.readline()
        line = fr.readline()
        scale_length = float(clean(line)[0])
        lattice = []
        for i in range(3):
            lattice.append(clean(fr.readline()).astype(float))
        lattice = np.array(lattice)
        ele_name = clean(fr.readline())
        counts = clean(fr.readline()).astype(int)
        ele_num = dict(zip(ele_name, counts))
        fr.readline()
        fr.readline()
        positions = {}
        for ele in ele_name:
            position = []
            for _ in range(ele_num[ele]):
                line = clean(fr.readline())
                position.append(line[:3].astype(float))
            positions[ele] = np.asarray(position)
    info = {'comment': comment, 'scale': scale_length, 'lattice': lattice, 'ele_num': ele_num,
            'ele_name': tuple(ele_name)}
    return info, positions


class CfgNode(CN):
    def __init__(self):
        super(CfgNode, self).__init__()

    def merge_from_dict(self, dic):
        for key in self:
            for sub_key in self[key]:
                if sub_key in dic and dic[sub_key] is not None and dic[sub_key] != "":
                    self.merge_from_list(
                        [f"{key}.{sub_key}", f"{dic[sub_key]}"])


def fill_dict(dic, cfg: CfgNode):
    for key in cfg:
        for sub_key in cfg[key]:
            if sub_key in dic:
                dic[sub_key] = cfg[key][sub_key]
    return dic

def output_target_to_imgs(output, target):
    batch, X, Y, Z, _ = output.shape
    out_img = output[0, ..., (3, 7)] > 0
    out_img = torch.permute(out_img, (3, 2, 0, 1))  # X, Y, Z, C -> C, Z, H, W
    out_img = torch.reshape(out_img, (2, Z, 1, X, Y))
    tar_img = target[0, ..., (3, 7)]
    tar_img = torch.permute(tar_img, (3, 2, 0, 1))
    tar_img = torch.reshape(tar_img, (2, Z, 1, X, Y))
    imgs = []
    for out, tar in zip(out_img, tar_img):
        img = torch.cat([tar, tar, out], dim = 1)
        img = make_grid(img, padding = 1, pad_value = 1)
        imgs.append(img)
    imgs = make_grid(imgs, nrow = 1)
    return imgs

class metStat():
    def __init__(self, value = None, mode:str = "mean", dtype = torch.float64, device = "cpu"):
        """To help you automatically find the mean or sum, which allows us to calculate something easily and consuming less space. 
        Args:
            mode (str): should be 'mean' or 'sum'
        """
        self.n = 0
        self._dtype = dtype
        self._mode = mode
        self._device = device
        self._value = torch.tensor(0, dtype= dtype, device= device)
        self._last = self._value.item()
        if value is not None:
            self.add(value)
    
    def add(self, other):
        if isinstance(other, Iterable) or isinstance(other ,metStat):
            self.extend(other)
        else:
            self.append(other)
        
    def append(self, x):
        if isinstance(x, torch.Tensor):
            x = x.item()
        self._last = x
        self.n += 1
        if self._mode == "mean":
            self._value = self._value * ((self.n - 1) / self.n) + x * (1 / self.n)
        elif self._mode == "sum":
            self._value = self._value + x
        self._value = self._value.type(self._dtype)
    
    def extend(self, xs):
        if isinstance(xs, metStat):
            value = xs.value.to(self.device, non_blocking=True)
            self._last = value.item()
            n = len(xs)
            if self._mode == "mean":
                self._value = self._value * (self.n / (n + self.n)) + value * (n / (n + self.n))
            elif self._mode == "sum":
                self._value = self._value + value
            self.n = self.n + n
            
        else:
            if isinstance(xs, torch.Tensor) and xs.dim() == 0:
                self.append(xs)
                
            elif isinstance(xs, Iterable):
                for value in xs:
                    self.append(value)
            else:
                raise TypeError(f"{type(xs)} is not an iterable")
          
    def __repr__(self):
        return f"{self._value.item():.4f}, mode: {self._mode}, len: {self.n}"
    
    def __call__(self):
        return self._value.item()
    
    def __str__(self):
        return str(self._value.item())
    
    def __len__(self):
        return self.n
    
    def __format__(self, code):
        return self._value.item().__format__(code)
    
    @property
    def device(self):
        return self._device
    
    @property
    def value(self):
        return self._value
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def last(self):
        return self._last
    
    
def Parser():

    parser = OptionParser(
        description=f'AFM Structural Prediction v{__version__} ({__date__})',
        version=__version__,
    )

    # custom input files

    parser.add_option(
        '--batch-size',
        type=int,
        help='the training batch-size')

    parser.add_option(
        '--checkpoint',
        type=str,
        help='loading model, example: "model_name"')

    parser.add_option(
        '--dataset',
        type=str,
        help='the training dataset path, example: "bulkice"')

    parser.add_option(
        '--train-filelist',
        type=str,
        help='the name of file list')

    parser.add_option(
        '--valid-filelist',
        type=str,
        help='the name of file list')

    parser.add_option(
        '--test-filelist',
        type=str,
        help='the name of file list')

    parser.add_option(
        '--pred-filelist',
        type=str,
        help='the name of file list')

    parser.add_option(
        '--num-workers',
        type=int,
        help='the number of worker')

    parser.add_option(
        '--device',
        type=str,
        help='using which gpu, example: 0,1')

    parser.add_option(
        '--local-epoch',
        type=int,
        help='using local loss after epoch')

    parser.add_option(
        '--epochs',
        type=int,
        help='train epoch')

    return parser
