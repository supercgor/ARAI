#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filename: tools.py
# modified: 2022-11-06

import os
import torch
from optparse import OptionParser
from utils import __version__, __date__

def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
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
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)

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


def Parser():

    parser = OptionParser(
        description=f'AFM Structural Prediction v{__version__} ({__date__})',
        version=__version__,
    )

    # custom input files

    parser.add_option(
        '--mode',
        type = str,
        default = "train",
        help = 'Must be "train", "predict", "test"')
    
    parser.add_option(
        '--batch-size',
        type = int,
        help = 'the training batch-size')
    
    parser.add_option(
        '--model',
        type = str,
        help = 'loading model, example: "model_name"')

    parser.add_option(
        '--dataset',
        type = str,
        help = 'the training dataset path, example: "bulk_ice"')

    parser.add_option(
        '--train-filelist',
        type = str,
        help = 'the name of file list')

    parser.add_option(
        '--valid-filelist',
        type = str,
        help = 'the name of file list')

    parser.add_option(
        '--test-filelist',
        type = str,
        help = 'the name of file list')

    parser.add_option(
    '--pred-filelist',
    type = str,
    help = 'the name of file list')
    
    parser.add_option(
        '--worker',
        type = int,
        help = 'the number of worker')
    
    parser.add_option(
        '--gpu',
        type = str,
        help = 'using which gpu, example: 0,1')
    
    parser.add_option(
        '--local-epoch',
        type = int,
        help = 'using local loss after epoch')
    
    parser.add_option(
        '--epoch',
        type = int,
        help = 'train epoch')

    return parser