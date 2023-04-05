import math
import copy
from torch.nn import Dropout, LayerNorm, Softmax, Linear, LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Function
import os

class basicParts():
    def __init__(self, save_num, *args, **kwargs):
        self.save_num = save_num
        self.save_model = []
        
    @property
    def name(self):
        return self._name
    
    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def load(self, path):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())
        pretrained_state_dict = torch.load(path)
        pretrained_param_names = list(pretrained_state_dict.keys())
        match_list = []
        for i, param in enumerate(pretrained_param_names):
            if i == len(param_names):
                break
            if param == param_names[i]:
                match_list.append(param)
                state_dict[param] = pretrained_state_dict[param]
            else:
                break
        self.load_state_dict(state_dict)
        return match_list

    def save(self, path = ..., ignore: bool = False):
        if path is Ellipsis:
            path =  "./" + self._name + ".pkl"
        try:
            state_dict = self.module.state_dict()
        except AttributeError:
            state_dict = self.state_dict()
        torch.save(state_dict, path)
        if not ignore:
            if path in self.save_model:
                    self.save_model.remove(path)
            elif len(self.save_model) == self.save_num:
                os.remove(self.save_model.pop(0))
            self.save_model.append(path)
    
    def structure(self):
        structure(self)
        
class basicModel(basicParts, nn.Module):
    def __init__(self, save_num: int = -1):
        nn.Module.__init__(self)
        self._name = self._get_name()
        basicParts.__init__(self, save_num)
    
class basicParallel(basicParts, nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0, save_num: int = -1):
        nn.DataParallel.__init__(self, module, device_ids, output_device, dim)
        self._name = module.name
        basicParts.__init__(self, save_num)
        
    def _get_name(self):
        return self._name
    
def structure(model):
    blank = ' '
    print('-' * 100)
    print('|' + ' ' * 21 + 'weight name' + ' ' * 20 + '|'
        + ' ' * 10 + 'weight shape' + ' ' * 10 + '|'
        + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 100)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) > 50:
            key = key.split(".")
            key = ".".join(i[:7] for i in key)
        if len(key) <= 50:
            key = key + (50 - len(key)) * blank
        shape = str(tuple(w_variable.shape))[1:-1]
        if len(shape) <= 30:
            shape = shape + (30 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 100)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(
        model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 100)

def conv3d(in_channels, out_channels, kernel_size, bias, padding, stride = 1):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, padding_mode='replicate', bias=bias, stride = stride)

def create_conv(in_channels, out_channels, kernel_size: int = 3, order: str = "cr", padding: tuple | int = 0, num_groups: int | None = None, stride: tuple | int = 1):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add replicate-padding added to all three sides of the input
    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(
                ('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding, stride=stride)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(
                num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(
                f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules

class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple):
    """

    def __init__(self, in_channels, out_channels, kernel_size: int = 3, order: str = "cr", padding: tuple | int = 0, num_groups: int | None = None, stride: tuple | int = 1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups = num_groups, padding = padding, stride= stride):
            self.add_module(name, module)

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

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None