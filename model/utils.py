import torch
import torch.nn as nn
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
        
        return self
    
    def load(self, path, pretrained = False):
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
        if pretrained:
            try:
                self.load_state_dict(state_dict)
            except RuntimeError:
                pass
        else:
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
            elif len(self.save_model) >= self.save_num:
                os.remove(self.save_model.pop(0))
            self.save_model.append(path)
     
    def structure(self):
        structure(self)
        
class basicModel(basicParts, nn.Module):
    def __init__(self, save_num: int = 3):
        nn.Module.__init__(self)
        self._name = self._get_name()
        basicParts.__init__(self, save_num)
    
class basicParallel(basicParts, nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0, save_num: int = 3):
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
    print(f'The memory used now: {torch.cuda.memory_allocated() / 1024 / 1024:.2f}MB')
    print('-' * 100)