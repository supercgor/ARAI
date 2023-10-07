import torch
import torch.nn as nn
def model_structure(model: nn.Module)-> list[str]:
    out = []
    blank = ' '
    out.append('-' * 100)
    out.append('|' + ' ' * 21 + 'weight name' + ' ' * 20 + '|'
        + ' ' * 10 + 'weight shape' + ' ' * 10 + '|'
        + ' ' * 3 + 'number' + ' ' * 3 + '|')
    out.append('-' * 100)
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

        out.append('| {} | {} | {} |'.format(key, shape, str_num))
    out.append('-' * 100)
    out.append('The total number of parameters: ' + str(num_para))
    out.append('The parameters of Model {}: {:4f}M'.format(
        model._get_name(), num_para * type_size / 1000 / 1000))
    out.append(f'The memory used now: {torch.cuda.memory_allocated() / 1024 / 1024:.2f}MB')
    out.append('-' * 100)
    return out
    
def model_save(module: nn.Module, path: str):
        try:
            state_dict = module.module.state_dict()
        except AttributeError:
            state_dict = module.state_dict()
        torch.save(state_dict, path)

def model_load(module: nn.Module, path: str, strict = False) -> list[str]:
    state_dict = module.state_dict()
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
    if strict:
        module.load_state_dict(state_dict)
    else:
        try:
            module.load_state_dict(state_dict)
        except RuntimeError:
            pass
    return match_list