import torch
from torch import nn
from .const import MODEL_DIR
import os


def gpu_condition(model, devices=None, logger=None):  # return a model and main device
    info = []
    if devices is not None:
        if torch.cuda.is_available():
            model = model.cuda()
            if isinstance(devices, list) and len(devices) > 1:
                device_name_list = []
                for device_idx in devices:
                    device_name_list.append(
                        torch.cuda.get_device_name(device_idx))
                device = f"cuda:{devices[0]}"
                model = nn.DataParallel(model, device_ids=devices)
                
                info.append(f"Data parallel, using {len(device_name_list)} GPU: {device_name_list}")
                info.append(f"Main Device: {device_name_list[0]}")
            else:   
                device = 'cuda'                
                info.append(f'Use cuda: {torch.cuda.get_device_name(device)}')
        else:
            if logger is not None:
                logger.error("No gpu! Stop training!")
            raise "No gpu! Stop training!"
    if logger is not None:
        for i in info:
            logger.info(i)
    return model, device


def model_load(model, channels, output_z, load_name=None, logger=None):
    info = []
    load_dir = os.path.join(MODEL_DIR,load_name)
    model = model(1, channels=channels, output_z=output_z)
    if load_name != "":
        try:
            model.load_state_dict(torch.load(load_dir))
            info.append(f'Load model parameters from {load_dir}')
        except RuntimeError:
            match_list = model.load_pretrained_layer(load_dir)
            info.append(f"Different model! Load match layers: {match_list}")
    else:
        model.weight_init()
        info.append(f"No model is loaded! Start training a new one")
    
    if logger is not None:
        for i in info:
            logger.info(i)
    return model


