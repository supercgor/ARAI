import logging
import logging.handlers
import torch
import os
import tqdm
from torchvision.utils import make_grid
from einops import rearrange, repeat, reduce

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

    def epoch_info(self, epoch, train_dic, valid_dic):
        info = f"\nEpoch = {epoch}" + "\n"
        info += f"Max memory use = {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MB" + "\n"
        
        for dic_name, dic in zip(["Train", "Valid"], [train_dic, valid_dic]):
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

def savetblogger(inputs, outputs, targets, logger, step, mode):
    B, C, Z, X, Y = outputs.shape
    t = targets[:B, :C, :Z, :X, :Y]
    x = rearrange(inputs, "B C Z X Y -> B C X (Z Y)")
    x = make_grid(x)
    logger.add_image(f"{mode}/Input Image", x, global_step=step)
    yp = rearrange(outputs, "B (C E) Z X Y -> C B 1 (E X) (Z Y)", C = 4)
    yt = rearrange(t, "B (C E) Z X Y -> C B 1 (E X) (Z Y)", C = 4)
    yp = yp[0,:2,...]
    yt = yt[0,:2,...]
    y = torch.cat([yp, yp, yt], dim = 1)
    y = make_grid(y)
    logger.add_image(f"{mode}/Out&Tag Image", y, global_step=step)