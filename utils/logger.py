import logging
import logging.handlers
import torch
import os
import tqdm

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
        info += f"Max memory use={torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MB" + "\n"

        info += "Train INFO:"
        for met in train_dic.keys():
            if met != "analyse" and train_dic[met].n != 0:
                info += f" {met} = {train_dic[met]:.6f},"
        info += "\n"
        if "analyse" in train_dic:
            for ele in train_dic["analyse"].keys():
                info += f"{ele}:\n"
                for low, up in train_dic["analyse"][ele].keys():
                    info += f"\t({low}-{up}A):"
                    for met_name in train_dic["analyse"][ele][(low, up)]:
                        met = train_dic["analyse"][ele][(low, up)][met_name]()
                        if isinstance(met, int):
                            info += f" {met_name} = {met:8.0f},"
                        else:
                            info += f" {met_name} = {met:6.4f}"
                    info += "\n"
        
        info += "Valid INFO:"
        for met in valid_dic.keys():
            if met != "analyse" and valid_dic[met].n != 0:
                info += f" {met} = {valid_dic[met]:.6f},"
        info += "\n"
        if "analyse":
            for ele in valid_dic["analyse"].keys():
                info += f"{ele}:\n"
                for low, up in valid_dic["analyse"][ele].keys():
                    info += f"\t({low}-{up}A):"
                    for met_name in valid_dic["analyse"][ele][(low, up)]:
                        met = valid_dic["analyse"][ele][(low, up)][met_name]()
                        if isinstance(met, int):
                            info += f" {met_name} = {met:8.0f},"
                        else:
                            info += f" {met_name} = {met:6.4f}"
                    info += "\n"
                        
        self.info(info)

    def test_info(self, test_dic):
        info = f"\nTesting info" + "\n"
        info += f"Max memory use={torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MB" + "\n"

        for met in test_dic.keys():
            if met != "analyse":
                info += f" {met} = {test_dic[met]:.6f},"
        info += "\n"
        if "analyse" in test_dic:
            for ele in test_dic["analyse"].keys():
                info += f"{ele}:\n"
                for low, up in test_dic["analyse"][ele].keys():
                    info += f"\t({low}-{up}A):"
                    for met_name in test_dic["analyse"][ele][(low, up)]:
                        met = test_dic["analyse"][ele][(low, up)][met_name]()
                        if isinstance(met, int):
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
