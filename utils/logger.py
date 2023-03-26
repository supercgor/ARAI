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

        info += f"Train INFO: loss = {train_dic['loss']:.10f}, grad = {train_dic['grad']:.10f}"
        for ele in self.elem:
            for split in self.split:
                key = f"{ele}-{split}-"
                info += "\n" + \
                    f"{ele}({split}A): ACC = {train_dic[f'{key}ACC']:10.8f} SUC = {train_dic[f'{key}SUC']:10.8f} TP = {train_dic[f'{key}TP']:8.0f} FP = {train_dic[f'{key}FP']:8.0f} FN = {train_dic[f'{key}FN']:8.0f}"

        info += "\n" + f"Valid INFO: loss = {valid_dic['loss']:.15f}"
        for ele in self.elem:
            for split in self.split:
                key = f"{ele}-{split}-"
                # O(0-3A):	accuracy=nan	success=1.0000	TP=0, FP=0, FN=0
                info += "\n" + \
                    f"{ele}({split}A): ACC = {valid_dic[f'{key}ACC']:10.8f} SUC = {valid_dic[f'{key}SUC']:10.8f} TP = {valid_dic[f'{key}TP']:8.0f} FP = {valid_dic[f'{key}FP']:8.0f} FN = {valid_dic[f'{key}FN']:8.0f}"

        self.info(info)

    def test_info(self, test_dic):
        info = f"\nTesting info" + "\n"
        info += f"Max memory use={torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MB" + "\n"

        info += f"Testing info: loss = {test_dic['loss']:.10}, grad = {test_dic['grad']:.10f}"
        for ele in self.elem:
            for split in self.split:
                key = f"{ele}-{split}-"
                info += "\n" + \
                    f"{ele}({split}A): ACC = {test_dic[f'{key}ACC']:10.8f} SUC = {test_dic[f'{key}SUC']:10.8f} TP = {test_dic[f'{key}TP']:8.0f} FP = {test_dic[f'{key}FP']:8.0f} FN = {test_dic[f'{key}FN']:8.0f}"

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
