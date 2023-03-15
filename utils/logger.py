import logging
import logging.handlers
import torch
import os

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

        info += f"Train info: loss = {train_dic['loss'].item():.15f}"
        dic = train_dic['count']
        for ele in self.elem:
            for split in self.split:
                key = f"{ele}-{split}-"
                info += "\n" + \
                    f"{ele}({split}A): ACC = {dic[f'{key}ACC'].item():10.8f} SUC = {dic[f'{key}SUC'].item():10.8f} TP = {dic[f'{key}TP'].item():8.0f} FP = {dic[f'{key}FP'].item():8.0f} FN = {dic[f'{key}FN'].item():8.0f}"

        info += "\n" + f"Valid info: loss = {valid_dic['loss'].item():.15f}"
        dic = valid_dic['count']
        for ele in self.elem:
            for split in self.split:
                key = f"{ele}-{split}-"
                # O(0-3A):	accuracy=nan	success=1.0000	TP=0, FP=0, FN=0
                info += "\n" + \
                    f"{ele}({split}A): ACC = {dic[f'{key}ACC'].item():10.8f} SUC = {dic[f'{key}SUC'].item():10.8f} TP = {dic[f'{key}TP'].item():8.0f} FP = {dic[f'{key}FP'].item():8.0f} FN = {dic[f'{key}FN'].item():8.0f}"

        self.info(info)

    def test_info(self, test_dic):
        info = f"\nTesting info" + "\n"
        info += f"Max memory use={torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MB" + "\n"

        info += f"Testing info: loss = {test_dic['loss'].item():.15f}"
        dic = test_dic['count']
        for ele in self.elem:
            for split in self.split:
                key = f"{ele}-{split}-"
                info += "\n" + \
                    f"{ele}({split}A): ACC = {dic[f'{key}ACC'].item():10.8f} SUC = {dic[f'{key}SUC'].item():10.8f} TP = {dic[f'{key}TP'].item():8.0f} FP = {dic[f'{key}FP'].item():8.0f} FN = {dic[f'{key}FN'].item():8.0f}"

        self.info(info)

    def get_logger(self, save_dir, log_name):
        logger_name = "main"
        log_path = os.path.join(save_dir, log_name)
        # 记录器
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        # 处理器
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_path, when='D', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # 格式化器
        formatter = logging.Formatter(fmt='[{asctime} - {name} - {levelname:>8s}]: {message}', datefmt='%m/%d/%Y %H:%M:%S',
                                      style='{')
        # 给处理器设置格式
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # 给记录器添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger
