import os
import torch
from collections import OrderedDict

class ModelSaver():
    def __init__(self, max_save_num=5):
        self.max_save_num = max_save_num
        self.model_path_list = []

    def save_new_model(self, model, save_dir, parallel: bool):
        if len(self.model_path_list) == self.max_save_num:
            # model save num == threshold, delete the oldest one and save new model

            # delete the oldest model
            os.remove(self.model_path_list.pop(0))

        # save new model
        state_dict = model.state_dict()
        if parallel:
            state_dict = self._Parallel2Single(state_dict)
        torch.save(state_dict, save_dir)
        self.model_path_list.append(save_dir)

    @staticmethod
    def _Parallel2Single(state_dict):
        """
        将并行的权值参数转换为串行的权值参数
        :param state_dict : 原始串行权值参数
        :return             : 并行的权值参数
        """

        converted = OrderedDict()

        for k, v in state_dict.items():
            name = k[7:]
            converted[name] = v

        return converted
