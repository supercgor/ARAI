import os
import time
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from utils.analyze_data import *
from utils.criterion import Criterion
from utils.tools import metStat, output_target_to_imgs
from datasets.dataset import make_dataset
from utils.loader import Loader, modelLoader
from utils.logger import Logger

class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.cuda = cfg.setting.device != []

        self.load_dir, self.work_dir = Loader(cfg, make_dir=True)

        self.logger = Logger(
            path=f"{self.work_dir}",
            log_name="train.log",
            elem=cfg.data.elem_name,
            split=cfg.setting.split)

        i = 0
        while True:
            if not os.path.exists(f"{self.work_dir}/runs/{i}"):
                break
            else:
                i += 1

        self.tb_writer = SummaryWriter(log_dir=f"{self.work_dir}/runs/{i}")

        self.main_network = modelLoader(
            self.load_dir, self.work_dir, model_keeps=cfg.setting.max_save, cuda=True)

        log = self.main_network.load(net=cfg.model.net,
                                     inp_size=cfg.model.inp_size,
                                     out_size=cfg.model.out_size,
                                     hidden_channels=cfg.model.channels,
                                     out_feature= False)

        self.model = self.main_network.model

        self.logger.info(log)

        if self.cuda:
            self.analyzer = Analyzer(cfg).cuda()

    def fit(self):
        # --------------------------------------------------

        self.train_loader = make_dataset('train', self.cfg)

        self.valid_loader = make_dataset('valid', self.cfg)
    
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.setting.learning_rate)

        self.criterion = Criterion(self.cfg, self.cfg.setting.local_epoch)

        if self.cuda:
            self.criterion = self.criterion.cuda()

        self.best_LOSS = 999.0

        # --------------------------------------------------
        self.logger.info(f'Start training.')

        for epoch in range(1, self.cfg.setting.epochs + 1):
            epoch_start_time = time.time()

            log_train_dic = self.train(epoch)

            if True:  # Can add some condition
                log_valid_dic = self.valid(epoch)
            else:
                log_valid_dic = {}

            self.logger.epoch_info(epoch, log_train_dic, log_valid_dic)
            for key in log_train_dic:
                self.tb_writer.add_scalar(
                    f"EPOCH/TRAIN {key}", log_train_dic[key].value, epoch)

            for key in log_valid_dic:
                self.tb_writer.add_scalar(
                    f"EPOCH/VALID {key}", log_valid_dic[key].value, epoch)

            # ---------------------------------------------
            # Saver here

            self.save(epoch, log_valid_dic)

            self.logger.info(
                f"Spend time: {time.time() - epoch_start_time:.2f}s")

            self.logger.info(f'End training epoch: {epoch:0d}')

        # --------------------------------------------------

        self.logger.info(f'End training.')

    def train(self, epoch):
        # -------------------------------------------

        len_loader = len(self.train_loader)
        it_train_loader = iter(self.train_loader)

        # -------------------------------------------

        self.model.train()
        log_dic = {'loss': metStat(),
                   'grad': metStat()}

        pbar = tqdm(total=len_loader - 1,
                    desc=f"Epoch {epoch} - Train", position=0, leave=True, unit='it')

        i = 0
        while i < len_loader - 1:
            self.model.requires_grad_(True)

            inputs, targets, _ = next(it_train_loader)

            inputs = inputs.cuda(non_blocking=True)

            targets = targets.cuda(non_blocking=True)

            predictions = self.model(inputs)

            loss = self.criterion(predictions, targets)

            info = self.analyzer(predictions, targets)

            loss = loss + self.criterion.loss_local(epoch, info)

            loss.backward()
            # Train for data domain

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.setting.clip_grad, error_if_nonfinite=True)

            self.optimizer.step()

            self.optimizer.zero_grad()

            count_info = self.analyzer.count(info)

            for key in count_info:
                if key in log_dic:
                    log_dic[key].add(count_info[key])
                else:
                    log_dic[key] = count_info[key]

            log_dic['loss'].add(loss)
            log_dic['grad'].add(grad_norm)

            for key in ['loss', 'grad']:
                self.tb_writer.add_scalar(
                    f"TRAIN/{key}", log_dic[key](), (epoch-1) * len_loader + i)

            if i % 100 == 0:
                batch, _, Z, X, Y = inputs.shape
                input_img = torch.reshape(inputs, (batch, Z, 1, X, Y))
                input_img = make_grid(input_img[0])
                self.tb_writer.add_image(f"TRAIN/Input Image",
                                        input_img, global_step=(epoch-1) * len_loader + i)
                imgs = output_target_to_imgs(predictions, targets)
                self.tb_writer.add_image(f"TRAIN/Output Image", imgs,
                                        global_step=(epoch-1) * len_loader + i)

            if self.cfg.setting.show:
                pbar.set_postfix(loss=log_dic['loss'].last, grad=log_dic['grad'].last)
                pbar.update(1)
                
            i += 1
        # -------------------------------------------
        pbar.update(1)
        pbar.close()
        
        return log_dic

    @torch.no_grad()
    def valid(self, epoch):
        # -------------------------------------------

        self.model.eval()  # 切换模型为预测模式
        log_dic = {'loss': metStat()}
        len_loader = len(self.valid_loader)
        it_loader = iter(self.valid_loader)

        pbar = tqdm(total=len_loader - 1,
                    desc=f"Epoch {epoch} -  Test", position=0, leave=True, unit='it')
        
        i = 0
        while i < len_loader - 1:
            inputs, targets, _ = next(it_loader)

            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            predictions = self.model(inputs)

            loss = self.criterion(predictions, targets)

            info = self.analyzer(predictions, targets)

            loss_local = self.criterion.loss_local(epoch, info)

            loss = loss + loss_local

            count_info = self.analyzer.count(info)

            for key in count_info:
                if key in log_dic:
                    log_dic[key].add(count_info[key])
                else:
                    log_dic[key] = count_info[key]

            log_dic['loss'].add(loss)

            for key in ['loss']:
                self.tb_writer.add_scalar(
                    f"VALID/{key}", log_dic[key](), (epoch-1) * len_loader + i)

            if i == 0:
                batch, _, Z, X, Y = inputs.shape
                input_img = torch.reshape(inputs, (batch, Z, 1, X, Y))
                input_img = make_grid(input_img[0])
                self.tb_writer.add_image(f"VALID/Input Image",
                                         input_img, global_step=(epoch-1) * len_loader + i)
                imgs = output_target_to_imgs(predictions, targets)
                self.tb_writer.add_image(f"VALID/Output Image", imgs,
                                         global_step=(epoch-1) * len_loader + i)
            
            if self.cfg.setting.show:
                pbar.set_postfix(loss=log_dic['loss'].last)
                pbar.update(1)

            i += 1

        # -------------------------------------------

        pbar.update(1)
        pbar.close()

        return log_dic

    def save(self, epoch, log_dic):

        cfg = self.cfg
        log_loss = log_dic['loss']()
        logger = self.logger
        elem = cfg.data.elem_name
        split = cfg.setting.split
        split = [f"{split[i]}-{split[i+1]}" for i in range(len(split)-1)]

        # -------------------------------------------

        save = False
        ele_ACC = []
        for ele in elem:
            ele_ACC.append(min(log_dic[f"{ele}-{i}-ACC"]() for i in split))

        save = True if log_loss < self.best_LOSS else False

        if save:
            self.best_LOSS = log_loss
            model_name = f"CP{epoch:02d}_"
            model_name += "_".join([f"{ele}{ACC:.4f}" for ele,
                                   ACC in zip(elem, ele_ACC)])
            model_name += f"_{self.best_LOSS:.6f}"

            self.main_network.save_model("MN", tag=model_name)

            self.main_network.save_info(cfg)

            logger.info(f"Saved a new model: {model_name}")
        else:
            logger.info(f"No model was saved")
