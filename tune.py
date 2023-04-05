import os
import time
import numpy as np
from tqdm import tqdm
from itertools import chain

import torch
from torch import nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from utils.analyze_data import *
from utils.criterion import Criterion, ganLoss, grad_penalty
from utils.tools import metStat, output_target_to_imgs
from datasets.dataset import make_dataset
from utils.loader import Loader, modelLoader
from utils.logger import Logger


class Tuner():
    def __init__(self, cfg):
        self.cfg = cfg
        self.cuda = cfg.setting.device != []

        self.load_dir, self.work_dir = Loader(cfg, make_dir=False)

        self.logger = Logger(
            path=f"{self.work_dir}",
            log_name="train.log",
            elem=cfg.data.elem_name,
            split=cfg.setting.split)

        self.tb_writer = SummaryWriter(log_dir=f"{self.work_dir}/runs/Tuner")

        self.main_network = modelLoader(
            self.load_dir, self.work_dir, model_keeps=cfg.setting.max_save, cuda=True)

        log = self.main_network.load(net=cfg.model.net,
                                     inp_size=cfg.model.inp_size,
                                     out_size=cfg.model.out_size,
                                     hidden_channels=cfg.model.channels,
                                     out_feature=True)

        self.model = self.main_network.model

        self.logger.info(log)

        self.domain_network = modelLoader(
            self.load_dir, self.work_dir, load_info_name = "test.json", work_info_name = "test.json", model_keeps=3, cuda=True)

        log = self.domain_network.load(net="NLNN", in_channels=64) # 64 for late x3, 128 for early x3

        self.classicfier = self.domain_network.model

        self.analyzer = Analyzer(cfg)
        
        self.ganLoss = ganLoss()

        self.gp = grad_penalty(output_size=(
            self.cfg.setting.batch_size * self.cfg.model.inp_size[0], ))
        
        if self.cuda:
            self.analyzer.cuda()
            
            self.ganLoss.cuda()
            
            self.gp.cuda()

        self.logger.info(log)

    def fit(self):
        # --------------------------------------------------

        self.train_loader = make_dataset('train', self.cfg)

        self.valid_loader = make_dataset('valid', self.cfg)

        self.dann_loader = make_dataset('dann', self.cfg)
        
        self.main_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.setting.learning_rate)
        
        self.sec_optimizer = torch.optim.Adam(self.classicfier.parameters(), lr=self.cfg.setting.learning_rate)

        self.criterion = Criterion(self.cfg, self.cfg.setting.local_epoch)

        if self.cuda:
            self.criterion = self.criterion.cuda()

        self.best_ACC = [0.0, 0.0]

        self.best_LOSS = 999

        # --------------------------------------------------
        self.logger.info(f'Start training.')

        for epoch in tqdm(range(1, self.cfg.setting.epochs + 1), desc="Training", position=0):
            epoch_start_time = time.time()

            log_train_dic = self.train(epoch)

            if True:  # Can add some condition
                log_valid_dic = self.valid(epoch)
            else:
                log_valid_dic = {}

            self.logger.epoch_info(epoch, log_train_dic, log_valid_dic)
            for key in log_train_dic:
                self.tb_writer.add_scalar(
                    f"epoch/train_{key}", log_train_dic[key].value, epoch)

            for key in log_valid_dic:
                self.tb_writer.add_scalar(
                    f"epoch/valid_{key}", log_valid_dic[key].value, epoch)

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

        it_train_loader = iter(self.train_loader)
        it_dann_loader = iter(self.dann_loader)

        len_loader = min(len(self.dann_loader), len(self.train_loader))
        

        # -------------------------------------------

        self.model.train()
        log_dic = {'loss': metStat(), 'loss_dis': metStat(),
                   'grad': metStat(), 'loss_gan': metStat()}

        pbar = tqdm(total=len_loader - 1,
                    desc=f"Train: {epoch}", position=1, leave=True, unit='it')

        i = 0
        while i < len_loader - 1:
            # ------ Discriminator training ------ #
            self.classicfier.requires_grad_(True)
            self.model.requires_grad_(False)
            
            self.sec_optimizer.zero_grad()

            inputs_R, targets_R, _ = next(it_train_loader)

            inputs_R = inputs_R.cuda(non_blocking=True)

            with torch.no_grad():
                _, feature_pred = self.model(inputs_R)

            feature_pred = torch.transpose(feature_pred, 1, 2)
            feature_pred = torch.reshape(feature_pred, (-1, *feature_pred.shape[2:]))
            dis = self.classicfier(feature_pred)

            loss_dis = self.ganLoss(dis, False)

            # Train with target domain

            inputs_T, _ = next(it_dann_loader)

            inputs_T = inputs_T.cuda(non_blocking=True)

            with torch.no_grad():
                _, feature_real = self.model(inputs_T)

            feature_real = torch.transpose(feature_real, 1, 2)
            feature_real = torch.reshape(feature_real, (-1, *feature_real.shape[2:]))
            
            dis = self.classicfier(feature_real)

            loss_dis = loss_dis + self.ganLoss(dis, True)

            loss_dis = -loss_dis + \
                self.gp(self.classicfier, feature_pred, feature_real)

            loss_dis.backward()
            
            self.sec_optimizer.step()
            
            # ------ Generator training ------ #
            
            alpha = 0.1
            
            if (i + 1) % 5 == 0:
                self.classicfier.requires_grad_(False)
                self.model.requires_grad_(True)
                
                self.main_optimizer.zero_grad()

                targets_R = targets_R.cuda(non_blocking=True)

                predictions, feature_pred = self.model(inputs_R)

                loss = self.criterion(predictions, targets_R)

                info = self.analyzer(predictions, targets_R)

                loss = loss + self.criterion.loss_local(epoch, info)

                log_dic['loss'].add(loss)
                feature_pred = torch.transpose(feature_pred, 1, 2)
                feature_pred = torch.reshape(feature_pred, (-1, *feature_pred.shape[2:]))
                dis_pred = self.classicfier(feature_pred)

                loss_gan = self.ganLoss(dis_pred, False)

                _, feature_real = self.model(inputs_T)

                feature_real = torch.transpose(feature_real, 1, 2)
                feature_real = torch.reshape(feature_real, (-1, *feature_real.shape[2:]))
                dis_real = self.classicfier(feature_real)

                loss_gan = loss_gan + self.ganLoss(dis_real, True)

                log_dic['loss_gan'].add(loss_gan)

                loss = loss + loss_gan * alpha
                
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1000, error_if_nonfinite=True)

                self.main_optimizer.step()

                count_info = self.analyzer.count(info)

                for key in count_info:
                    if key in log_dic:
                        log_dic[key].add(count_info[key])
                    else:
                        log_dic[key] = count_info[key]
                
                log_dic['grad'].add(grad_norm)

                batch, _, Z, X, Y = inputs_R.shape
                input_img = torch.reshape(inputs_R, (batch, Z, 1, X, Y))
                input_img = make_grid(input_img[0])
                
                self.tb_writer.add_image(f"train/input img2",
                                        input_img, global_step=(epoch-1) * len_loader + i)
                imgs = output_target_to_imgs(predictions, targets_R)
                self.tb_writer.add_image(f"train/out img2", imgs,
                                        global_step=(epoch-1) * len_loader + i)
            
            log_dic['loss_dis'].add(loss_dis)
            
            for key in ['loss', 'grad', 'loss_gan', 'loss_dis']:
                if key not in log_dic:
                    continue
                self.tb_writer.add_scalar(f"train/{key}", log_dic[key].last, (epoch-1) * len_loader + i)

            if self.cfg.setting.show:
                pbar.set_postfix(**{key: log_dic[key].last for key in log_dic if key in ['loss', 'grad', 'loss_gan', 'loss_dis']})
                pbar.update(1)
                
            i += 1
        # -------------------------------------------

        pbar.close()
        
        return log_dic

    @torch.no_grad()
    def valid(self, epoch):
        # -------------------------------------------

        self.model.eval()  # 切换模型为预测模式
        log_dic = {'loss': metStat()}
        len_loader = min(len(self.valid_loader), 100)
        it_loader = iter(self.valid_loader)

        pbar = tqdm(total=len_loader - 1,
                    desc=f"Test: {epoch}", position=0, leave=True, unit='it')
        
        i = 0
        while i < len_loader - 1:
            inputs, targets, _ = next(it_loader)

            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            predictions, _ = self.model(inputs)

            loss = self.criterion(predictions, targets)

            info = self.analyzer(predictions, targets)

            loss_local = self.criterion.loss_local(epoch, info)

            loss = loss + loss_local

            log_dic['loss'].add(loss)

            count_info = self.analyzer.count(info)

            for key in count_info:
                if key in log_dic:
                    log_dic[key].add(count_info[key])
                else:
                    log_dic[key] = count_info[key]

            for key in ['loss', 'grad', 'loss_gan', 'loss_dis']:
                if key not in log_dic:
                    continue
                self.tb_writer.add_scalar(f"valid/step.{key}", log_dic[key].last, (epoch-1) * len_loader + i)
                
            if i == 0:
                batch, _, Z, X, Y = inputs.shape
                input_img = torch.reshape(inputs, (batch, Z, 1, X, Y))
                input_img = make_grid(input_img[0])
                self.tb_writer.add_image(f"valid/input img2",
                                         input_img, global_step=epoch)
                imgs = output_target_to_imgs(predictions, targets)
                self.tb_writer.add_image(f"valid/out img2", imgs,
                                         global_step=epoch)
            
            if self.cfg.setting.show:
                pbar.set_postfix(**{key: log_dic[key].last for key in log_dic if key in ['loss', 'grad', 'loss_gan', 'loss_dis']})
                pbar.update(1)

            i += 1

        # -------------------------------------------

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
        if (ele_ACC[0] > self.best_ACC[0]) and (ele_ACC[1] > self.best_ACC[1]):
            save = True
        elif (ele_ACC[0] < self.best_ACC[0]) and (ele_ACC[1] > self.best_ACC[1]):
            save = True
        else:
            save = False

        if log_loss < self.best_LOSS:
            save = True

        if True:
            self.best_ACC = ele_ACC
            self.best_LOSS = log_loss
            model_name = f"CP{epoch:02d}_"
            model_name += "_".join([f"{ele}{ACC:.4f}" for ele,
                                   ACC in zip(elem, ele_ACC)])
            model_name += f"_{self.best_LOSS:.6f}"

            self.main_network.save_model("MN", tag=model_name)

            self.domain_network.save_model("CL", tag = model_name)

            self.main_network.save_info(cfg)

            logger.info(f"Saved a new model: {model_name}")
        else:
            logger.info(f"No model was saved")
