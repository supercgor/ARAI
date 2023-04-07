import os
import time
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import json
import math


import torch
from einops import rearrange
from torch import nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import network
from network.basic import basicParallel

from utils.analyze_data import analyse, FIDQ3D
from utils.criterion import basicLoss, wassersteinLoss, grad_penalty
from utils.tools import metStat, fill_dict
from datasets.dataset import make_dataset
from utils.loader import Loader
from utils.logger import Logger


class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.times_DET = 1
        self.times_GAN = 0
        self.times_DISC = 0
        
        self.cuda = cfg.setting.device != []

        self.load_dir, self.work_dir = Loader(cfg, make_dir=True)

        self.logger = Logger(path= self.work_dir, elem=cfg.data.elem_name, split=cfg.setting.split)

        self.tb_writer = SummaryWriter(log_dir=f"{self.work_dir}/runs/{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

        log = []
        self.model = {}
        if self.times_DET:
            try:
                self.DET = network.UNet3D().cuda() if self.cuda else network.UNet3D()
                self.DET.requires_grad_(False)
                self.loss_DET = basicLoss().cuda() if self.cuda else basicLoss()
                
                path = f"{self.load_dir}/{cfg.model.DET}"
                self.DET.load(path)
                log.append(f"Load Detector parameters from {path}")
            except (FileNotFoundError, IsADirectoryError):
                self.DET.init()
                log.append(f"No Detector is loaded, start a new model: {self.DET.name}")
            if len(cfg.setting.device) >= 2:
                self.DET = basicParallel(self.DET, cfg.seting.device)
            
            self.model["DET"] = self.DET
            self.OPT_DET = torch.optim.Adam(self.DET.parameters(), lr=self.cfg.setting.learning_rate)
                
        if self.times_GAN:
            try:
                self.GAN = network.StyleGAN3D().cuda() if self.cuda else network.StyleGAN3D()
                self.GAN.requires_grad_(False)
                self.loss_GAN = wassersteinLoss(alpha = 1)
                
                path = f"{self.load_dir}/{cfg.model.GAN}"
                self.GAN.load(path)
                log.append(f"Load Generator parameters from {path}")
            except FileNotFoundError:
                self.GAN.init()
                log.append(f"No Generator is loaded, start a new model: {self.GAN.name}")
            if len(cfg.setting.device) >= 2:
                self.GAN = basicParallel(self.GAN, cfg.seting.device)
                
            self.model["GAN"] = self.GAN
            self.OPT_GAN = torch.optim.Adam(self.GAN.parameters(), lr=self.cfg.setting.learning_rate)
        
        if self.times_DISC:
            try:
                self.DISC = network.Discriminator3D().cuda() if self.cuda else network.Discriminator3D()
                self.DISC.requires_grad_(False)
                self.loss_DISC = wassersteinLoss(alpha = -1)
                self.loss_GP = grad_penalty(net = self.DISC)
                
                path = f"{self.load_dir}/{cfg.model.DISC}"
                self.DISC.load(path)
                log.append(f"Load Discriminator parameters from {path}")
            except FileNotFoundError:
                self.DISC.init()
                log.append(f"No Discriminator is loaded, start a new model: {self.DISC.name}")
            if len(cfg.setting.device) >= 2:
                self.DISC = basicParallel(self.DISC, cfg.seting.device)
            
            self.model["DISC"] = self.DISC
            self.OPT_DISC = torch.optim.Adam(self.DISC.parameters(), lr=self.cfg.setting.learning_rate)
                
        self.analyse = analyse().cuda() if self.cuda else analyse()
        
        self.FID = FIDQ3D(feature = 192).cuda() if self.cuda else FIDQ3D(feature= 192)
        self.FID.requires_grad_(False)

        for l in log:
            self.logger.info(l)

    def fit(self):
        # --------------------------------------------------

        self.train_loader = make_dataset('train', self.cfg)

        self.valid_loader = make_dataset('valid', self.cfg)

        self.best = {'loss': metStat(mode = "min"), 'FID': metStat(mode = "min")}
        self.best_met = 9999

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
            for key, MET in log_train_dic.items():
                if key != 'analyse':
                    self.tb_writer.add_scalar(f"EPOCH/TRAIN {key}", MET.value, epoch)
                else:
                    for e_name, sub in MET.items():
                        for layer, subsub in sub.items():
                            for met_name, met in subsub.items():
                                self.tb_writer.add_scalar(f"EPOCH/TRAIN {e_name} {layer} {met_name}", met.value, epoch)

            for key, MET in log_valid_dic.items():
                if key != 'analyse':
                    self.tb_writer.add_scalar(f"EPOCH/VALID {key}", MET.value, epoch)
                else:
                    for e_name, sub in MET.items():
                        for layer, subsub in sub.items():
                            for met_name, met in subsub.items():
                                self.tb_writer.add_scalar(f"EPOCH/VALID {e_name} {layer} {met_name}", met.value, epoch)

            # ---------------------------------------------
            # Saver here

            self.save(epoch, log_valid_dic)

            self.logger.info(
                f"Spend time: {time.time() - epoch_start_time:.2f}s")

            self.logger.info(f'End training epoch: {epoch:0d}')

        # --------------------------------------------------

        self.logger.info(f'End training.')
    
    @staticmethod
    def get_dict():
        return OrderedDict(
            grad_DET = metStat(mode = "mean"),
            grad_DISC = metStat(mode = "mean"),
            grad_GAN = metStat(mode = "mean"),
            FID = metStat(mode = "mean"),
            loss_GAN = metStat(mode = "mean"),
            loss_GP = metStat(mode = "mean"),
            loss_DISC = metStat(mode = "mean"),
            loss_DET = metStat(mode = "mean"),
        )

    def train(self, epoch):
        # -------------------------------------------
        
        len_loader = len(self.train_loader)
        
        len_loader = len_loader - len_loader % (self.times_DET + self.times_GAN + self.times_DISC)
        
        it_loader = iter(self.train_loader)

        # -------------------------------------------

        log_dic = self.get_dict()

        for name, model in self.model.items():
            model.train()
            
        pbar = tqdm(total=len_loader - 1,
                    desc=f"Epoch {epoch} - Train", position=0, leave=True, unit='it')

        i = 0
        while i < len_loader:
            if self.times_DET > 0:
                for t in range(self.times_DET):
                    inputs, targets, _ = next(it_loader)
                    if self.cuda:
                        inputs, targets = inputs.cuda(non_blocking = True), targets.cuda(non_blocking = True)
                    # random apply noise
                    targets[:, ::4, ...] = (targets[:, ::4, ...] - 0.05).abs() + (torch.randn_like(targets[:, ::4, ...]) * 0.05).clip(-0.05, 0.05)
                    
                    self.DET.requires_grad_(True)
                    y,_ = self.DET(inputs)
                    self.analyse(y, targets)
                    ana_dic = self.analyse.compute()
                    loss_det = self.loss_DET(y, targets)
                    log_dic['loss_DET'].add(loss_det)
                    loss_det.backward()
                    i += 1
                    pbar.update(1)
                    inp_img = rearrange(inputs, "B C Z X Y -> B C X (Z Y)")
                    inp_img = make_grid(inp_img, nrow = 1)
                    self.tb_writer.add_image(f"TRAIN/Input Image", inp_img, global_step=(epoch-1) * len_loader + i)
                    out_img = rearrange(y, "B (C E) Z X Y -> C B E X (Z Y)", C = 4)
                    out_img = make_grid(out_img[0,:,(0,),...], nrow = 1)
                    self.tb_writer.add_image(f"TRAIN/Output Image", out_img, global_step=(epoch-1) * len_loader + i)
                    tar_img = rearrange(targets, "B (C E) Z X Y -> C B E X (Z Y)", C = 4)
                    tar_img = make_grid(tar_img[0,:,(0,),...], nrow = 1)
                    self.tb_writer.add_image(f"TRAIN/Target Image", tar_img, global_step=(epoch-1) * len_loader + i)
                
                grad = nn.utils.clip_grad_norm_(self.DET.parameters(), self.cfg.setting.clip_grad, error_if_nonfinite=True)
                log_dic['grad_DET'].add(grad)
                pbar.set_postfix({key:f"{log_dic[key]:.4f}" for key in log_dic if log_dic[key].n != 0})
                self.OPT_DET.step()
                self.OPT_DET.zero_grad()
                self.DET.requires_grad_(False)
            
            elif self.times_DISC > 0 and self.times_GAN > 0:
                for t in range(self.times_DISC):
                    inputs, targets, _ = next(it_loader)
                    if self.cuda:
                        inputs, targets = inputs.cuda(non_blocking = True), targets.cuda(non_blocking = True)
                    # random apply noise
                    targets[:, ::4, ...] = (targets[:, ::4, ...] - 0.05).abs() + (torch.randn_like(targets[:, ::4, ...]) * 0.05).clip(-0.05, 0.05)
                    
                    self.DISC.requires_grad_(True)
                    x, f = self.DET(inputs)
                    mask = x[:, (0,0,0), ...] < self.cfg.model.threshold
                    x[:, 1:, ...][mask] = 0
                    
                    yh = self.GAN(f)
                    mask = yh[:, (0,0,0), ...] < self.cfg.model.threshold
                    yh[:, 1:, ...][mask] = 0
                    
                    y = torch.cat([targets[:,:4,...],yh], dim = 1)
                    yp = self.DISC(y)
                    yt = self.DISC(targets)
                    loss_disc = self.loss_DISC(yp, yt)
                    loss_gp = self.loss_GP(y, yp, yt)
                    log_dic['loss_DISC'].add(loss_disc)
                    log_dic['loss_GP'].add(loss_gp)
                    loss = loss_disc + loss_gp
                    loss.backward()
                    i += 1
                    pbar.update(1)
                
                grad = nn.utils.clip_grad_norm_(self.DISC.parameters(), 999, error_if_nonfinite=True)
                log_dic['grad_DISC'].add(grad)
                pbar.set_postfix({key:f"{log_dic[key]:.4f}" for key in log_dic if log_dic[key].n != 0})
                self.OPT_DISC.step()
                self.OPT_DISC.zero_grad()
                self.DISC.requires_grad_(False)

                for t in range(self.times_GAN):
                    self.GAN.requires_grad_(True)
                    if self.cuda:
                        inputs, targets = inputs.cuda(non_blocking = True), targets.cuda(non_blocking = True)
                    x, f = self.DET(inputs)
                    mask = x[:, (0,0,0), ...] < self.cfg.model.threshold
                    x[:, 1:, ...][mask] = 0
                    
                    yh = self.GAN(f)
                    mask = yh[:, (0,0,0), ...] < self.cfg.model.threshold
                    yh[:, 1:, ...][mask] = 0
                    
                    y = torch.cat([x,yh], dim = 1)
                    yp = self.DISC(y)
                    yt = self.DISC(targets)
                    self.analyse(x, targets)
                    ana_dic = self.analyse.compute()
                    loss_gan = self.loss_DISC(yp, yt)
                    log_dic['loss_GAN'].add(loss_gan)
                    loss_gan.backward()
                    i += 1
                    pbar.update(1)
                    inp_img = rearrange(inputs, "B C Z X Y -> B C X (Z Y)")
                    inp_img = make_grid(inp_img, nrow = 1)
                    self.tb_writer.add_image(f"TRAIN/Input Image", inp_img, global_step=(epoch-1) * len_loader + i)
                    out_img = rearrange(y, "B (C E) Z X Y -> C B E X (Z Y)", C = 4)
                    out_img = make_grid(out_img[0,:,(0,0,1),...], nrow = 1)
                    self.tb_writer.add_image(f"TRAIN/Output Image", out_img, global_step=(epoch-1) * len_loader + i)
                    tar_img = rearrange(targets, "B (C E) Z X Y -> C B E X (Z Y)", C = 4)
                    tar_img = make_grid(tar_img[0,:,(0,0,1),...], nrow = 1)
                    self.tb_writer.add_image(f"TRAIN/Target Image", tar_img, global_step=(epoch-1) * len_loader + i)
                    
                    
                grad = nn.utils.clip_grad_norm_(self.GAN.parameters(), 999, error_if_nonfinite=True)
                log_dic['grad_GAN'].add(grad)
                pbar.set_postfix({key:f"{log_dic[key]:.4f}" for key in log_dic if log_dic[key].n != 0})
                self.OPT_GAN.step()
                self.OPT_GAN.zero_grad()
                self.GAN.requires_grad_(False)

            for key in log_dic.keys():
                if key != 'analyse':
                    self.tb_writer.add_scalar(
                        f"TRAIN/{key}", log_dic[key](), (epoch-1) * len_loader + i)

        # -------------------------------------------
        pbar.update(1)
        pbar.close()
        log_dic['analyse'] = ana_dic

        return log_dic

    @torch.no_grad()
    def valid(self, epoch):
        # -------------------------------------------

        log_dic = self.get_dict()
        len_loader = len(self.valid_loader)
        it_loader = iter(self.valid_loader)
        
        for name, model in self.model.items():
            model.eval()

        pbar = tqdm(total=len_loader - 1,
                    desc=f"Epoch {epoch} -  Valid", position=0, leave=True, unit='it')
        
        i = 0
        while i < len_loader:
            if self.times_DET > 0:
                inputs, targets, _ = next(it_loader)
                if self.cuda:
                    inputs, targets = inputs.cuda(non_blocking = True), targets.cuda(non_blocking = True)
                targets[:, ::4, ...] = (targets[:, ::4, ...] - 0.05).abs() + (torch.randn_like(targets[:, ::4, ...]) * 0.05).clip(-0.05, 0.05)
                
                y,_ = self.DET(inputs)
                self.analyse(y, targets)
                ana_dic = self.analyse.compute()
                loss_det = self.loss_DET(y, targets)
                log_dic['loss_DET'].add(loss_det)
                
                if i == 0:
                    inp_img = rearrange(inputs, "B C Z X Y -> B C X (Z Y)")
                    inp_img = make_grid(inp_img, nrow = 1)
                    self.tb_writer.add_image(f"VALID/Input Image", inp_img, global_step=(epoch-1) * len_loader + i)
                    out_img = rearrange(y, "B (C E) Z X Y -> C B E X (Z Y)", C = 4)
                    out_img = make_grid(out_img[0,:,(0,),...], nrow = 1)
                    self.tb_writer.add_image(f"VALID/Output Image", out_img, global_step=(epoch-1) * len_loader + i)
                    tar_img = rearrange(targets, "B (C E) Z X Y -> C B E X (Z Y)", C = 4)
                    tar_img = make_grid(tar_img[0,:,(0,),...], nrow = 1)
                    self.tb_writer.add_image(f"VALID/Target Image", tar_img, global_step=(epoch-1) * len_loader + i)
                     
            elif self.times_DISC > 0 and self.times_GAN > 0:
                if self.cuda:
                    inputs, targets = inputs.cuda(non_blocking = True), targets.cuda(non_blocking = True)
                x, f = self.DET(inputs)
                y = torch.cat([x,self.GAN(f)], dim = 1)
                yp = self.DISC(y)
                yt = self.DISC(targets)
                self.analyse(x, targets)
                ana_dic = self.analyse.compute()
                loss_gan = self.loss_GAN(yp, yt)
                loss_disc = self.loss_DISC(yp, yt)
                log_dic['loss_GAN'].add(loss_gan)
                log_dic['loss_DISC'].add(loss_disc)
                i += 1
                pbar.update(1)
                
                if i == 0:
                    inp_img = rearrange(inputs, "B C Z X Y -> B C X (Z Y)")
                    inp_img = make_grid(inp_img, nrow = 1)
                    self.tb_writer.add_image(f"VALID/Input Image", inp_img, global_step=(epoch-1) * len_loader + i)
                    out_img = rearrange(y, "B (C E) Z X Y -> C B E X (Z Y)", C = 4)
                    out_img = make_grid(out_img[0,:,(0,0,1),...], nrow = 1)
                    self.tb_writer.add_image(f"VALID/Output Image", out_img, global_step=(epoch-1) * len_loader + i)
                    tar_img = rearrange(targets, "B (C E) Z X Y -> C B E X (Z Y)", C = 4)
                    tar_img = make_grid(tar_img[0,:,(0,0,1),...], nrow = 1)
                    self.tb_writer.add_image(f"VALID/Target Image", tar_img, global_step=(epoch-1) * len_loader + i)
                
            
            i += 1
            pbar.update(1)
            pbar.set_postfix({key:f"{log_dic[key]:.4f}" for key in log_dic if log_dic[key].n != 0})

            log_dic['FID'].add(self.FID(y, targets))
            
            for key in log_dic.keys():
                    if key != 'analyse':
                        self.tb_writer.add_scalar(
                            f"VALID/{key}", log_dic[key](), (epoch-1) * len_loader + i)

        # -------------------------------------------
        pbar.update(1)
        pbar.close()
        log_dic['analyse'] = ana_dic
        
        return log_dic

    def save(self, epoch, log_dic):
        met = 0
        if log_dic["loss_DISC"].n > 0:
            met += math.log2(log_dic["loss_DISC"]())
        elif log_dic["FID"].n > 0:
            met += math.log2(log_dic["FID"]())
            
        logger = self.logger
        
        if met < self.best_met:
            self.best_met = met

            log = []
            try:
                name = f"DET.CP{epoch:02d}_LOSS{log_dic['loss_DET']:.4f}_FID{log_dic['FID']}.pkl"
                self.DET.save(f"{self.work_dir}/{name}")
                log.append(f"Saved a new DET: {name}")
            except AttributeError:
                pass
            
            try:
                name = f"DISC.CP{epoch:02d}_LOSS{log_dic['loss_DISC']:.4f}_FID{log_dic['FID']}.pkl"
                self.DISC.save(f"{self.work_dir}/{name}")
                log.append(f"Saved a new DISC: {name}")
            except AttributeError:
                pass
            
            try:
                name = f"GAN.CP{epoch:02d}_LOSS{log_dic['loss_GAN']:.4f}_FID{log_dic['FID']}.pkl"
                self.GAN.save(f"{self.work_dir}/{name}")
                log.append(f"Saved a new GAN: {name}")
            except AttributeError:
                pass
            
            for i in log:
                logger.info(i)
                
        else:
            logger.info(f"No model was saved")
