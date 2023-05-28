import time
import os
import tqdm
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import UNetModel, Regression
from model.utils import basicParallel
from itertools import chain

from datasets import AFMDataset
from utils import *
from utils.metrics import metStat, analyse_cls
from utils.criterion import modelLoss
from utils.schedular import Scheduler

if os.environ.get("USER") == "supercgor":
    from config.config import get_config
else:
    from config.wm import get_config

class Trainer():
    def __init__(self):
        cfg = get_config()
        self.cfg = cfg

        assert cfg.setting.device != [], "No device is specified!"
        
        self.work_dir = f"{cfg.path.check_root}/Train_{time.strftime('%Y%m%d-%H%M%S', time.localtime())}"
        os.makedirs(self.work_dir, exist_ok=True)
        
        self.logger = Logger(path=self.work_dir,
                             elem=cfg.data.elem_name, 
                             split=cfg.setting.split)

        self.tb_writer = SummaryWriter(log_dir=f"{self.work_dir}/runs/train")

        # load the feature extractor network
        self.net = UNetModel(image_size=(16, 128, 128),
                             in_channels=1,
                             model_channels=32,
                             out_channels=32,
                             num_res_blocks=2,
                             attention_resolutions=(8,),
                             dropout=0.0,
                             channel_mult=(1, 2, 4, 4),
                             dims=3,
                             num_heads = 4,
                             time_embed=None,
                             use_checkpoint=False).cuda()
        
        self.logger.info(f"UNet parameters: {sum([p.numel() for p in self.net.parameters()])}")
        
        # load the regression network
        self.reg = Regression(in_channels=32, out_channels=8).cuda()
        self.logger.info(f"Reg parameters: {sum([p.numel() for p in self.reg.parameters()])}")

        self.cyc = UNetModel(image_size = (16, 128, 128), 
              in_channels = 1, 
              model_channels = 32,
              out_channels = 1,
              num_res_blocks = 1,
              attention_resolutions = [], 
              dropout = 0.1,
              channel_mult = (1,2,2,4), 
              dims = 3, 
              time_embed= None,
              use_checkpoint=False).cuda()
        
        self.logger.info(f"CycUNet parameters: {sum([p.numel() for p in self.cyc.parameters()])}")

        log = []
        
        try:
            log.extend(self.net.load(f"{cfg.path.check_root}/{cfg.model.checkpoint}/{cfg.model.fea}", pretrained=True))
            log.append(f"Load parameters from {cfg.model.checkpoint}/{cfg.model.fea}")
        except (FileNotFoundError, IsADirectoryError):
            raise FileNotFoundError(f"No feature extractor network is loaded at '{cfg.path.check_root}/{cfg.model.checkpoint}/{cfg.model.fea}'")

            
        try:
            log.extend(self.reg.load(f"{cfg.path.check_root}/{cfg.model.checkpoint}/{self.cfg.model.reg}", pretrained=True))
            log.append(f"Load parameters from {cfg.model.checkpoint}/{cfg.model.reg}")
        except (FileNotFoundError, IsADirectoryError):
            raise FileNotFoundError(f"No regression network is loaded at '{cfg.path.check_root}/{cfg.model.checkpoint}/{cfg.model.reg}'")
        
        try:
            log.extend(self.cyc.load(f"{cfg.path.check_root}/{cfg.model.checkpoint}/{self.cfg.model.cyc}", pretrained=True))
            log.append(f"Load parameters from {cfg.model.checkpoint}/{cfg.model.cyc}")
        except (FileNotFoundError, IsADirectoryError):
            raise FileNotFoundError(f"No cycle network is loaded at '{cfg.path.check_root}/{cfg.model.checkpoint}/{cfg.model.cyc}'")
        
        if len(cfg.setting.device) >= 2:
            self.net = basicParallel(self.net, device_ids = cfg.setting.device)
            self.reg = basicParallel(self.reg, device_ids = cfg.setting.device)
            self.cyc = basicParallel(self.cyc, device_ids = cfg.setting.device)

        self.net.train()
        self.reg.train()
        self.cyc.eval()
        self.cyc.requires_grad_(False)

        self.analyse = analyse_cls(threshold=cfg.model.threshold).cuda()
        
        self.LOSS = modelLoss(pos_w=cfg.criterion.pos_weight).cuda()
        
        self.OPT = torch.optim.AdamW(
            chain(self.net.parameters(), self.reg.parameters()), 
            lr=self.cfg.setting.lr, 
            weight_decay=5e-3)
        
        self.SCHEDULER = Scheduler(self.OPT, warmup_steps=5, decay_factor=50000)
        
        for l in log:
            self.logger.info(l)

    def fit(self):
        # --------------------------------------------------

        train_data = AFMDataset(f"{self.cfg.path.data_root}/{self.cfg.data.dataset}",
                                self.cfg.data.elem_name,
                                file_list= "train.filelist",
                                img_use=self.cfg.data.img_use,
                                model_inp=self.cfg.model.inp_size,
                                model_out=self.cfg.model.out_size)

        self.train_loader = DataLoader(train_data,
                                       batch_size=self.cfg.setting.batch_size,
                                       num_workers=self.cfg.setting.num_workers,
                                       pin_memory=self.cfg.setting.pin_memory,
                                       shuffle=True)

        valid_data = AFMDataset(f"{self.cfg.path.data_root}/{self.cfg.data.dataset}",
                                self.cfg.data.elem_name,
                                file_list= "valid.filelist",
                                img_use=self.cfg.data.img_use,
                                model_inp=self.cfg.model.inp_size,
                                model_out=self.cfg.model.out_size)
        
        self.valid_loader = DataLoader(valid_data,
                                       batch_size=self.cfg.setting.batch_size,
                                       num_workers=self.cfg.setting.num_workers,
                                       pin_memory=self.cfg.setting.pin_memory)

        self.best = {'loss': metStat(mode="min")}
        self.best_met = 9999

        # --------------------------------------------------
        self.logger.info(f'Start training.')

        for epoch in range(1, self.cfg.setting.epochs + 1):
            epoch_start_time = time.time()
            
            log_train_dic = self.train(epoch)

            log_valid_dic = self.valid(epoch)

            self.logger.epoch_info(epoch, log_train_dic, log_valid_dic)

            for dic_name, dic in (("Train", log_train_dic), ("Valid", log_valid_dic)):
                for key, MET in dic.items():
                    if key != 'MET':
                        self.tb_writer.add_scalar(
                            f"EPOCH/{dic_name} {key}", MET.value, epoch)
                    else:
                        # log Metrics dict
                        for e in MET.elems:
                            self.tb_writer.add_scalars(
                                f"EPOCH/{dic_name} {e} COUNT", {f"{met_name} {l}": MET[e, i, met_name] for i, l in enumerate(MET.split) for met_name in ["TP", "FP", "FN", "T", "P"]}, epoch)
                            self.tb_writer.add_scalars(
                                f"EPOCH/{dic_name} {e} AP/AR", {f"{met_name} {l}": MET[e, i, met_name] for i, l in enumerate(MET.split) for met_name in ["AP", "AR"]}, epoch)
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
        T_dict = OrderedDict(
            Grad=metStat(mode="mean"),
            Loss=metStat(mode="mean"))

        return T_dict

    def train(self, epoch):
        # -------------------------------------------

        accu = self.cfg.setting.batch_accumulation

        iter_times = len(self.train_loader) // accu
        
        it_loader = iter(self.train_loader)

        self.net.train()
        self.reg.train()

        # ----------------------------- --------------

        T_dict = self.get_dict()

        pbar = tqdm.tqdm(total=iter_times - 1, desc = f"Epoch {epoch} - Train", position=0, leave=True, unit='it')

        i = 0
        while i < iter_times:
            step = (epoch-1) * iter_times + i
            for t in range(accu):
                # imgs : (B, C, D, H, W), gt_box : (B, D, H, W, 2, 4), filenames : (B)
                imgs, gt_box, filenames = next(it_loader)

                imgs = imgs.cuda(non_blocking=True)
                imgs = random.choice([lambda x: x, self.cyc])(imgs)
                
                gt_box = gt_box.cuda(non_blocking=True)
                pd_box = self.reg(self.net(imgs))
                
                loss = self.LOSS(pd_box, gt_box)
                loss.backward()

                match = self.analyse(pd_box, gt_box)
                T_dict['Loss'].add(loss)

                i += 1
                pbar.update(1)

            grad = nn.utils.clip_grad_norm_(
                chain(self.net.parameters(),self.reg.parameters()), self.cfg.setting.clip_grad, error_if_nonfinite=True)

            T_dict['Grad'].add(grad)

            pbar.set_postfix(Loss=T_dict['Loss'].last,
                             Grad=T_dict['Grad'].last)

            self.OPT.step()
            self.SCHEDULER.step()
            self.OPT.zero_grad()

            # log Train dict
            if step % 100 == 0:
                self.tb_writer.add_images(
                    "Train/In IMG", imgs[0].permute(1, 0, 2, 3), step)
                
                self.tb_writer.add_image(
                    "Train/OUT BOX", torch.stack([label2img(pd_box, format = "BZXYEC")] * 2 + [label2img(gt_box, format = "BZXYEC")], dim = 1), step)

            self.tb_writer.add_scalar(
                f"TRAIN/LR_rate", self.OPT.param_groups[0]['lr'], step)

            self.tb_writer.add_scalars(
                f"TRAIN", {key: value.last for key, value in T_dict.items()}, step)

            # log Metrics dict
            for e in ["O", "H"]:
                for j, l in enumerate(match.split):
                    self.tb_writer.add_scalars(
                        f"TRAIN/{e} {l}", {key: match[e, j, key] for key in ["AP", "AR"]}, step)

        # --------------------------------  -----------
        pbar.update(1)
        pbar.close()
        return {**T_dict, "MET": self.analyse.summary()}

    @torch.no_grad()
    def valid(self, epoch):
        # -------------------------------------------

        T_dict = self.get_dict()
        len_loader = len(self.valid_loader)
        it_loader = iter(self.valid_loader)

        self.net.eval()
        self.reg.eval()
        
        T_dict = self.get_dict()

        pbar = tqdm.tqdm(total=len_loader - 1, desc=f"Epoch {epoch} -  Valid", position=0, leave=True, unit='it')

        i = 0
        while i < len_loader:
            step = (epoch-1) * len_loader + i
            imgs, gt_box, filenames = next(it_loader)

            imgs = imgs.cuda(non_blocking=True)
            gt_box = gt_box.cuda(non_blocking=True)
            pd_box = self.reg(self.net(imgs))

            loss = self.LOSS(pd_box, gt_box)

            match = self.analyse(pd_box, gt_box)
            T_dict['Loss'].add(loss)

            pbar.set_postfix(Loss=T_dict['Loss'].last)

            i += 1
            pbar.update(1)

            # log Valid dict
            self.tb_writer.add_scalars(
                f"VALID", {key: value.last for key, value in T_dict.items()}, step)
        # -------------------------------------------

        pbar.update(1)
        pbar.close()

        return {**T_dict, "MET": self.analyse.summary()}

    def save(self, epoch, log_dic):
        met = 0
        if log_dic["Loss"].n > 0:
            met += log_dic["Loss"]()

        logger = self.logger

        if met < self.best_met:
            self.best_met = met

            log = []
            name = f"CP{epoch:02d}_LOSS{log_dic['Loss']:.4f}.pkl"
            self.net.save(path=f"{self.work_dir}/unet_{name}")
            self.reg.save(path=f"{self.work_dir}/reg_{name}")
            log.append(f"Saved a new net: {name}")

            for i in log:
                logger.info(i)

        else:
            logger.info(f"No model was saved")

if __name__ == "__main__":
    Trainer().fit()