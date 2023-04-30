import time
from tqdm import tqdm
from collections import OrderedDict
import math


import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import pytorch_warmup as warmup
import model
from model.utils import basicParallel

from datasets.dataset import make_dataset
from utils.logger import Logger
from utils.loader import Loader
from utils.metrics import metStat, analyse_cls
from utils.criterion import modelLoss
from utils.schedular import Scheduler
from demo.plot import out2img


class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg

        assert cfg.setting.device != [], "No device is specified!"

        self.load_dir, self.work_dir = Loader(cfg, make_dir=True)

        self.logger = Logger(path=self.work_dir,
                             elem=cfg.data.elem_name, split=cfg.setting.split)

        self.tb_writer = SummaryWriter(
            log_dir=f"{self.work_dir}/runs/{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

        log = []

        # load the feature extractor network
        self.net = model.CombineModel().cuda()
    
        try:
            paths = {"fea": f"{self.load_dir}/{cfg.model.fea}",
                    "neck": f"{self.load_dir}/{cfg.model.neck}", 
                    "head": f"{self.load_dir}/{cfg.model.head}"}
            match_list = self.net.load(paths, pretrained=True)
            match_list = "\n".join(match_list)
            log.append(f"Load parameters from {self.load_dir}")
            log.append(f"\n{match_list}")
        except FileNotFoundError:
            self.net.init()
            log.append(
                f"No network is loaded, start a new model: {self.net.name}")

        if len(cfg.setting.device) >= 2:
            self.net.parallel(devices_ids=cfg.setting.device)

        self.analyse = analyse_cls(threshold=cfg.model.threshold).cuda()
        self.LOSS = modelLoss(threshold = cfg.model.threshold, pos_w = cfg.criterion.pos_weight).cuda()
        self.OPT = torch.optim.AdamW(self.net.parameters(), lr=self.cfg.setting.lr, weight_decay= 1e-4)
        self.SCHEDULER = Scheduler(self.OPT, warmup_steps=2000, decay_factor=10000)
        for l in log:
            self.logger.info(l)

    def fit(self):
        # --------------------------------------------------

        self.train_loader = make_dataset('train', self.cfg)

        self.valid_loader = make_dataset('valid', self.cfg)

        self.best = {'loss': metStat(mode="min")}
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

            for dic_name, dic in zip(["Train", "Valid"], [log_train_dic, log_valid_dic]):
                for key, MET in dic.items():
                    if key != 'MET':
                        self.tb_writer.add_scalar(
                            f"EPOCH/{dic_name} {key}", MET.value, epoch)
                    else:
                        # log Metrics dict
                        for e in MET.elems:
                            for met_name in MET.met:
                                self.tb_writer.add_scalars(
                                        f"EPOCH/{dic_name} {e} {met_name}", {l: MET[e, i, met_name] for i,l in enumerate(MET.split)}, epoch)
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
        
        len_loader = len(self.train_loader)
        len_loader = len_loader - len_loader % accu
        it_loader = iter(self.train_loader)

        self.net.train()

        # ----------------------------- --------------

        T_dict = self.get_dict()

        pbar = tqdm(total=len_loader - 1,
                    desc=f"Epoch {epoch} - Train", position=0, leave=True, unit='it')

        i = 0
        while i < len_loader:
            step = (epoch-1) * len_loader + i
            for t in range(accu):
                # imgs : (B, C, D, H, W), gt_bbox : (B, N, 4), gt_clss : (B, N, 1), filenames : (B)
                imgs, gt_box, filenames = next(it_loader)
                
                imgs = imgs.cuda(non_blocking=True)
                gt_box = gt_box.cuda(non_blocking=True)

                pd_box = self.net(imgs)
                
                loss = self.LOSS(pd_box, gt_box)
                loss.backward()

                match = self.analyse(pd_box, gt_box)
                T_dict['Loss'].add(loss)
                
                i += 1
                pbar.update(1)

            grad = nn.utils.clip_grad_norm_(
                self.net.parameters(), 10, error_if_nonfinite=True)

            T_dict['Grad'].add(grad)

            pbar.set_postfix(Loss= T_dict['Loss'].last,
                             Grad= T_dict['Grad'].last)

            self.OPT.step()
            self.SCHEDULER.step()

            # log Train dict
            if step % 100 == 0:
                self.tb_writer.add_images("Train/In IMG", imgs[0].permute(1,0,2,3), step)
                self.tb_writer.add_image("Train/OUT BOX", out2img(pd_box, gt_box), step)
            
            self.tb_writer.add_scalar(
                f"TRAIN/LR_rate", self.OPT.param_groups[0]['lr'], step)

            for key, value in T_dict.items():
                self.tb_writer.add_scalar(f"TRAIN/{key}", value.last, step)

            
            # log Metrics dict
            for e in ["O","H"]:
                for key in ["AP", "AR"]:
                    self.tb_writer.add_scalars(f"TRAIN/{e} {key}", {l: match[e,i,key] for i,l in enumerate(match.split)}, step)

        # -------------------------------------------
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

        T_dict = self.get_dict()

        pbar = tqdm(total=len_loader - 1,
                    desc=f"Epoch {epoch} -  Valid", position=0, leave=True, unit='it')

        i = 0
        while i < len_loader:
            step = (epoch-1) * len_loader + i
            imgs, gt_box, filenames = next(it_loader)

            imgs = imgs.cuda(non_blocking=True)
            gt_box = gt_box.cuda(non_blocking=True)
            pd_box = self.net(imgs)

            loss = self.LOSS(pd_box, gt_box)
                        
            match = self.analyse(pd_box, gt_box)
            T_dict['Loss'].add(loss)

            pbar.set_postfix(Loss= T_dict['Loss'].last)
            
            i += 1
            pbar.update(1)

            # log Valid dict
            self.tb_writer.add_scalar(f"VALID/Loss", T_dict['Loss'].last, step)
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
            try:
                name = f"CP{epoch:02d}_LOSS{log_dic['Loss']:.4f}.pkl"
                self.net.save(path = self.work_dir, name = name)
                log.append(f"Saved a new net: {name}")
            except AttributeError:
                pass

            for i in log:
                logger.info(i)

        else:
            logger.info(f"No model was saved")
