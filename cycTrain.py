from datasets import (PixelShift, Blur, ColorJitter, CutOut, Noisy)
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from torchvision.utils import make_grid

from model import build_cyc_model
from model.utils import save_model, load_model
from datasets import AFMDataset

from utils import *
from utils.metrics import metStat
from utils.schedular import Scheduler

import tqdm
import time
import os
from itertools import chain
from copy import deepcopy

user = os.environ.get("USER") == "supercgor"
if user:
    from config.config import get_config
else:
    from config.wm import get_config


class Trainer():
    def __init__(self):
        cfg = get_config()
        self.cfg = cfg

        devices = list(range(torch.cuda.device_count()))
        batch_size = len(devices)

        self.work_dir = f"{cfg.path.check_root}/CycleTrain_{time.strftime('%Y%m%d-%H%M%S', time.localtime())}"
        os.makedirs(self.work_dir, exist_ok=True)

        self.tb_logger = SummaryWriter(
            log_dir=f"{self.work_dir}/runs/cyctrain")

        self.logger = Logger(path=f"{self.work_dir}",
                             log_name="tune.log",
                             elem=cfg.data.elem_name,
                             split=cfg.setting.split)

        self.logger.info(f"devices: {devices}")

        self.G_S2T, self.D_S = build_cyc_model(disc=True)

        self.logger.info(f"Generator parameters: {sum(p.numel() for p in self.G_S2T.parameters() if p.requires_grad)}")
        self.logger.info(f"Discriminator parameters: {sum(p.numel() for p in self.D_S.parameters() if p.requires_grad)}")
        
        self.G_S2T.cuda()
        self.D_S.cuda()

        self.G_T2S, self.D_T = deepcopy(self.G_S2T), deepcopy(self.D_S)

        if len(devices) >= 2:
            self.G_S2T = nn.DataParallel(self.G_S2T, device_ids=devices)
            self.G_T2S = nn.DataParallel(self.G_T2S, device_ids=devices)
            self.D_S = nn.DataParallel(self.D_S, device_ids=devices)
            self.D_T = nn.DataParallel(self.D_T, device_ids=devices)

        self.models = {"G_S2T": self.G_S2T, "G_T2S": self.G_T2S,
                       "D_S": self.D_S, "D_T": self.D_T}

        self.G_opt = torch.optim.Adam(
            self.G_paras, lr=2e-4, betas=(0.5, 0.999))
        self.D_opt = torch.optim.SGD(self.D_paras, lr=2e-4)

        self.G_scheduler = Scheduler(
            self.G_opt, warmup_steps=50, decay_factor=4000)
        self.D_scheduler = Scheduler(
            self.D_opt, warmup_steps=50, decay_factor=4000)

        trans_S = Compose([PixelShift(fill=None), CutOut(), ColorJitter(), Noisy(), Blur()])

        source_data = AFMDataset(f"{self.cfg.path.data_root}/bulkiceMix",
                                 self.cfg.data.elem_name,
                                 file_list="train.filelist",
                                 transform=trans_S,
                                 img_use=self.cfg.data.img_use,
                                 model_inp=self.cfg.model.inp_size,
                                 model_out=self.cfg.model.out_size,
                                 label=False)

        self.source_loader = DataLoader(source_data, batch_size=batch_size, num_workers=6,
                                        pin_memory=self.cfg.setting.pin_memory, shuffle=True, drop_last=True)

        trans_T = Compose([Blur(sigma = 0.5), ColorJitter()])

        target_data = AFMDataset(f"{self.cfg.path.data_root}/bulkexpR",
                                 self.cfg.data.elem_name,
                                 file_list="allbetter.filelist",
                                 transform=trans_T,
                                 img_use=self.cfg.data.img_use,
                                 model_inp=self.cfg.model.inp_size,
                                 model_out=self.cfg.model.out_size,
                                 label=False)

        self.target_loader = DataLoader(target_data, batch_size=batch_size, num_workers=6,
                                        pin_memory=self.cfg.setting.pin_memory, shuffle=True, drop_last=True)

    @property
    def G_paras(self):
        return chain(self.G_S2T.parameters(), self.G_T2S.parameters())

    @property
    def D_paras(self):
        return chain(self.D_S.parameters(), self.D_T.parameters())

    def fit(self):
        self.best = {'loss': metStat(mode="min")}
        self.best_met = 9999

        self.logger.info(f"Start training cycleGAN")

        for epoch in range(1, 200 + 1):
            epoch_start_time = time.time()

            log_train_dic = self.train(epoch)

            self.logger.info("")

            self.save(epoch, log_train_dic)

            log = [f"Epoch {epoch} - Cycle Train"]
            log += [f"Used Time: {time.time() - epoch_start_time:.2f} s"]
            log += [f"Used Memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.0f} MB / {torch.cuda.memory_reserved() / 1024 ** 2:.0f} MB"]
            log += [f"Loss: {log_train_dic['G_S2T']:.4f} / {log_train_dic['G_T2S']:.4f} / {log_train_dic['D_S']:.4f} / {log_train_dic['D_T']:.4f}"]
            log += [f"Grad: {log_train_dic['G']:.4f} / {log_train_dic['D']:.4f}"]
            log += [f"LR rate: {self.G_opt.param_groups[0]['lr']:.6f}"]
            log += [f'End training epoch: {epoch:0d}']
            for i in log:
                self.logger.info(i)

    def train_passdata(self, source_real, target_real):
        self.source_real = source_real.cuda()
        self.target_real = target_real.cuda()
        self.source_fake = self.G_T2S(self.target_real).sigmoid()
        self.target_fake = self.G_S2T(self.source_real).sigmoid()
        self.source_idt = self.G_T2S(self.source_real).sigmoid()
        self.target_idt = self.G_S2T(self.target_real).sigmoid()
        self.source_cyc = self.G_T2S(self.target_fake).sigmoid()
        self.target_cyc = self.G_S2T(self.source_fake).sigmoid()

    def train_disc(self):
        for name, model in self.models.items():
            model.requires_grad_(
                True) if "D" in name else model.requires_grad_(False)
            
        with torch.no_grad():
            source_fake = self.G_T2S(self.target_real).sigmoid()
            target_fake = self.G_S2T(self.source_real).sigmoid()
            
        pred_SR = self.D_S(self.source_real).sigmoid()
        pred_SF = self.D_S(source_fake).sigmoid()
        pred_TR = self.D_T(self.target_real).sigmoid()
        pred_TF = self.D_T(target_fake).sigmoid()

        loss_DS = (F.mse_loss(pred_SR, torch.ones_like(pred_SR)) +
                   F.mse_loss(pred_SF, torch.zeros_like(pred_SF))) / 2
        loss_DT = (F.mse_loss(pred_TR, torch.ones_like(pred_TR)) +
                   F.mse_loss(pred_TF, torch.zeros_like(pred_TF))) / 2

        return loss_DS, loss_DT

    def train_gen(self):
        for name, model in self.models.items():
            model.requires_grad_(True) if "G" in name else model.requires_grad_(False)
        
        with torch.no_grad():
            pred_SF = self.D_S(self.source_fake).sigmoid()
            pred_TF = self.D_T(self.target_fake).sigmoid()

        loss_GS2T_cls = F.mse_loss(pred_SF, torch.ones_like(pred_SF)) * 2
        loss_GS2T_idt = F.l1_loss(self.source_idt, self.source_real) * 1
        loss_GS2T_cyc = F.l1_loss(self.source_cyc, self.source_real) * 10

        loss_GT2S_cls = F.mse_loss(pred_TF, torch.ones_like(pred_TF)) * 2
        loss_GT2S_idt = F.l1_loss(self.target_idt, self.target_real) * 1
        loss_GT2S_cyc = F.l1_loss(self.target_cyc, self.target_real) * 10

        loss_GS2T = (loss_GS2T_cls + loss_GS2T_idt + loss_GS2T_cyc) / 13
        loss_GT2S = (loss_GT2S_cls + loss_GT2S_idt + loss_GT2S_cyc) / 13

        loss_item = {"G_S2T_cls": loss_GS2T_cls.item(), 
                     "G_S2T_idt": loss_GS2T_idt.item(), 
                     "G_S2T_cyc": loss_GS2T_cyc.item(), 
                     "G_T2S_cls": loss_GT2S_cls.item(), 
                     "G_T2S_idt": loss_GT2S_idt.item(), 
                     "G_T2S_cyc": loss_GT2S_cyc.item()}

        return loss_GS2T, loss_GT2S, loss_item

    def train(self, epoch):

        A_iter = iter(self.source_loader)
        B_iter = iter(self.target_loader)
        max_iter = min(len(A_iter), len(B_iter), 1001)

        Loss = {"G_S2T": metStat(), "G_T2S": metStat(),
                "D_S": metStat(), "D_T": metStat()}
        Grad = {"G": metStat(), "D": metStat()}

        pbar = tqdm.tqdm(
            total=max_iter - 1, desc=f"Epoch {epoch} - Cycle Train", position=0, leave=True, unit='it')

        i = 0
        while i < max_iter:
            step = (epoch - 1) * max_iter + i
            self.train_passdata(next(A_iter)[0], next(B_iter)[0])
            # Train Discriminator
            self.D_opt.zero_grad()
            loss_DS, loss_DT = self.train_disc()
            (loss_DS + loss_DT).backward()
            GD = nn.utils.clip_grad_norm_(
                self.D_paras, 10, error_if_nonfinite=True)
            self.D_opt.step()

            Grad["D"].add(GD)
            Loss["D_S"].add(loss_DS)
            Loss["D_T"].add(loss_DT)

            # Train Generator
            self.G_opt.zero_grad()
            loss_GS2T, loss_GT2S, loss_dic = self.train_gen()
            (loss_GS2T + loss_GT2S).backward()
            GG = nn.utils.clip_grad_norm_(
                self.G_paras, 10, error_if_nonfinite=True)
            self.G_opt.step()

            Loss["G_S2T"].add(loss_GS2T)
            Loss["G_T2S"].add(loss_GT2S)
            Grad["G"].add(GG)

            self.G_scheduler.step()
            self.D_scheduler.step()

            pbar.set_postfix(**{key: value.last for key, value in Loss.items()},
                             **{key: value.last for key, value in Grad.items()})
            pbar.update(1)

            self.tb_logger.add_scalar(
                "TRAIN/LR_rate", self.G_opt.param_groups[0]['lr'], step)

            self.tb_logger.add_scalars("TRAIN/Loss item", loss_dic, step)

            self.tb_logger.add_scalars(
                "TRAIN/Loss", {key: value.last for key, value in Loss.items()}, step)
            self.tb_logger.add_scalars(
                "TRAIN/Grad", {key: value.last for key, value in Grad.items()}, step)
            if i % 100 == 0:
                imgs = torch.cat([self.source_real[0, :, 0::2],
                                  self.target_fake[0, :, 0::2],
                                  self.source_fake[0, :, 0::2],
                                  self.target_real[0, :, 0::2]], dim=1)
                self.tb_logger.add_image("GAN Image (RealA & FakeB - FakeA & RealB)", make_grid(
                    imgs.permute(1, 0, 2, 3).cpu(), nrow=16), step)
                del imgs
            i += 1
            print(torch.cuda.max_memory_allocated() / 1024 / 1024)

        pbar.update()
        pbar.close()
        return {key: value() for key, value in chain(Loss.items(), Grad.items())}

    def save(self, epoch, log_dic):
        met = 0
        met += log_dic["G_S2T"]
        met += log_dic["G_T2S"]

        logger = self.logger

        if True:
            self.best_met = met

            log = []
            subname = f"CP{epoch:02d}_LOSS{met:.4f}.pkl"
            for name, model in self.models.items():
                save_model(model, f"{self.work_dir}/{name}_{subname}")
            log.append(f"Saved a new net: {subname}")

        else:
            logger.info(f"No model was saved")

        for i in log:
            logger.info(i)


if __name__ == "__main__":
    Trainer().fit()
