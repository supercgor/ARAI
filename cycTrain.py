from datasets import (PixelShift, Noisy, Blur, ColorJitter)
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

from model import UNetModel, NLayerDiscriminator
from model.utils import basicParallel
from datasets import AFMDataset

from utils import *
from utils.metrics import metStat
from utils.schedular import Scheduler

import tqdm
import time
import os
from itertools import chain

user = os.environ.get("USER") == "supercgor"
if user:
    from config.config import get_config
else:
    from config.wm import get_config


class Trainer():
    def __init__(self):
        cfg = get_config()
        self.cfg = cfg

        self.epochs = 50

        assert cfg.setting.device != [], "No device is specified!"

        self.work_dir = f"{cfg.path.check_root}/CycleTrain_{time.strftime('%Y%m%d-%H%M%S', time.localtime())}"
        os.makedirs(self.work_dir, exist_ok=True)

        self.tb_writer = SummaryWriter(
            log_dir=f"{self.work_dir}/runs/cyctrain")

        self.logger = Logger(path=f"{self.work_dir}",
                             log_name="tune.log",
                             elem=cfg.data.elem_name,
                             split=cfg.setting.split)

        self.A2B = UNetModel(image_size=(16, 128, 128),
                             in_channels=1,
                             model_channels=32,
                             out_channels=1,
                             num_res_blocks=1,
                             attention_resolutions=[],
                             dropout=0.1,
                             channel_mult=(1, 2, 2, 4),
                             dims=3,
                             time_embed=None,
                             use_checkpoint=False).init().cuda()

        self.B2A = UNetModel(image_size=(16, 128, 128),
                             in_channels=1,
                             model_channels=32,
                             out_channels=1,
                             num_res_blocks=1,
                             attention_resolutions=[],
                             dropout=0.1,
                             channel_mult=(1, 2, 2, 4),
                             dims=3,
                             time_embed=None,
                             use_checkpoint=False).init().cuda()

        self.A = NLayerDiscriminator(in_channels=1,
                                     model_channels=32,
                                     channel_mult=(1, 2, 4, 8),
                                     max_down_mult=(4, 8, 8),
                                     reduce="mean").init().cuda()

        self.B = NLayerDiscriminator(in_channels=1,
                                     model_channels=32,
                                     channel_mult=(1, 2, 4, 8),
                                     max_down_mult=(4, 8, 8),
                                     reduce="mean").init().cuda()

        load = {"A2B": "",
                "B2A": "",
                "A": "",
                "B": ""}

        log = []
        for key, name in load.items():
            if len(cfg.setting.device) >= 2:
                setattr(self, key,
                        basicParallel(getattr(self, key), device_ids=cfg.setting.device))
            if name == "":
                continue
            try:
                getattr(self, key).load(name, pretrained=True)
                log.append(f"Loade model {key} from {name}")
            except (FileNotFoundError, IsADirectoryError):
                log.append(f"Model {key} loading warning, {name}")

        self.OPTG = torch.optim.Adam(chain(self.A2B.parameters(
        ), self.B2A.parameters()), lr=2e-4, betas=(0.5, 0.999))

        self.OPTD = torch.optim.Adam(
            chain(self.A.parameters(), self.B.parameters()), lr=2e-4, betas=(0.5, 0.999))

        self.SCHE_G = Scheduler(
            self.OPTG, warmup_steps=1000, decay_factor=50000)
        self.SCHE_D = Scheduler(
            self.OPTD, warmup_steps=1000, decay_factor=50000)

        A_trans = torchvision.transforms.Compose([
            torchvision.transforms.RandomApply([PixelShift(fill=None)], p=0.5),
            torchvision.transforms.RandomApply([Noisy()], p=0.3),
            torchvision.transforms.RandomApply([Blur()], p=0.3),
            torchvision.transforms.RandomApply([ColorJitter()], p=0.3),
        ])

        A_data = AFMDataset(f"../data/bulkice",
                            self.cfg.data.elem_name,
                            file_list="train.filelist",
                            transform=A_trans,
                            img_use=self.cfg.data.img_use,
                            model_inp=self.cfg.model.inp_size,
                            model_out=self.cfg.model.out_size,
                            label = False)

        self.A_loader = DataLoader(A_data,
                                   batch_size=self.cfg.setting.batch_size,
                                   num_workers=self.cfg.setting.num_workers,
                                   pin_memory=self.cfg.setting.pin_memory,
                                   shuffle=True,
                                   drop_last=True)

        B_trans = torchvision.transforms.Compose([
            Noisy(0.05),
            torchvision.transforms.RandomApply([Blur()], p=0.3),
            ColorJitter()])

        B_data = AFMDataset(f"../data/bulkexpR",
                            self.cfg.data.elem_name,
                            file_list="train.filelist",
                            transform=B_trans,
                            img_use=self.cfg.data.img_use,
                            model_inp=self.cfg.model.inp_size,
                            model_out=self.cfg.model.out_size,
                            label = False)

        self.B_loader = DataLoader(B_data,
                                   batch_size=self.cfg.setting.batch_size,
                                   num_workers=self.cfg.setting.num_workers,
                                   pin_memory=self.cfg.setting.pin_memory,
                                   drop_last=True)

    def fit(self):
        self.best = {'loss': metStat(mode="min")}
        self.best_met = 9999

        self.logger.info(f"Start training cycleGAN")

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()

            log_train_dic = self.train(epoch)

            self.logger.info("")

            self.save(epoch, log_train_dic)

            self.logger.info(
                f"Spend time: {time.time() - epoch_start_time:.2f}s")

            self.logger.info(
                f"Used memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB")

            self.logger.info(f'End training epoch: {epoch:0d}')

    def train(self, epoch):
        A_iter = iter(self.A_loader)
        B_iter = iter(self.B_loader)
        max_iter = min(len(A_iter), len(B_iter))

        Loss = {
            "A2B": metStat(),
            "B2A": metStat(),
            "A": metStat(),
            "B": metStat()
        }
        Grad = {
            "G": metStat(),
            "D": metStat()
        }

        pbar = tqdm.tqdm(
            total=max_iter - 1, desc=f"Epoch {epoch} - Cycle Train", position=0, leave=True, unit='it')

        i = 0
        while i < max_iter:
            step = (epoch - 1) * max_iter + i
            self.A.requires_grad_(True)
            self.B.requires_grad_(True)
            self.A2B.requires_grad_(False)
            self.B2A.requires_grad_(False)
            with torch.no_grad():
                real_A, _ = next(A_iter)
                real_B, _ = next(B_iter)
                real_A, real_B = real_A.cuda(), real_B.cuda()
                fake_B, fake_A = self.A2B(
                    real_A).sigmoid(), self.B2A(real_B).sigmoid()
            self.OPTD.zero_grad()
            P_A, R_A = self.A(fake_A).sigmoid(), self.A(real_A).sigmoid()
            P_B, R_B = self.B(fake_B).sigmoid(), self.B(real_B).sigmoid()
            L_A = (F.mse_loss(P_A, torch.zeros_like(P_A)) +
                   F.mse_loss(R_A, torch.ones_like(R_A))) / 2
            L_A.backward()
            L_B = (F.mse_loss(P_B, torch.zeros_like(P_B)) +
                   F.mse_loss(R_B, torch.ones_like(R_B))) / 2
            L_B.backward()
            GD = nn.utils.clip_grad_norm_(
                chain(self.A.parameters(), self.B.parameters()), 100, error_if_nonfinite=False)
            self.OPTD.step()
            Grad["D"].add(GD)
            Loss["A"].add(L_A)
            Loss["B"].add(L_B)

            self.OPTG.zero_grad()
            self.A.requires_grad_(False)
            self.B.requires_grad_(False)
            self.A2B.requires_grad_(True)
            self.B2A.requires_grad_(True)

            real_A, _ = next(A_iter)
            real_B, _ = next(B_iter)
            real_A, real_B = real_A.cuda(), real_B.cuda()
            fake_B, fake_A = self.A2B(
                real_A).sigmoid(), self.B2A(real_B).sigmoid()
            with torch.no_grad():
                P_A, R_A = self.A(fake_A).sigmoid(), self.A(real_A).sigmoid()
                P_B, R_B = self.B(fake_B).sigmoid(), self.B(real_B).sigmoid()

            L_B2A = (F.mse_loss(P_A, torch.ones_like(P_A)) +
                     F.mse_loss(R_A, torch.zeros_like(R_A))) * 2 + \
                10 * F.l1_loss(self.B2A(fake_B).sigmoid(), real_A) + \
                0.5 * F.l1_loss(self.B2A(real_A).sigmoid(), real_A)

            L_B2A.backward(retain_graph=True)
            L_A2B = (F.mse_loss(P_B, torch.ones_like(P_B)) +
                     F.mse_loss(R_B, torch.zeros_like(R_B))) * 2 + \
                10 * F.l1_loss(self.A2B(fake_A).sigmoid(), real_B) + \
                0.5 * F.l1_loss(self.A2B(real_B).sigmoid(), real_B)
            L_A2B.backward()
            GG = nn.utils.clip_grad_norm_(
                chain(self.A2B.parameters(), self.B2A.parameters()), 100, error_if_nonfinite=True)
            self.OPTG.step()
            Loss["A2B"].add(L_A2B)
            Loss["B2A"].add(L_B2A)
            Grad["G"].add(GG)

            self.SCHE_D.step()
            self.SCHE_G.step()

            pbar.set_postfix(**{key: value.last for key, value in Loss.items()},
                             **{key: value.last for key, value in Grad.items()})

            self.tb_logger.add_scalars(
                "Loss", {key: value.last for key, value in Loss.items()}, step)
            self.tb_logger.add_scalars(
                "Grad", {key: value.last for key, value in Grad.items()}, step)
            if i % 100 == 0:
                self.tb_logger.add_image("GAN Image (RealA & FakeB)", torchvision.utils.make_grid(
                    torch.cat([real_A[0], fake_B[0]], dim=1).permute(1, 0, 2, 3).cpu(), nrow=16), step)
                self.tb_logger.add_image("GAN Image B (RealB & Fake)", torchvision.utils.make_grid(
                    torch.cat([real_B[0], fake_A[0]], dim=1).permute(1, 0, 2, 3).cpu(), nrow=16), step)
                
        return {key: value() for key, value in chain(Loss.items(), Grad.items())}

    def save(self, epoch, log_dic):
        met = 0
        if log_dic["A2B"].n > 0:
            met += log_dic["A2B"]
            met += log_dic["B2A"]

        logger = self.logger

        if True:
            self.best_met = met

            log = []
            name = f"CP{epoch:02d}_LOSS{log_dic['Loss']:.4f}.pkl"
            self.A2B.save(path=f"{self.work_dir}/A2B_{name}")
            self.B2A.save(path=f"{self.work_dir}/B2A_{name}")
            self.A.save(path=f"{self.work_dir}/A_{name}")
            self.B.save(path=f"{self.work_dir}/B_{name}")
            log.append(f"Saved a new net: {name}")

            for i in log:
                logger.info(i)

        else:
            logger.info(f"No model was saved")

if __name__ == "__main__":
    Trainer().fit()