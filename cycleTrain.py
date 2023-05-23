import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

from model import UNetModel, NLayerDiscriminator
from model.utils import basicParallel
from datasets.dataset import AFMDataset
from utils.metrics import metStat
from utils.schedular import Scheduler

import numpy as np
import tqdm
import time
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import chain

user = os.environ.get("USER") == "supercgor"
if user:
    from config.config import get_config
else:
    from config.wm import get_config

device = (0, 1)
cfg = get_config()
batch_size = 1 if user else 4
load = {"A2B": "", 
        "B2A": "", 
        "A": "", 
        "B": ""}
workdir = f"./model/pretrain/Cyc-{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
tb_logger = SummaryWriter(f"{workdir}/runs")
A_path = "../data/bulkice"
B_path = "../data/bulkexpR"
epochs = 50
dis_accu = 1

os.makedirs(workdir, exist_ok=True)

A2B = UNetModel(image_size = (16, 128, 128), 
              in_channels = 1, 
              model_channels = 32,
              out_channels = 1,
              num_res_blocks = 1,
              attention_resolutions = [], 
              dropout = 0.1,
              channel_mult = (1,2,2,4), 
              dims = 3, 
              time_embed= None,
              use_checkpoint=False).init().cuda()

B2A = UNetModel(image_size = (16, 128, 128), 
              in_channels = 1, 
              model_channels = 32,
              out_channels = 1,
              num_res_blocks = 1,
              attention_resolutions = [], 
              dropout = 0.1,
              channel_mult = (1,2,2,4), 
              dims = 3, 
              time_embed= None,
              use_checkpoint=False).init().cuda()

A = NLayerDiscriminator(in_channels = 1, 
                        model_channels = 32, 
                        channel_mult = (1,2, 4, 8),
                        max_down_mult = (4, 8, 8),
                        reduce = "mean").cuda()

B = NLayerDiscriminator(in_channels = 1, 
                        model_channels = 32, 
                        channel_mult = (1,2, 4, 8),
                        max_down_mult = (4, 8, 8),
                        reduce = "mean").cuda()

if load["A2B"] != "":
    A2B.load(load["A2B"])
if load["B2A"] != "":
    B2A.load(load["B2A"])
if load["A"] != "":
    A.load(load["A"])
if load["B"] != "":
    B.load(load["B"])
    
if len(device) >= 2:
    A2B = basicParallel(A2B, device_ids = device)
    B2A = basicParallel(B2A, device_ids = device)
    A = basicParallel(A, device_ids = device)
    B = basicParallel(B, device_ids = device)
print("Model loaded")
    
OPTG = torch.optim.Adam(chain(A2B.parameters(), B2A.parameters()), lr=2e-4, betas=(0.5, 0.999))
OPTD = torch.optim.Adam(chain(A.parameters(), B.parameters()), lr=2e-4, betas=(0.5, 0.999))

SCHE_G = Scheduler(OPTG, warmup_steps=1000, decay_factor=50000)
SCHE_D = Scheduler(OPTD, warmup_steps=1000, decay_factor=50000)

from datasets.trans_pic import *
transform_sou = torchvision.transforms.Compose([
    tf.RandomApply([PixelShift(fill=None)], p = 0.5),
    tf.RandomApply([Noisy()], p =0.3),
    tf.RandomApply([Blur()], p = 0.3),
    tf.RandomApply([ColorJitter()], p = 0.3),
])

transform_tag = torchvision.transforms.Compose([
    Noisy(0.05),
    tf.RandomApply([Blur()], p = 0.3),
    ColorJitter(),
])

A_data = AFMDataset(root_path = A_path, preload= True, label = False, transform = transform_sou)
B_data = AFMDataset(root_path = B_path, preload= True, label = False, transform = transform_tag)

A_loader = DataLoader(A_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, drop_last=True)
B_loader = DataLoader(B_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, drop_last=True)

# for i in iter(B_loader):
#     print("OK")

best_loss = 1e10
for e in range(epochs):
    A_iter = iter(A_loader)
    B_iter = iter(B_loader)
    max_iter = min(len(A_iter), len(B_iter)) // (dis_accu + 1)
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
    pbar = tqdm.tqdm(total=max_iter - 1, desc=f"Epoch {e} - Train", position=0, leave=True, unit='it')
    i = 0
    while i < max_iter:
        step = max_iter * e + i
        A.requires_grad_(True)
        B.requires_grad_(True)
        A2B.requires_grad_(False)
        B2A.requires_grad_(False)
        for t in range(dis_accu):
            with torch.no_grad():
                real_A, _ = next(A_iter)
                real_A = real_A.cuda(non_blocking=True)
                fake_B = A2B(real_A).sigmoid()
                real_B, _ = next(B_iter)
                real_B = real_B.cuda(non_blocking=True)
                fake_A = B2A(real_B).sigmoid()
            OPTD.zero_grad()
            P_A = A(fake_A).sigmoid()
            R_A = A(real_A).sigmoid()
            L_A = 0.5 * (F.mse_loss(P_A, torch.zeros_like(P_A)) + F.mse_loss(R_A, torch.ones_like(R_A)))
            L_A.backward()
            P_B = B(fake_B).sigmoid()
            R_B = B(real_B).sigmoid()
            L_B = 0.5 * (F.mse_loss(P_B, torch.zeros_like(P_B)) + F.mse_loss(R_B, torch.ones_like(R_B)))
            L_B.backward()
            GD = nn.utils.clip_grad_norm_(chain(A.parameters(),B.parameters()), 100, error_if_nonfinite=True)
            OPTD.step()
            Loss["A"].add(L_A)
            Loss["B"].add(L_B)
            Grad["D"].add(GD)
            
        A.requires_grad_(False)
        B.requires_grad_(False)
        A2B.requires_grad_(True)
        B2A.requires_grad_(True)
        real_A, _ = next(A_iter)
        real_A = real_A.cuda(non_blocking=True)
        fake_B = A2B(real_A).sigmoid()
        real_B, _ = next(B_iter)
        real_B = real_B.cuda(non_blocking=True)
        fake_A = B2A(real_B).sigmoid()
        P_A = A(fake_A).sigmoid()
        R_A = A(real_A).sigmoid()
        P_B = B(fake_B).sigmoid()
        R_B = B(real_B).sigmoid()
        OPTG.zero_grad()
        L_B2A = 2 * (F.mse_loss(P_A, torch.ones_like(P_A)) + F.mse_loss(R_A, torch.zeros_like(R_A))) + 10 * F.l1_loss(B2A(fake_B).sigmoid(), real_A) + 0.5 * F.l1_loss(B2A(real_A).sigmoid(), real_A)
        L_B2A.backward(retain_graph=True)
        L_A2B = 2 * (F.mse_loss(P_B, torch.ones_like(P_B)) + F.mse_loss(R_B, torch.zeros_like(R_B))) + 10 * F.l1_loss(A2B(fake_A).sigmoid(), real_B) + 0.5 * F.l1_loss(A2B(real_B).sigmoid(), real_B)
        L_A2B.backward()
        GG = nn.utils.clip_grad_norm_(chain(A2B.parameters(),B2A.parameters()), 100, error_if_nonfinite=True)
        OPTG.step()
        Loss["A2B"].add(L_A2B)
        Loss["B2A"].add(L_B2A)
        Grad["G"].add(GG)
        
        SCHE_D.step()
        SCHE_G.step()
        
        pbar.set_postfix(**{key: value.last for key, value in Loss.items()}, **{key: value.last for key, value in Grad.items()})
        
        tb_logger.add_scalars("Loss", {key: value.last for key, value in Loss.items()}, step)
        tb_logger.add_scalars("Grad", {key: value.last for key, value in Grad.items()}, step)
        if i % 100 == 0:
            tb_logger.add_image("GAN Image A (Real & Fake)", torchvision.utils.make_grid(torch.cat([real_A[0],fake_A[0]], dim=1).permute(1,0,2,3).cpu(), nrow=16), step)
            tb_logger.add_image("GAN Image B (Real & Fake)", torchvision.utils.make_grid(torch.cat([real_B[0],fake_B[0]], dim=1).permute(1,0,2,3).cpu(), nrow=16), step)
        i += 1
        pbar.update(1)
    pbar.update(1)
    pbar.close()
    
    e_loss = (Loss["A2B"]() + Loss["B2A"]())/2
    if True:
        A2B.save(f"{workdir}/{e}_A2B_{e_loss}.pkl")
        B2A.save(f"{workdir}/{e}_B2A_{e_loss}.pkl")
        A.save(f"{workdir}/{e}_A_{e_loss}.pkl")
        B.save(f"{workdir}/{e}_B_{e_loss}.pkl")
        print(f"Save model to {workdir}")
    print(f"Epoch {e} - Loss: {e_loss:.4f}")        