# torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py

import os
import hydra
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import binary_cross_entropy as bce
from torch.nn.functional import mse_loss as mse
from torch.nn.functional import l1_loss as l1
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image, make_grid

#chain
from itertools import chain

import dataset
import model
import utils

class Trainer():
    def __init__(self, rank, cfg, models, dts, loaders, opts, log, tblog):
        self.work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
        self.cfg = cfg
        self.rank = 0
        self.gpu_id = rank
        self.models = models
        for m in self.models:
            m.to(self.gpu_id)
        self.opt = opts
        self.LostStat = utils.metStat()
        self.GradStat = utils.metStat()
        self.datasets = dts
        self.loader = loaders
        self.sou = iter(loaders[0])
        self.tar = iter(loaders[1])
        self.fid = FrechetInceptionDistance(feature=64, reset_real_features=False, normalize=True).to(self.gpu_id)
        self.log = log
        self.tblog = tblog
        self.save_paths = []
        self.best = np.inf
        l = len(self.datasets[0])
        self.test_real_a = self.datasets[0][l//2][1].unsqueeze(0)
        self.test_real_b = self.datasets[1][0][1].unsqueeze(0)
        save_image(self.test_real_a[0].transpose(0,1), f"{self.work_dir}/real_a.png", pad_value = 0.5)
        save_image(self.test_real_b[0].transpose(0,1), f"{self.work_dir}/real_b.png", pad_value = 0.5)
        
    def fit(self, iters = 10000, log_every = 2):
        epoch_start_time = time.time()
        self.log.info(f"Start training....")
        it = 0
        nets2t, nett2s, disct, discs = self.models
        nets2t.train()
        nett2s.train()
        disct.train()
        discs.train()
        while it < iters:
            try:
                filenames_s, real_s = next(self.sou)
            except StopIteration:
                self.sou = iter(self.loader[0])
                filenames_s, real_s = next(self.sou)
            try:
                filenames_t, real_t = next(self.tar)
            except StopIteration:
                self.tar = iter(self.loader[1])
                filenames_t, real_t = next(self.tar)
            real_s = real_s.to(self.gpu_id, non_blocking = True)
            real_t = real_t.to(self.gpu_id, non_blocking = True)
            if it % 2 == 0:
                nets2t.requires_grad_(False)
                nett2s.requires_grad_(False)
                disct.requires_grad_(True)
                discs.requires_grad_(True)
                with torch.no_grad():
                    fake_s = nett2s(real_t)
                    fake_t = nets2t(real_s)
                
                d_fake_s = discs(fake_s)
                d_fake_t = disct(fake_t)
                d_real_s = discs(real_s)
                d_real_t = disct(real_t)
                
                loss_d = bce(d_fake_s, torch.zeros_like(d_fake_s)) + \
                         bce(d_fake_t, torch.zeros_like(d_fake_t)) + \
                         bce(d_real_s, torch.ones_like(d_real_s)) + \
                         bce(d_real_t, torch.ones_like(d_real_t))
                loss_d = loss_d / 2
                self.opt[1].zero_grad()
                loss_d.backward()
                grad_d = torch.nn.utils.clip_grad_norm_(chain(disct.parameters(), discs.parameters()), self.cfg.criterion.clip_grad, error_if_nonfinite=False)
                self.opt[1].step()
                
            else:
                nets2t.requires_grad_(True)
                nett2s.requires_grad_(True)
                disct.requires_grad_(False)
                discs.requires_grad_(False)
                fake_s = nett2s(real_t)
                fake_t = nets2t(real_s)
                with torch.no_grad():
                    d_fake_s = discs(fake_s)
                    d_fake_t = disct(fake_t)
                
                recon_s = nett2s(fake_t)
                recon_t = nets2t(fake_s)
                idt_s = nett2s(real_s)
                idt_t = nets2t(real_t)
                
                loss_g = 2 * bce(d_fake_s, torch.ones_like(d_fake_s)) + \
                       2 * bce(d_fake_t, torch.ones_like(d_fake_t)) + \
                       10 * l1(recon_s, real_s) + \
                       10 * l1(recon_t, real_t) + \
                       2 * l1(idt_s, real_s) + \
                       2 * l1(idt_t, real_t)
                
                loss_g = loss_g / 14
                
                self.opt[0].zero_grad()
                loss_g.backward()
                grad_g = torch.nn.utils.clip_grad_norm_(chain(nets2t.parameters(), nett2s.parameters()), 10, error_if_nonfinite=False)
                self.opt[0].step()
                
                self.fid.update(fake_t[:,0, (0, 3, 6)], False)
                self.fid.update(real_t[:,0, (0, 3, 6)], True)
            
            if it > 1 and it % log_every == 0:
                fid = self.fid.compute()
                with torch.no_grad():
                    nets2t.eval()
                    fake_b = nets2t(self.test_real_a.to(self.gpu_id))[0] # C D H W
                    fake_b = fake_b.transpose(0, 1)
                    save_image(fake_b, f"{self.work_dir}/fake_{it:05d}.png", pad_value = 0.5)
                    nets2t.train()
                self.log.info(f"Epoch {it:5d}/{iters:5d} | fid {fid:5.2f} |loss_d {loss_d:.2e} | loss_g {loss_g:.2e} | grad_d {grad_d:.2e} | grad_g {grad_g:.2e}")
                self.tblog.add_scalar("Train/FID", fid, it)
                self.tblog.add_scalar("Train/Loss_d", loss_d, it)
                self.tblog.add_scalar("Train/Loss_g", loss_g, it)
                self.tblog.add_scalar("Train/Grad_d", grad_d, it)
                self.tblog.add_scalar("Train/Grad_g", grad_g, it)
                                
            if it > log_every and it % 100 == 0:
                self.save_model(it, metric = self.fid.compute())
                
            it+=1
            
        self.save_model("end", metric = self.fid.compute())
            
    def save_model(self, epoch, metric):
        if self.rank == 0: # save model
            path = f"{self.work_dir}/cycv0_It{epoch:02d}_fid{metric:.4f}"
            if len(self.save_paths) >= self.cfg.setting.max_save:
                old = self.save_paths.pop(0)
                os.remove(f"{old}_neta.pth")
                os.remove(f"{old}_netb.pth")
                os.remove(f"{old}_disca.pth")
                os.remove(f"{old}_discb.pth")
            utils.model_save(self.models[0], f"{path}_neta.pth")
            utils.model_save(self.models[1], f"{path}_netb.pth")
            utils.model_save(self.models[2], f"{path}_disca.pth")
            utils.model_save(self.models[3], f"{path}_discb.pth")
            self.save_paths.append(path)
    
def load_train_objs(rank, cfg):
    work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    
    log = utils.get_logger(f"Rank {rank}")
    tblog = SummaryWriter(f"{work_dir}/runs")
    
    neta = getattr(model, cfg.model.net.name)(**cfg.model.net.params)
    netb = getattr(model, cfg.model.net.name)(**cfg.model.net.params)
    disca = getattr(model, cfg.model.disc.name)(**cfg.model.disc.params)
    discb = getattr(model, cfg.model.disc.name)(**cfg.model.disc.params)
        
    neta.apply_transform(None, torch.nn.functional.sigmoid)
    netb.apply_transform(None, torch.nn.functional.sigmoid)
    disca .apply_transform(None, torch.nn.functional.sigmoid)
    discb.apply_transform(None, torch.nn.functional.sigmoid)
    
    log.info(f"Generater parameters: {sum([p.numel() for p in neta.parameters()])}")
    log.info(f"Discriminator parameters: {sum([p.numel() for p in disca.parameters()])}")
    
    if True: # cfg.model.checkpoint is None:
            log.info("Start a new model")
    else:
        missing = utils.model_load(net, cfg.model.checkpoint, True)
        log.info(f"Load parameters from {cfg.model.checkpoint}")
        log.info(f"Missing keys: {missing}")
    
    sourcetransform = torch.nn.Sequential(dataset.PixelShift(),
                                    dataset.Cutout(),
                                    dataset.ColorJitter(),
                                    dataset.Noisy(),
                                    dataset.Blur()
                                    )
    targettransform = torch.nn.Sequential(dataset.ColorJitter(),
                                          dataset.Noisy(0.03)
                                         )
        
    sourceDataset = dataset.AFMDataset_V2(cfg.dataset.source_path, useLabel=False, useEmb=False, transform=sourcetransform)
    targetDataset = dataset.AFMDataset_V2(cfg.dataset.target_path, useLabel=False, useEmb=False, transform=targettransform)
    
    genopt = torch.optim.Adam(chain(neta.parameters(), netb.parameters()), 
                                        lr=cfg.criterion.lr, 
                                        weight_decay=cfg.criterion.weight_decay,
                                        betas=(0.5, 0.999)
                                        )
    
    discopt = torch.optim.SGD(chain(disca.parameters(), discb.parameters()), 
                                        lr=4 * cfg.criterion.lr, 
                                        weight_decay=cfg.criterion.weight_decay
                                        )
        
    return [neta, netb, disca, discb], [sourceDataset, targetDataset], [genopt, discopt], log, tblog

def prepare_dataloader(datasets, cfg):
    sourceLoader = DataLoader(datasets[0], 
                              batch_size=cfg.setting.batch_size, 
                              shuffle=True, 
                              num_workers=cfg.setting.num_workers, 
                              pin_memory=cfg.setting.pin_memory,
                              )
        
    targetLoader = DataLoader(datasets[1],
                            batch_size=cfg.setting.batch_size,
                            shuffle=True,
                            num_workers=cfg.setting.num_workers,
                            pin_memory=cfg.setting.pin_memory,
                            )
    
    return [sourceLoader, targetLoader]


@hydra.main(config_path="config", config_name="cycv0_local", version_base=None) # hydra will automatically relocate the working dir.
def main(cfg):
    rank = "cuda" if torch.cuda.is_available() else "cpu"
    models, datasets, opts, log, tblog = load_train_objs(rank, cfg)
    loaders = prepare_dataloader(datasets, cfg)
    trainer = Trainer(rank, cfg, models, datasets, loaders, opts, log, tblog)
    trainer.fit(10000, cfg.setting.log_every)

if __name__ == "__main__":
    main()
    
