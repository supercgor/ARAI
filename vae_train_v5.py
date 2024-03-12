# torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py

import os
import hydra
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset.dataset import PointGridDataset, collate_fn
from torchmetrics import MeanMetric
import model
import utils
import warnings
# warnings.filterwarnings('ignore')

user = os.environ.get('USER') == "supercgor"
config_name = "vae44_local" if user else "vae44_wm"
log_every = 1 if user else 25
    
class Trainer():
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.work_dir = utils.workdir()
        self.log = utils.get_logger()
        self.tblog = SummaryWriter(f"{self.work_dir}/runs")
        self.device = device
        self.iters = 0
        
        self.model = getattr(model, cfg.model.net)(**cfg.model.params).to(device)
        self.model.compile_loss(**cfg.model.losses)
        self.log.info(f"Network parameters: {sum([p.numel() for p in self.model.parameters()])}")
        
        with open(f"{self.work_dir}/model_summary.txt", "w") as f:
            if cfg.model.checkpoint is None:
                self.log.info("Start a new model")
                f.write(f"New model\n")
            else:
                missing_params = utils.model_load(self.model, cfg.model.checkpoint, True)
                self.log.info(f"Load parameters from {cfg.model.checkpoint}")
                f.write(f"Load parameters from {cfg.model.checkpoint}\n")
                f.write(f"Missing parameters: {missing_params}\n")
                
            for line in utils.model_structure(self.model):
                f.write(line + "\n")

        torch.compile(self.model)
        workers = 0 if user else cfg.setting.num_workers
        
        self.train_dts = PointGridDataset(cfg.dataset.train_path, ignore_H=True, random_transform=True)
        self.test_dts = PointGridDataset(cfg.dataset.test_path, ignore_H=True, random_transform=True)
        
        self.train_dtl = DataLoader(self.train_dts, cfg.setting.batch_size, True, num_workers=workers, pin_memory=cfg.setting.pin_memory, collate_fn=collate_fn)
        self.test_dtl = DataLoader(self.test_dts, cfg.setting.batch_size, True, num_workers=workers, pin_memory=cfg.setting.pin_memory, collate_fn=collate_fn)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.criterion.lr, weight_decay=cfg.criterion.weight_decay)
        self.schedular = getattr(torch.optim.lr_scheduler, cfg.criterion.schedular.name)(self.opt, **cfg.criterion.schedular.params)
        
        self.atom_metrics = utils.MetricCollection(
            M = utils.confusion.confusion_matrix(real_size = (25.0, 25.0, 16.0), split = self.cfg.dataset.split, match_distance = 1.0)
        )
        
        self.grid_metrics = utils.MetricCollection(
            loss = MeanMetric(),
            grad = MeanMetric(),
            conf = MeanMetric(),
            off = MeanMetric(),
            kl = MeanMetric(),
        )

        self.save_paths = []
        self.best = np.inf
        
    def fit(self):
        for epoch in range(self.cfg.setting.epoch):
            self.log.info(f"Start training epoch: {epoch}...")
            epoch_start_time = time.time()
            gm = self.train_one_epoch(epoch, log_every = self.cfg.setting.log_every)
            logstr = f"Train Summary: Epoch: {epoch:2d}, Used time: {(time.time() - epoch_start_time)/60:4.1f} mins, last loss: {gm['loss']:.2e}"
            
            gm, atom_metric = self.test_one_epoch(epoch, log_every = self.cfg.setting.log_every)
            loss = gm['loss']
            M = atom_metric['M']
            logstr = f"\n============= Summary Test | Epoch {epoch:2d} ================\nloss: {loss:.2e} | used time: {(time.time() - epoch_start_time)/60:4.1f} mins | {'Model saved' if loss < self.best else 'Model not saved'}"
            logstr += f"\n=================   Element - 'H2O'   ================="
            logstr += f"\n(Overall)  AP: {M[:,3].mean():.2f} | AR: {M[:,4].mean():.2f} | ACC: {M[:,5].mean():.2f} | SUC: {M[:,6].mean():.2f}"
            for i, (low, high) in enumerate(zip(self.cfg.dataset.split[:-1], self.cfg.dataset.split[1:])):
                logstr += f"\n({low:.1f}-{high:.1f}A) AP: {M[i,3]:.2f} | AR: {M[i,4]:.2f} | ACC: {M[i,5]:.2f} | SUC: {M[i,6]:.2f}\nTP: {M[i,0]:10.0f} | FP: {M[i,1]:10.0f} | FN: {M[i,2]:10.0f}"
                
            self.save_model(epoch, loss)
            self.log.info(logstr)
        
    def train_one_epoch(self, epoch, log_every = log_every):
        self.model.train()

        for i, (filenames, targs, atoms) in enumerate(self.train_dtl):
            targs = targs.to(self.device, non_blocking = True)
            self.opt.zero_grad()
                        
            out, latents, conds = self.model(targs)
            
            loss, loss_values = self.model.compute_loss(out, targs, latents, conds)
            loss.backward()
            
            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.criterion.clip_grad, error_if_nonfinite=True)
            self.opt.step()
            
            out_atoms = utils.lib.box2atom(out, (25, 25, 16), 0.5, 2.0)
            
            self.atom_metrics.update(M=(out_atoms, atoms))
            self.grid_metrics.update(loss=loss, grad=grad, conf=loss_values['conf'], off=loss_values['offset'], kl=loss_values['vae'])
            
            self.iters += 1
            
            if i % log_every == 0:
                self.log.info(f"E[{epoch+1:2d}/{self.cfg.setting.epoch}], I[{i:5d}/{len(self.train_dtl):5d}], L{loss:.2e}, G{grad:.2e}")
                utils.log_to_csv(f"{self.work_dir}/train.csv", total= self.iters, epoch=epoch, iter=i, **self.grid_metrics.compute())
                self.grid_metrics.reset()
            
            if user and i > 100:
                break
            
        self.schedular.step()

        return self.grid_metrics.compute()
    
    @torch.no_grad()
    def test_one_epoch(self, epoch, log_every = log_every):
        self.model.eval()
        for i, (filenames, targs, atoms) in enumerate(self.train_dtl):
            targs = targs.to(self.device, non_blocking = True)
                        
            out, latents, conds = self.model(targs)
            
            loss, loss_values = self.model.compute_loss(out, targs, latents, conds)
            
            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.criterion.clip_grad, error_if_nonfinite=True)
            
            out_atoms = utils.lib.box2atom(out, (25, 25, 16), 0.5, 2.0)
            
            self.atom_metrics.update(M=(out_atoms, atoms))
            self.grid_metrics.update(loss=loss, grad=grad, conf=loss_values['conf'], off=loss_values['offset'], kl=loss_values['vae'])
            
            self.iters += 1
            
            if i % log_every == 0:
                self.log.info(f"E[{epoch+1:2d}/{self.cfg.setting.epoch}], I[{i:5d}/{len(self.train_dtl):5d}], L{loss:.2e}, G{grad:.2e}")
                utils.log_to_csv(f"{self.work_dir}/train.csv", total= self.iters, epoch=epoch, iter=i)
            
            if user and i > 100:
                break
        
        return self.grid_metrics.compute(), self.atom_metrics.compute()
        
    def save_model(self, epoch, metric):
        if metric is None or metric < self.best:
            self.best = metric
            path = f"{self.work_dir}/EP{epoch}_L{metric:.2f}.pkl"
            if len(self.save_paths) >= self.cfg.setting.max_save:
                os.remove(self.save_paths.pop(0))
            utils.model_save(self.model, path)
            self.save_paths.append(path)

@hydra.main(config_path="config", config_name=config_name, version_base=None)
def main(cfg):    
    trainer = Trainer(cfg, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    trainer.fit()

if __name__ == "__main__":
    main()
    
