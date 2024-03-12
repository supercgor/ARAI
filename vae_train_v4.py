# torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py

import os
import hydra
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from dataset.dataset import PointGridDataset, collate_fn
import model
import utils

user = os.environ.get('USER') == "supercgor"
config_name = "vae44_local" if user else "vae44_wm"
log_every = 1 if user else 25
    
class Trainer():
    def __init__(self, cfg, model, TrainLoader, TestLoader, Optimizer, Schedular, log, tblog):
        self.work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
        self.cfg = cfg
        self.rank = 0
        self.iters = 0
        self.gpu_id = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.gpu_id)
        self.model.compile_loss(**cfg.model.losses)
        self.TrainLoader = TrainLoader
        self.TestLoader = TestLoader
        self.Optimizer = Optimizer
        self.Schedular = Schedular
        self.Analyser = utils.parallelAnalyser(real_size=(25, 25, 12), split = [0.0, 4.0, 8.0, 12.0]).to(self.gpu_id)
        self.pos_weight = torch.tensor([self.cfg.criterion.pos_weight]).to(self.gpu_id)
        self.ConfusionCounter = utils.ConfusionRotate()
        self.LostStat = utils.metStat()
        self.LostConfidenceStat = utils.metStat()
        self.LostPositionStat = utils.metStat()
        self.LostRotationStat = utils.metStat()
        self.LostVAEStat = utils.metStat()
        self.GradStat = utils.metStat()
        self.RotStat = utils.metStat()
            
        self.log = log
        self.tblog = tblog
        self.save_paths = []
        self.best = np.inf
        
    def fit(self):
        for epoch in range(self.cfg.setting.epoch):
            self.log.info(f"Start training epoch: {epoch}...")
            epoch_start_time = time.time()
            loss = self.train_one_epoch(epoch, log_every = self.cfg.setting.log_every)
            logstr = f"Train Summary: Epoch: {epoch:2d}, Used time: {(time.time() - epoch_start_time)/60:4.1f} mins"
            
            loss, cms, rot = self.test_one_epoch(epoch, log_every = self.cfg.setting.log_every)
            
            logstr = f"\n============= Summary Test | Epoch {epoch:2d} ================\nloss: {loss:.2e} | used time: {(time.time() - epoch_start_time)/60:4.1f} mins | {'Model saved' if (loss < self.best) and (self.rank == 0) else 'Model not saved'}"
            logstr += f"\n=================   Element - 'H2O'   ================="
            logstr += f"\n(Overall)  AP: {cms.AP[0].mean():.2f} | AR: {cms.AR[0].mean():.2f} | ACC: {cms.ACC[0].mean():.2f} | SUC: {cms.SUC[0].mean():.2f} | Mmean: {rot:.2e}"
            for i, (low, high) in enumerate(zip(self.cfg.dataset.split[:-1], self.cfg.dataset.split[1:])):
                logstr += f"\n({low:.1f}-{high:.1f}A) AP: {cms.AP[0,i]:.2f} | AR: {cms.AR[0,i]:.2f} | ACC: {cms.ACC[0,i]:.2f} | SUC: {cms.SUC[0,i]:.2f}\nTP: {cms.TP[0,i]:10.0f} | FP: {cms.FP[0,i]:10.0f} | FN: {cms.FN[0,i]:10.0f}"
                
            self.save_model(epoch, loss)
            self.log.info(logstr)
        
    def train_one_epoch(self, epoch, log_every = log_every) -> tuple[torch.Tensor]:
        self.model.train()
        self.LostStat.reset()
        self.LostConfidenceStat.reset()
        self.LostPositionStat.reset()
        self.LostRotationStat.reset()
        self.LostVAEStat.reset()
        self.GradStat.reset()
        self.RotStat.reset()
        self.ConfusionCounter.reset()

        for i, (filenames, targs, atoms) in enumerate(self.TrainLoader):
            targs = targs.to(self.gpu_id, non_blocking = True)
            self.Optimizer.zero_grad()
            
            condition, target = targs[:,:,:,:2], targs[:,:,:,2:]
            
            x, mu, sigma = self.model(target, condition if epoch >= 0 else None)
            
            loss, loss_values = self.model.compute_loss(x, target, mu, sigma)
                   
            loss.backward()
            
            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.criterion.clip_grad, error_if_nonfinite=True)
            
            self.Optimizer.step()
                        
            CMS = self.Analyser(x, target)
            self.ConfusionCounter.add(CMS[...,:3])
            self.RotStat.add(CMS[...,3].mean(dim=[1,2]))
            self.iters += 1
            
            if self.rank == 0 and i % log_every == 0:
                self.log.info(f"E[{epoch+1:2d}/{self.cfg.setting.epoch}], I[{i:5d}/{len(self.TrainLoader):5d}], L{loss:.2e}, G{grad:.2e}")
                enc_grad = torch.nn.utils.clip_grad_norm_(self.model.vae_enocder.parameters(), self.cfg.criterion.clip_grad, error_if_nonfinite=True)
                cond_grad = torch.nn.utils.clip_grad_norm_(self.model.condional_layer.parameters(), self.cfg.criterion.clip_grad, error_if_nonfinite=True)
                utils.log_to_csv(f"{self.work_dir}/train.csv", total= self.iters, epoch=epoch, iter=i, loss=loss.item(), **loss_values, grad=grad.item(), enc_grad=enc_grad.item(), cond_grad=cond_grad.item())
            
            if user and i > 100:
                break
            
        self.Schedular.step()

        return loss
    
    @torch.no_grad()
    def test_one_epoch(self, epoch, log_every = log_every) -> tuple[torch.Tensor]:
        self.model.eval()
        self.LostStat.reset()
        self.LostConfidenceStat.reset()
        self.LostPositionStat.reset()
        self.LostRotationStat.reset()
        self.LostVAEStat.reset()
        self.ConfusionCounter.reset()
        self.RotStat.reset()
        for i, (filenames, targs, atoms) in enumerate(self.TestLoader):
            targs = targs.to(self.gpu_id, non_blocking = True)
            
            condition, target = targs[:,:,:,:2], targs[:,:,:,2:]
            
            x, mu, sigma = self.model(target, condition if epoch >= 0 else None)
            
            loss, loss_values = self.model.compute_loss(x, target, mu, sigma)
                   
            self.LostStat.add(loss)
            self.LostConfidenceStat.add(loss_values['conf'])
            self.LostPositionStat.add(loss_values['offset'])
            self.LostRotationStat.add(loss_values['rot'])
            self.LostVAEStat.add(loss_values['vae'])
        
            CMS = self.Analyser(x, target)
            self.ConfusionCounter.add(CMS[...,:3])
            self.RotStat.add(CMS[...,3].mean(dim=[1,2]))
            
            if self.rank == 0 and i % log_every == 0:
                self.log.info(f"E[{epoch+1:2d}/{self.cfg.setting.epoch}], I[{i:5d}/{len(self.TestLoader):5d}], L{loss:.2e}")
            
            if user and i > 50:
                break
            
        losses = self.LostStat.calc()
        cms = self.ConfusionCounter.calc()
        rot = self.RotStat.calc()
        
        return losses, cms, rot
        
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
    work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    
    log = utils.get_logger()
    tblog = SummaryWriter(f"{work_dir}/runs")
    net = getattr(model, cfg.model.net)(**cfg.model.params)
        
    log.info(f"Network parameters: {sum([p.numel() for p in net.parameters()])}")
    with open(f"{work_dir}/model_summary.txt", "w") as f:
        if cfg.model.checkpoint is None:
            log.info("Start a new model")
            f.write(f"New model\n")
        else:
            missing = utils.model_load(net, cfg.model.checkpoint, True)
            log.info(f"Load parameters from {cfg.model.checkpoint}")
            f.write(f"Load parameters from {cfg.model.checkpoint}\n")
            if len(missing) > 0:
                f.write(f"Missing parameters: {missing}\n")
        for line in utils.model_structure(net):
            f.write(line + "\n")

    torch.compile(net)
    
    workers = 0 if user else cfg.setting.num_workers
    
    TrainDataset = PointGridDataset(cfg.dataset.train_path, random_transform=True)
    TestDataset = PointGridDataset(cfg.dataset.test_path, random_transform=True)
    TrainLoader = DataLoader(TrainDataset, cfg.setting.batch_size, True, num_workers=workers, pin_memory=cfg.setting.pin_memory, collate_fn=collate_fn)
    TestLoader = DataLoader(TestDataset, cfg.setting.batch_size, True, num_workers=workers, pin_memory=cfg.setting.pin_memory, collate_fn=collate_fn)

    
    params_gp = [[], []]
    
    for name, param in net.named_parameters():
        if param.requires_grad:
            if "enc" in name:
                params_gp[0].append(param)
            else:
                params_gp[1].append(param)    
    Optimizer = torch.optim.Adam([{'params': params_gp[0], 'lr': cfg.criterion.lr * 1.0}, 
                                  {'params': params_gp[1], 'lr': cfg.criterion.lr}],
                                 lr=cfg.criterion.lr, 
                                 weight_decay=cfg.criterion.weight_decay)
    
    Schedular = getattr(torch.optim.lr_scheduler, cfg.criterion.schedular.name)(Optimizer, **cfg.criterion.schedular.params)
    
    trainer = Trainer(cfg, net, TrainLoader, TestLoader, Optimizer, Schedular, log, tblog)
    trainer.fit()
        

if __name__ == "__main__":
    main()
    
