# torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py

import os
import hydra
import time
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, get_rank, destroy_process_group, is_initialized
from dataset.dataset import Point_Grid_Dataset_hdf
import model
import utils

user = os.environ.get('USER') == "supercgor"
config_name = "vae44_local" if user else "vae44_wm"
log_every = 1 if user else 25


def key_filter(key):
    import re
    return True
    #return True if "HDA" in key or "ss" in key else False
    return re.match(r"T\d{1,3}_\d{1,5}", key) is not None or "icehup" in key

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if torch.cuda.device_count() > 1:
        init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        pass
    
class Trainer():
    def __init__(self, rank, cfg, model, TrainLoader, TestLoader, Optimizer, Schedular, log, tblog):
        self.work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
        self.cfg = cfg
        if is_initialized():
            self.rank = get_rank()
        else:
            self.rank = 0
        self.gpu_id = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.gpu_id)
        if is_initialized():
            DDP(self.model, device_ids=[self.gpu_id]) 
        
        self.TrainLoader = TrainLoader
        self.TestLoader = TestLoader
        self.Optimizer = Optimizer
        self.Schedular = Schedular
        
        self.Analyser = utils.parallelAnalyser(real_size=(25,25,16), split = self.cfg.dataset.split).to(self.gpu_id)
        self.pos_weight = torch.tensor([self.cfg.criterion.pos_weight]).to(self.gpu_id)
        self.ConfusionCounter = utils.ConfusionRotate()
        self.LostStat = utils.metStat()
        self.LostConfidenceStat = utils.metStat()
        self.LostPositionStat = utils.metStat()
        self.LostRotationStat = utils.metStat()
        self.LostVAEStat = utils.metStat()
        self.GradStat = utils.metStat()
        self.RotStat = utils.metStat()
            
        self.h_decay = torch.as_tensor([1.0, 1.0, 1.0, 1.0, 
                                        0.9, 0.8, 0.7, 0.6, 
                                        0.5, 0.4, 0.3, 0.3,
                                        0.3, 0.3, 0.2, 0.2], device=self.gpu_id)[None, None, None, :, None] # 1 X 1 X 1 X 16 X 1
        
        self.log = log
        self.tblog = tblog
        self.save_paths = []
        self.best = np.inf
        
    def fit(self):
        for epoch in range(self.cfg.setting.epoch):
            self.log.info(f"Start training epoch: {epoch}...")
            epoch_start_time = time.time()
            loss, grad, cms, rot = self.train_one_epoch(epoch, log_every = self.cfg.setting.log_every)
            logstr = f"\n============== Summary Train | Epoch {epoch:2d} ==============\nloss: {loss:.2e} | grad: {grad:.2e} | used time: {(time.time() - epoch_start_time)/60:4.1f} mins"
            logstr += f"\n=================   Element - 'H2O'   ================="
            logstr += f"\n(Overall)  AP: {cms.AP[0].mean():.2f} | AR: {cms.AR[0].mean():.2f} | ACC: {cms.ACC[0].mean():.2f} | SUC: {cms.SUC[0].mean():.2f} | Mmean: {rot:.2e}"
            for i, (low, high) in enumerate(zip(self.cfg.dataset.split[:-1], self.cfg.dataset.split[1:])):
                logstr += f"\n({low:.1f}-{high:.1f}A) AP: {cms.AP[0,i]:.2f} | AR: {cms.AR[0,i]:.2f} | ACC: {cms.ACC[0,i]:.2f} | SUC: {cms.SUC[0,i]:.2f}\nTP: {cms.TP[0,i]:10.0f} | FP: {cms.FP[0,i]:10.0f} | FN: {cms.FN[0,i]:10.0f}"
            self.log.info(logstr)
            
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
    
        for i, (filenames, inps, targs) in enumerate(self.TrainLoader):
            inps = inps[...,:4].to(self.gpu_id, non_blocking = True)
            targs = targs.to(self.gpu_id, non_blocking = True)
            preds, mu, logvar = self.model(inps)
            
            loss_conf = F.binary_cross_entropy_with_logits(preds[...,(0,)], targs[...,(0,)], pos_weight=self.pos_weight, reduce = 'none')
            loss_conf = (loss_conf * self.h_decay).mean()
            mask = targs[...,0] > 0.5
            loss_pos = F.mse_loss(preds[...,1:4][mask], targs[...,1:4][mask])
            loss_rot = 0 # F.mse_loss(preds[...,4:], targs[...,4:])
            loss_vae = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim = 1).mean()

            loss = self.cfg.criterion.cond_weight * loss_conf + \
                   self.cfg.criterion.xyz_weight * loss_pos + \
                   self.cfg.criterion.rot_weight * loss_rot + \
                   self.cfg.criterion.vae_weight * loss_vae
                        
            self.Optimizer.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.criterion.clip_grad, error_if_nonfinite=False)
            self.Optimizer.step()
            
            self.LostStat.add(loss)
            self.LostConfidenceStat.add(loss_conf)
            self.LostPositionStat.add(loss_pos)
            self.LostRotationStat.add(loss_rot)
            self.LostVAEStat.add(loss_vae)
            self.GradStat.add(grad)
        
            CMS = self.Analyser(preds, targs)
            self.ConfusionCounter.add(CMS[...,:3])
            self.RotStat.add(CMS[...,3].mean(dim=[1,2]))
            
            if self.rank == 0 and i % log_every == 0:
                self.log.info(f"Epoch {epoch:2d} | Iter {i:5d}/{len(self.TrainLoader):5d} | loss {loss:.2e} | grad {grad:.2e} | conf {loss_conf:.2e} | pos {loss_pos:.2e} | rot {loss_rot:.2e} | vae {loss_vae:.2e}")

                
        self.Schedular.step()
        losses = self.LostStat.calc()
        grad = self.GradStat.calc()
        cms = self.ConfusionCounter.calc()
        rot = self.RotStat.calc()
        
        return losses, grad, cms, rot
    
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
        for i, (filenames, inps, targs) in enumerate(self.TestLoader):
            inps = inps[...,:4].to(self.gpu_id, non_blocking = True)
            targs = targs.to(self.gpu_id, non_blocking = True)
            preds, mu, logvar = self.model(inps)
            
            loss_conf = F.binary_cross_entropy_with_logits(preds[...,(0,)], targs[...,(0,)], pos_weight=self.pos_weight, reduce = 'none')
            loss_conf = (loss_conf * self.h_decay).mean()
            mask = targs[...,0] > 0.5
            loss_pos = F.mse_loss(preds[...,1:4][mask], targs[...,1:4][mask])
            loss_rot = 0 # F.mse_loss(preds[...,4:], targs[...,4:])
            loss_vae = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim = 1).mean()

            loss = self.cfg.criterion.cond_weight * loss_conf + \
                   self.cfg.criterion.xyz_weight * loss_pos + \
                   self.cfg.criterion.rot_weight * loss_rot + \
                   self.cfg.criterion.vae_weight * loss_vae
                        
            self.LostStat.add(loss)
            self.LostConfidenceStat.add(loss_conf)
            self.LostPositionStat.add(loss_pos)
            self.LostRotationStat.add(loss_rot)
            self.LostVAEStat.add(loss_vae)
            
            CMS = self.Analyser(preds, targs)
            self.ConfusionCounter.add(CMS[...,:3])
            self.RotStat.add(CMS[...,3])
            
            if self.rank == 0 and i % log_every == 0:
                self.log.info(f"Epoch {epoch:2d} | Iter {i:5d}/{len(self.TestLoader):5d} | loss {loss:.2e}")
            
        losses = self.LostStat.calc()
        cms = self.ConfusionCounter.calc()
        rot = self.RotStat.calc()
        
        return losses, cms, rot
        
    def save_model(self, epoch, metric):
        if self.rank == 0:
            # save model
            if metric is None or metric < self.best:
                self.best = metric
                path = f"{self.work_dir}/vae_CP{epoch:02d}_L{metric:.4f}.pkl"
                if len(self.save_paths) >= self.cfg.setting.max_save:
                    os.remove(self.save_paths.pop(0))
                utils.model_save(self.model, path)
                self.save_paths.append(path)
        
def load_train_objs(rank, cfg: DictConfig):
    work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    
    log = utils.get_logger(f"Rank {rank}")
    tblog = SummaryWriter(f"{work_dir}/runs")
    
    net = getattr(model, cfg.model.net)(**cfg.model.params)
    
    net.apply_transform(inp_transform, out_transform)
    
    log.info(f"Network parameters: {sum([p.numel() for p in net.parameters()])}")
    
    if cfg.model.checkpoint is None:
            log.info("Start a new model")
    else:
        loaded = utils.model_load(net, cfg.model.checkpoint, True)
        log.info(f"Load parameters from {cfg.model.checkpoint}")
            
    TrainDataset = Point_Grid_Dataset_hdf(cfg.dataset.train_path, noise_position= 0.01)
    TestDataset = Point_Grid_Dataset_hdf(cfg.dataset.test_path, noise_position= 0.01)
    
    Optimizer = torch.optim.Adam(net.parameters(), lr=cfg.criterion.lr, weight_decay=cfg.criterion.weight_decay)
    
    Schedular = getattr(torch.optim.lr_scheduler, cfg.criterion.schedular.name)(Optimizer, **cfg.criterion.schedular.params)
    
    return net, TrainDataset, TestDataset, Optimizer, Schedular, log, tblog

def inp_transform(inp: torch.Tensor):
    # B X Y Z C -> B C Z X Y
    inp = inp.permute(0, 4, 3, 1, 2)
    return inp

def out_transform(inp: torch.Tensor):
    # B C Z X Y -> B X Y Z C
    inp = inp.permute(0, 3, 4, 2, 1)
    conf, pos, rotx, roty = torch.split(inp, [1, 3, 3, 3], dim = -1)
    pos = pos.sigmoid()
    c1 = rotx / torch.norm(rotx, dim=-1, keepdim=True)    
    c2 = roty - (c1 * roty).sum(-1, keepdim=True) * c1
    c2 = c2 / torch.norm(c2, dim=-1, keepdim=True)
    return torch.cat([conf, pos, c1, c2], dim=-1)
    
def prepare_dataloader(train_data, test_data, cfg: DictConfig):
    workers = cfg.setting.num_workers if user else 0
    if is_initialized():
        TrainLoader = DataLoader(train_data, cfg.setting.batch_size, False, DistributedSampler(train_data), num_workers=cfg.setting.num_workers, pin_memory=cfg.setting.pin_memory)
            
        TestLoader = DataLoader(test_data, cfg.setting.batch_size, False, DistributedSampler(test_data), num_workers=cfg.setting.num_workers, pin_memory=cfg.setting.pin_memory)
    else:
        TrainLoader = DataLoader(train_data, cfg.setting.batch_size, True, num_workers=workers, pin_memory=cfg.setting.pin_memory)
            
        TestLoader = DataLoader(test_data, cfg.setting.batch_size, True, num_workers=workers, pin_memory=cfg.setting.pin_memory)
            
    return TrainLoader, TestLoader


@hydra.main(config_path="config", config_name=config_name, version_base=None) # hydra will automatically relocate the working dir.
def main(cfg):
    if user:
        rank = 0
    else:
        if "LOCAL_RANK" in os.environ:
            rank = int(os.environ["LOCAL_RANK"])
        else:
            rank = 0
        world_size = torch.cuda.device_count()
        ddp_setup(rank, world_size)
        
    model, TrainDataset, TestDataset, Optimizer, Schedular, log, tblog = load_train_objs(rank, cfg)
    TrainLoader, TestLoader = prepare_dataloader(TrainDataset, TestDataset, cfg)
    trainer = Trainer(rank, cfg, model, TrainLoader, TestLoader, Optimizer, Schedular, log, tblog)
    trainer.fit()
        
    if is_initialized():
        destroy_process_group()

if __name__ == "__main__":
    main()
    
