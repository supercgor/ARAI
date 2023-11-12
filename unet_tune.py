# torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py

import os
import hydra
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf, DictConfig
from torchvision.transforms import RandomApply
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, get_rank, destroy_process_group, is_initialized
from dataset.sampler import z_sampler
from functools import partial
import dataset
import model
import utils

user = os.environ.get('USER') == "supercgor"
config_name = "unetv3_local" if user else "unetv3_wm"

def key_filter(key):
    import re
    #return True
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
        self.Analyser = utils.parallelAnalyser(cfg.dataset.real_size, split = self.cfg.dataset.split).to(self.gpu_id)
        
        self.ConfusionCounter = utils.ConfusionRotate()
        self.LostStat = utils.metStat()
        self.GradStat = utils.metStat()
        self.RotStat = utils.metStat()
            
        
        self.Criterion = utils.BoxClsLoss2(cls_weight = cfg.criterion.cond_weight, 
                                           xy_weight = cfg.criterion.xy_weight,
                                           z_weight = cfg.criterion.z_weight,
                                           rot_weight = cfg.criterion.rot_weight,
                                           pos_weight = cfg.criterion.pos_weight
                                          ).to(self.gpu_id)
        
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
        
    def train_one_epoch(self, epoch, log_every: int = 100) -> tuple[torch.Tensor]:
        self.model.train()
        self.LostStat.reset()
        self.GradStat.reset()
        self.RotStat.reset()
        self.ConfusionCounter.reset()
        for i, (filenames, inps, targs, _) in enumerate(self.TrainLoader):
            inps = inps.to(self.gpu_id, non_blocking = True)
            targs = targs.to(self.gpu_id, non_blocking = True)
            preds = self.model(inps)
            loss= self.Criterion(preds, targs)
            
            self.Optimizer.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.criterion.clip_grad, error_if_nonfinite=False)
            self.Optimizer.step()
            
            self.LostStat.add(loss)
            self.GradStat.add(grad)
    
            CMS = self.Analyser(preds, targs)
            self.ConfusionCounter.add(CMS[...,:3])
            self.RotStat.add(CMS[...,3].mean(dim=[1,2]))
            
            if self.rank == 0 and i % log_every == 0:
                self.log.info(f"Epoch {epoch:2d} | Iter {i:5d}/{len(self.TrainLoader):5d} | loss {loss:.2e} | grad {grad:.2e}")
                self.tblog.add_scalar("Train/Loss", loss, epoch * len(self.TrainLoader) + i)
                self.tblog.add_scalar("Train/Grad", grad, epoch * len(self.TrainLoader) + i)
                
        loss = self.LostStat.calc()
        grad = self.GradStat.calc()
        cms = self.ConfusionCounter.calc()
        rot = self.RotStat.calc()
        
        return loss, grad, cms, rot
    
    @torch.no_grad()
    def test_one_epoch(self, epoch, log_every: int = 100) -> tuple[torch.Tensor]:
        self.model.eval()
        self.LostStat.reset()
        self.GradStat.reset()
        self.RotStat.reset()
        self.ConfusionCounter.reset()
        for i, (filenames, inps, targs, _) in enumerate(self.TestLoader):
            inps = inps.to(self.gpu_id, non_blocking = True)
            targs = targs.to(self.gpu_id, non_blocking = True)
            preds = self.model(inps)
            loss= self.Criterion(preds, targs)
            
            self.LostStat.add(loss)
    
            CMS = self.Analyser(preds, targs)
            self.ConfusionCounter.add(CMS[...,:3])
            self.RotStat.add(CMS[...,3].mean(dim=[1,2]))
            
            if self.rank == 0 and i % log_every == 0:
                self.log.info(f"Epoch {epoch:2d} | Iter {i:5d}/{len(self.TestLoader):5d} | loss {loss:.2e}")
                self.tblog.add_scalar("Train/Loss", loss, epoch * len(self.TestLoader) + i)
                
        loss = self.LostStat.calc()
        cms = self.ConfusionCounter.calc()
        rot = self.RotStat.calc()
        
        return loss, cms, rot
    
    def save_model(self, epoch, metric):
        print(self.rank)
        if self.rank == 0:
            # save model
            if metric is None or metric < self.best:
                self.best = metric
                path = f"{self.work_dir}/unetv3tuned_CP{epoch:02d}_L{metric:.4f}.pkl"
                if len(self.save_paths) >= self.cfg.setting.max_save:
                    os.remove(self.save_paths.pop(0))
                utils.model_save(self.model, path)
                self.save_paths.append(path)

def inp_transform(inp: torch.Tensor):
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
    
def load_train_objs(rank, cfg: DictConfig):
    work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    
    log = utils.get_logger(f"Rank {rank}")
    tblog = SummaryWriter(f"{work_dir}/runs")
    
    net = getattr(model, cfg.model.net)(**cfg.model.params)
    cyc = getattr(model, cfg.model.cyc.name)(**cfg.model.cyc.params).eval().requires_grad_(False)
    
    net.apply_transform(inp_transform, out_transform)
    cyc.apply_transform(lambda x: x[None, ...], lambda x: torch.nn.functional.sigmoid(x)[0, ...])
    utils.model_load(cyc, cfg.model.cyc.checkpoint, True)
    
    log.info(f"Network parameters: {sum([p.numel() for p in net.parameters()])}")
    
    if False: # cfg.model.checkpoint is None:
            log.info("Start a new model")
    else:
        missing = utils.model_load(net, cfg.model.checkpoint, True)
        log.info(f"Load parameters from {cfg.model.checkpoint}")
        log.info(f"Missing keys: {missing}")
    
    transform = torch.nn.Sequential(dataset.PixelShift(),
                                    dataset.Cutout(),
                                    dataset.ColorJitter(),
                                    dataset.Noisy(),
                                    dataset.Blur(),
                                    RandomApply([cyc]),
                                    )
        
    out_size = cfg.model.params.out_size
    out_size = (out_size[1], out_size[2], out_size[0])
    
    TrainDataset = dataset.AFMDataset_V2(cfg.dataset.train_path, useLabel=True, useEmb=False, useZ=cfg.dataset.image_size[0], transform=[transform, dataset.labelZnoise()], key_filter=key_filter, label_size=out_size, sampler=partial(z_sampler, is_rand=True))
    TestDataset = dataset.AFMDataset_V2(cfg.dataset.test_path, useLabel=True, useEmb=False, useZ=cfg.dataset.image_size[0], transform=[transform, dataset.labelZnoise()], key_filter=key_filter, label_size=out_size, sampler=partial(z_sampler, is_rand=True))
    
    Optimizer = torch.optim.AdamW(net.parameters(), 
                                    lr=cfg.criterion.lr, 
                                    weight_decay=cfg.criterion.weight_decay
                                    )
    
    Schedular = getattr(torch.optim.lr_scheduler, cfg.criterion.schedular.name)(Optimizer, **cfg.criterion.schedular.params)
    
    return net, TrainDataset, TestDataset, Optimizer, Schedular, log, tblog

def prepare_dataloader(train_data, test_data, cfg: DictConfig):
    if is_initialized():
        TrainLoader = torch.utils.data.DataLoader(train_data, 
                                                batch_size=cfg.setting.batch_size, 
                                                shuffle=False, 
                                                num_workers=cfg.setting.num_workers, 
                                                pin_memory=cfg.setting.pin_memory,
                                                sampler=DistributedSampler(train_data),
                                                )
            
        TestLoader = torch.utils.data.DataLoader(test_data,
                                                batch_size=cfg.setting.batch_size,
                                                shuffle=False,
                                                num_workers=cfg.setting.num_workers,
                                                pin_memory=cfg.setting.pin_memory,
                                                sampler=DistributedSampler(test_data),
                                                )
    else:
        TrainLoader = torch.utils.data.DataLoader(train_data, 
                                                batch_size=cfg.setting.batch_size, 
                                                shuffle=True, 
                                                num_workers=cfg.setting.num_workers, 
                                                pin_memory=cfg.setting.pin_memory,
                                                )
            
        TestLoader = torch.utils.data.DataLoader(test_data,
                                                batch_size=cfg.setting.batch_size,
                                                shuffle=True,
                                                num_workers=cfg.setting.num_workers,
                                                pin_memory=cfg.setting.pin_memory,
                                                )
            
    return TrainLoader, TestLoader
    
@hydra.main(config_path="config", config_name= config_name, version_base=None) # hydra will automatically relocate the working dir.
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
    
