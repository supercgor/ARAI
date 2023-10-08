import os
import hydra
import time

import numpy as np
import torch
from torch import multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, get_rank, get_world_size, destroy_process_group
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf, DictConfig

import dataset
import model
import utils

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
class Trainer():
    def __init__(self, 
                 cfg: DictConfig,
                 model: torch.nn.Module,
                 TrainLoader: torch.optim.Optimizer,
                 TestLoader: torch.optim.Optimizer,
                 log,
                 tblog,
                 rank: int,
                 gpu_id: int,
                ) -> None:
        self.cfg = cfg
        self.rank = rank
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.save_paths = []
        self.TrainLoader = TrainLoader
        self.TestLoader = TestLoader
        self.log = log
        self.tblog = tblog
        #Analyser = torch.jit.script(utils.Analyser(cfg.data.real_size, cfg.data.nms, split = OmegaConf.to_object(cfg.data.split))).to(device)
        self.Analyser = utils.Analyser(cfg.data.real_size, cfg.data.nms, split = OmegaConf.to_object(cfg.data.split)).to(self.gpu_id)
        self.ConfusionMatrixCounter = utils.ConfusionMatrixCounter()
        self.LostStat = utils.metStat()
        self.GradStat = utils.metStat()
        
        self.Criterion = utils.BoxClsLoss(wxy = cfg.criterion.xy_weight, 
                                     wz = cfg.criterion.z_weight, 
                                     wcls = cfg.criterion.cond_weight, 
                                     ion_weight=cfg.criterion.pos_weight
                                     ).to(self.gpu_id)
        
        self.best = np.inf
        
    def fit(self, epochs: int):
        for epoch in range(self.cfg.setting.epoch):
            self.log.info(f"Start training epoch: {epoch}...")
            epoch_start_time = time.time()
            train_loss, train_grad, train_confusion = self.train_one_epoch()
            logstr = f"""
            ============= Summary Train | Epoch {epoch:2d} ================
            loss: {train_loss:.2e} | grad: {train_grad:.2e} | used time: {(time.time() - epoch_start_time)/60:4.1f} mins
            """
            
            for i, e in enumerate(utils.ion_order.split()):
                if e in self.cfg.data.ion_order:
                    logstr += f"""
            =================   Element - '{e}'    =================
            (Overall)  AP: {train_confusion[i,:,4].mean():.2f} | AR: {train_confusion[i,:,3].mean():.2f} | ACC: {train_confusion[i,:,5].mean():.2f} | SUC: {train_confusion[i,:,6].mean():.2f}
            """
                    for j, (low, high) in enumerate(zip(self.cfg.data.split[:-1], self.cfg.data.split[1:])):
                        logstr += f"""
            ({low:.1f}-{high:.1f}A) AP: {train_confusion[i,j,4]:.2f} | AR: {train_confusion[i,j,3]:.2f} | ACC: {train_confusion[i,j,5]:.2f} | SUC: {train_confusion[i,j,6]:.2f}
            TP: {train_confusion[i,j,0]:10d} | FP: {train_confusion[i,j,1]:10d} | FN: {train_confusion[i,j,2]:10d}
            """
            log.info(logstr)
            
            test_loss, test_confusion = self.test_one_epoch()
            
            logstr = f"""
            ============= Summary Test | Epoch {epoch:2d} ================
            loss: {test_loss:.2e} | used time: {(time.time() - epoch_start_time)/60:4.1f} mins | {'Model saved' if (test_loss < best) and (local_rank == 0) else 'Model not saved'}
            """
            
            for i, e in enumerate(utils.ion_order.split()):
                if e in cfg.data.ion_order:
                    logstr += f"""
            =================   Element - '{e}'    =================
            (Overall)  AP: {test_confusion[i,:,4].mean():.2f} | AR: {test_confusion[i,:,3].mean():.2f} | ACC: {test_confusion[i,:,5].mean():.2f} | SUC: {test_confusion[i,:,6].mean():.2f}
            """
                    for j, (low, high) in enumerate(zip(self.cfg.data.split[:-1], self.cfg.data.split[1:])):
                        logstr += f"""
            ({low:.1f}-{high:.1f}A) AP: {test_confusion[i,j,4]:.2f} | AR: {test_confusion[i,j,3]:.2f} | ACC: {test_confusion[i,j,5]:.2f} | SUC: {test_confusion[i,j,6]:.2f}
            TP: {test_confusion[i,j,0]:10d} | FP: {test_confusion[i,j,1]:10d} | FN: {test_confusion[i,j,2]:10d}
            """
            
            self.log.info(logstr)
        
    def train_one_epoch(self, epoch, log_every: int = 25) -> tuple[torch.Tensor]:
        self.LostStat.reset()
        self.GradStat.reset()
        self.ConfusionMatrixCounter.reset()
        for i, (filename, afm, targ_type, targ_pos) in enumerate(self.TrainLoader):
            afm = afm.to(self.gpu_id, non_blocking = True)
            targ_type = targ_type.to(self.gpu_id, non_blocking = True)
            targ_pos = targ_pos.to(self.gpu_id, non_blocking = True)
            pred_type, pred_pos, mu, var = self.model(afm)
            loss = self.Criterion(pred_type, pred_pos, targ_type, targ_pos)
            self.Optimizer.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model.criterion.clip_grad, error_if_nonfinite=True)
            self.LostStat.add(loss)
            self.GradStat.add(grad)
            confusion_matrix = self.Analyser(pred_type, pred_pos, targ_type, targ_pos)
            self.ConfusionMatrixCounter(confusion_matrix)
            self.Optimizer.step()
            if self.rank == 0 and i % log_every == 0:
                self.log.info(f"Epoch {epoch} | Iter {i}/{len(self.TrainLoader)} | Loss {loss:.2e} | Grad {grad:.2e}")
                self.tblog.add_scalar("Train/Loss", loss, epoch * len(self.TrainLoader) + i)
                self.tblog.add_scalar("Train/Grad", grad, epoch * len(self.TrainLoader) + i)
        self.Schedular.step()
        return self.LostStat.calc(), self.GradStat.calc(), self.ConfusionMatrixCounter.calc()
    
    
    @torch.no_grad()
    def test_one_epoch(self, epoch, log_every: int = 25) -> tuple[torch.Tensor]:
        self.LostStat.reset()
        self.ConfusionMatrixCounter.reset()
        for i, (filename, afm, targ_type, targ_pos) in enumerate(self.TestLoader):
            print(i)
            afm = afm.to(self.gpu_id, non_blocking = True)
            targ_type = targ_type.to(self.gpu_id, non_blocking = True)
            targ_pos = targ_pos.to(self.gpu_id, non_blocking = True)
            pred_type, pred_pos, mu, var = self.model(afm)
            loss = self.Criterion(pred_type, pred_pos, targ_type, targ_pos)
            confusion_matrix = self.Analyser(pred_type, pred_pos, targ_type, targ_pos)
            self.LostStat.add(loss)
            self.ConfusionMatrixCounter(confusion_matrix)
            if self.rank == 0 and i % log_every == 0:
                self.log.info(f"Epoch {epoch} | Iter {i}/{len(self.TestLoader)} | Loss {loss:.2e}")
                self.tblog.add_scalar("Test/Loss", loss, epoch * len(self.TestLoader) + i)
                
        return self.LostStat.calc(), self.ConfusionMatrixCounter.calc()
    
    def save_model(self, epoch, metric):
        if self.rank == 0:
            # save model
            if metric is None or metric < self.best:
                self.best = metric
                path = f"{self.work_dir}/unet_CP{epoch:02d}_L{metric:.4f}.pkl"
                if len(self.save_paths) >= self.cfg.model.max_save:
                    os.remove(self.save_paths.pop(0))
                utils.model_save(self.model, path)
                self.save_paths.append(path)
                
def load_train_objs(cfg: DictConfig):
    work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    
    log = utils.get_logger(f"[{get_rank()}]Train")
    tblog = SummaryWriter(work_dir)
    
    net = getattr(model, cfg.model.net)(**cfg.model.params)
    
    log.info(f"Network parameters: {sum([p.numel() for p in net.parameters()])}")
    
    if cfg.model.checkpoint is None:
            log.info("Start a new model")
    else:
        missing = utils.model_load(net, cfg.model.checkpoint, True)
        log.info(f"Load parameters from {cfg.model.checkpoint}")
        log.info(f"Missing keys: {missing}")
    
    transform = torch.nn.Sequential(dataset.Resize(tuple(cfg.data.image_size[1:])),
                                    dataset.PixelShift(),
                                    dataset.Cutout(),
                                    dataset.ColorJitter(),
                                    dataset.Noisy(),
                                    dataset.Blur()
                                    )
        
    TrainDataset = dataset.AFMDataset(cfg.data.train_path, useLabel=True, useZ=cfg.data.image_size[0], transform=transform)
    TestDataset = dataset.AFMDataset(cfg.data.test_path, useLabel=True, useZ=cfg.data.image_size[0], transform=transform)
    
    Optimizer = torch.optim.AdamW(net.parameters(), 
                                    lr=cfg.criterion.lr, 
                                    weight_decay=cfg.criterion.weight_decay
                                    )
    
    Schedular = getattr(torch.optim.lr_scheduler, cfg.criterion.schedular.name)(Optimizer, **cfg.criterion.schedular.params)
    
    return net, TrainDataset, TestDataset, Optimizer, Schedular, log, tblog

def prepare_dataloader(train_data, test_data, cfg: DictConfig):
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
    
    return TrainLoader, TestLoader


@hydra.main(config_path="conf", config_name="config", version_base=None) # hydra will automatically relocate the working dir.
def main(rank, world_size, cfg: DictConfig) -> None:
    ddp_setup(rank, world_size)
    model, TrainDataset, TestDataset, Optimizer, Schedular, log, tblog = load_train_objs(cfg)
    TrainLoader, TestLoader = prepare_dataloader(TrainDataset, TestDataset, cfg)
    trainer = Trainer(cfg, model, TrainLoader, TestLoader, log, tblog, rank, gpu_id=rank)
    trainer.fit()
    destroy_process_group()
    
            
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, ), nprocs=world_size)
