# torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py

import os
import hydra
import time

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, get_rank, destroy_process_group
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

def out_transform(inp: torch.Tensor):
    inp = inp.permute(0, 2, 3, 4, 1)
    conf, pos, rotx, roty = torch.split(inp, [1, 3, 3, 3], dim = -1)
    pos = pos.sigmoid()
    c1 = rotx / torch.norm(rotx, dim=-1, keepdim=True)    
    c2 = roty - (c1 * roty).sum(-1, keepdim=True) * c1
    c2 = c2 / torch.norm(c2, dim=-1, keepdim=True)
    return torch.cat([conf, pos, c1, c2], dim=-1)
    
class Trainer():
    def __init__(self, 
                 work_dir: str,
                 cfg: DictConfig,
                 model: torch.nn.Module,
                 TrainLoader,
                 TestLoader,
                 Optimizer: torch.optim.Optimizer,
                 Schedular: torch.optim.lr_scheduler,
                 log,
                 tblog,
                 gpu_id: int,
                ):
        self.work_dir = work_dir
        self.cfg = cfg
        self.rank = get_rank()
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)
        self.save_paths = []
        self.TrainLoader = TrainLoader
        self.TestLoader = TestLoader
        self.Optimizer = Optimizer
        self.Schedular = Schedular
        self.GradScaler = torch.cuda.amp.GradScaler()
        self.log = log
        self.tblog = tblog
        self.Analyser = utils.parallelAnalyser(cfg.data.real_size, cfg.data.nms).to(gpu_id)
        self.ConfusionCounter = utils.ConfusionRotate()
        self.LostStat = utils.metStat()
        self.LostConfidenceStat = utils.metStat()
        self.LostPositionStat = utils.metStat()
        self.LostRotationStat = utils.metStat()
        self.LostVAEStat = utils.metStat()
        self.GradStat = utils.metStat()
        self.RotStat = utils.metStat()
        
        self.Criterion = utils.conditionVAELoss(wc = cfg.criterion.cond_weight,
                                                wpos_weight = cfg.criterion.pos_weight,
                                                wpos = cfg.criterion.xyz_weight,
                                                wr = cfg.criterion.rot_weight,
                                                wvae = cfg.criterion.vae_weight
                                                ).to(self.gpu_id)
        
        self.best = np.inf
        
    def fit(self):
        for epoch in range(self.cfg.setting.epoch):
            self.log.info(f"Start training epoch: {epoch}...")
            epoch_start_time = time.time()
            loss, grad, cms, rot = self.train_one_epoch(epoch, log_every = self.cfg.setting.log_every)
            logstr = f"\n============== Summary Train | Epoch {epoch:2d} ==============\nloss: {loss[0]:.2e} | grad: {grad:.2e} | used time: {(time.time() - epoch_start_time)/60:4.1f} mins"
            logstr += f"\nLoss | Confidence {loss[1]:.2e} | Position {loss[2]:.2e} | Rotation {loss[3]:.2e} | VAE {loss[4]:.2e}"
            logstr += f"\n=================   Element - 'H2O'   ================="
            logstr += f"\n(Overall)  AP: {cms.AP[0].mean():.2f} | AR: {cms.AP[0].mean():.2f} | ACC: {cms.ACC[0].mean():.2f} | SUC: {cms.SUC[0].mean():.2f} | Mmean: {rot:.2e}"
            logstr += f"\n({4:.1f}-{8:.1f}A) AP: {cms.AP[0,0]:.2f} | AR: {cms.AR[0,0]:.2f} | ACC: {cms.ACC[0,0]:.2f} | SUC: {cms.SUC[0,0]:.2f}\nTP: {cms.TP[0,0]:10.0f} | FP: {cms.FP[0,0]:10.0f} | FN: {cms.FN[0,0]:10.0f}"
            self.log.info(logstr)
            
            loss, cms, rot = self.test_one_epoch(epoch, log_every = self.cfg.setting.log_every)
            
            logstr = f"\n============= Summary Test | Epoch {epoch:2d} ================\nloss: {loss[0]:.2e} | used time: {(time.time() - epoch_start_time)/60:4.1f} mins | {'Model saved' if (loss[0] < self.best) and (self.rank == 0) else 'Model not saved'}"
            logstr += f"\n=================   Element - 'H20'   =================\n(Overall)  AP: {cms.AP[0].mean():.2f} | AR: {cms.AR[0].mean():.2f} | ACC: {cms.ACC[0].mean():.2f} | SUC: {cms.SUC[0].mean():.2f} | Mmean: {rot:.2e}"
            logstr += f"\n({4:.1f}-{8:.1f}A) AP: {cms.AP[0,0]:.2f} | AR: {cms.AR[0,0]:.2f} | ACC: {cms.ACC[0,0]:.2f} | SUC: {cms.SUC[0,0]:.2f}\nTP: {cms.TP[0,0]:10.0f} | FP: {cms.FP[0,0]:10.0f} | FN: {cms.FN[0,0]:10.0f}"
            
            self.save_model(epoch, loss[0])
            self.log.info(logstr)
        
    def train_one_epoch(self, epoch, log_every: int = 100) -> tuple[torch.Tensor]:
        self.model.train()
        self.LostStat.reset()
        self.LostConfidenceStat.reset()
        self.LostPositionStat.reset()
        self.LostRotationStat.reset()
        self.LostVAEStat.reset()
        self.GradStat.reset()
        self.RotStat.reset()
        self.ConfusionCounter.reset()
        for i, (filenames, inps, targs, embs) in enumerate(self.TrainLoader):
            inps = inps.to(self.gpu_id, non_blocking = True)
            targs = targs.to(self.gpu_id, non_blocking = True)
            embs = embs.to(self.gpu_id, non_blocking = True)
            with torch.autocast(device_type='cuda'):
                preds, mu, var = self.model(inps, embs)
                preds = out_transform(preds)
                loss_wc, loss_pos, loss_r, loss_vae = self.Criterion(preds, targs, mu, var)
                loss = loss_wc + loss_pos + loss_r + loss_vae
            self.Optimizer.zero_grad()
            self.GradScaler.scale(loss).backward()
            self.GradScaler.unscale_(self.Optimizer)
            
            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.criterion.clip_grad, error_if_nonfinite=False)
            
            self.LostStat.add(loss)
            self.LostConfidenceStat.add(loss_wc)
            self.LostPositionStat.add(loss_pos)
            self.LostRotationStat.add(loss_r)
            self.LostVAEStat.add(loss_vae)
            self.GradStat.add(grad)
            
            CMS = self.Analyser(preds, targs)
            
            self.ConfusionCounter.add(CMS[...,:3])
            self.RotStat.add(CMS[...,3].mean(dim=[1,2]))
            
            self.GradScaler.step(self.Optimizer)
            self.GradScaler.update()
            
            if self.rank == 0 and i % log_every == 0:
                self.log.info(f"Epoch {epoch:2d} | Iter {i:5d}/{len(self.TrainLoader):5d} | Loss {loss:.2e} | Grad {grad:.2e}\n Loss | Confidence {loss_wc:.2e} | Position {loss_wc:.2e} | Rotation {loss_r:.2e} | VAE {loss_vae:.2e}")
                self.tblog.add_scalar("Train/Loss", loss, epoch * len(self.TrainLoader) + i)
                self.tblog.add_scalar("Train/LossWC", loss_wc, epoch * len(self.TrainLoader) + i)
                self.tblog.add_scalar("Train/LossPos", loss_pos, epoch * len(self.TrainLoader) + i)
                self.tblog.add_scalar("Train/LossRot", loss_r, epoch * len(self.TrainLoader) + i)
                self.tblog.add_scalar("Train/LossVAE", loss_vae, epoch * len(self.TrainLoader) + i)
                self.tblog.add_scalar("Train/Grad", grad, epoch * len(self.TrainLoader) + i)
                self.tblog.add_scalar("Train/LR", self.Schedular.get_last_lr()[0], epoch * len(self.TrainLoader) + i)
                 
        self.Schedular.step()
        losses = [self.LostStat.calc(),self.LostConfidenceStat.calc(), self.LostPositionStat.calc(), self.LostRotationStat.calc(), self.LostVAEStat.calc()]
        grad = self.GradStat.calc()
        cms = self.ConfusionCounter.calc()
        rot = self.RotStat.calc()
        
        return losses, grad, cms, rot
    
    @torch.no_grad()
    def test_one_epoch(self, epoch, log_every: int = 100) -> tuple[torch.Tensor]:
        self.model.eval()
        self.LostStat.reset()
        self.LostConfidenceStat.reset()
        self.LostPositionStat.reset()
        self.LostRotationStat.reset()
        self.LostVAEStat.reset()
        self.ConfusionCounter.reset()
        self.RotStat.reset()
        for i, (filenames, inps, targs, embs) in enumerate(self.TestLoader):
            inps = inps.to(self.gpu_id, non_blocking = True)
            targs = targs.to(self.gpu_id, non_blocking = True)
            embs = embs.to(self.gpu_id, non_blocking = True)
            with torch.autocast(device_type='cuda'):
                preds, mu, var = self.model(inps, embs)
                preds = out_transform(preds)
                loss_wc, loss_pos, loss_r, loss_vae = self.Criterion(preds, targs, mu, var)
                loss = loss_wc + loss_pos + loss_r + loss_vae
            
            self.LostStat.add(loss)
            self.LostConfidenceStat.add(loss_wc)
            self.LostPositionStat.add(loss_pos)
            self.LostRotationStat.add(loss_r)
            self.LostVAEStat.add(loss_vae)
            
            CMS = self.Analyser(preds, targs)
            self.ConfusionCounter.add(CMS[...,:3])
            self.RotStat.add(CMS[...,3])
            
            if self.rank == 0 and i % log_every == 0:
                self.log.info(f"Epoch {epoch:2d} | Iter {i:5d}/{len(self.TestLoader):5d} | Loss {loss:.2e}")
                self.tblog.add_scalar("Test/Loss", loss, epoch * len(self.TestLoader) + i)
                self.tblog.add_scalar("Test/LossWC", loss_wc, epoch * len(self.TestLoader) + i)
                self.tblog.add_scalar("Test/LossPos", loss_pos, epoch * len(self.TestLoader) + i)
                self.tblog.add_scalar("Test/LossRot", loss_r, epoch * len(self.TestLoader) + i)
                self.tblog.add_scalar("Test/LossVAE", loss_vae, epoch * len(self.TestLoader) + i)
        
        losses = [self.LostStat.calc(),self.LostConfidenceStat.calc(), self.LostPositionStat.calc(), self.LostRotationStat.calc(), self.LostVAEStat.calc()]
        cms = self.ConfusionCounter.calc()
        rot = self.RotStat.calc()
        
        return losses, cms, rot
    
    def save_model(self, epoch, metric):
        print(self.rank)
        if self.rank == 0:
            # save model
            if metric is None or metric < self.best:
                self.best = metric
                path = f"{self.work_dir}/vae_CP{epoch:02d}_L{metric:.4f}.pkl"
                if len(self.save_paths) >= self.cfg.model.max_save:
                    os.remove(self.save_paths.pop(0))
                utils.model_save(self.model, path)
                self.save_paths.append(path)
                
def load_train_objs(rank, cfg: DictConfig):
    work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    
    log = utils.get_logger(f"Rank {rank}")
    tblog = SummaryWriter(f"{work_dir}/runs")
    
    net = getattr(model, cfg.model.net)(**cfg.model.params)
    
    log.info(f"Network parameters: {sum([p.numel() for p in net.parameters()])}")
    
    if cfg.model.checkpoint is None:
            log.info("Start a new model")
    else:
        missing = utils.model_load(net, cfg.model.checkpoint, True)
        log.info(f"Load parameters from {cfg.model.checkpoint}")
        log.info(f"Missing keys: {missing}")
            
    TrainDataset = dataset.AFMGenDataset(cfg.data.train_path, transform=None)
    TestDataset = dataset.AFMGenDataset(cfg.data.test_path, transform=None)
    
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
                                             sampler=DistributedSampler(test_data, shuffle=False),
                                             )
    
    return TrainLoader, TestLoader


@hydra.main(config_path="conf", config_name="4a8a", version_base=None) # hydra will automatically relocate the working dir.
def main(cfg):
    rank = int(os.environ["LOCAL_RANK"])
    world_size = torch.cuda.device_count()
    ddp_setup(rank, world_size)
    model, TrainDataset, TestDataset, Optimizer, Schedular, log, tblog = load_train_objs(rank, cfg)
    TrainLoader, TestLoader = prepare_dataloader(TrainDataset, TestDataset, cfg)
    trainer = Trainer(hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir'],
                      cfg,
                      model,
                      TrainLoader,
                      TestLoader,
                      Optimizer,
                      Schedular,
                      log,
                      tblog,
                      rank,
                      )
    trainer.fit()
    destroy_process_group()

if __name__ == "__main__":
    main()
    
