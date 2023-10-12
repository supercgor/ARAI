import os
import hydra
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig

import dataset
import model
import utils

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
                ) -> None:
        self.work_dir = work_dir
        self.cfg = cfg
        self.rank = 0
        self.model = model
        self.save_paths = []
        self.TrainLoader = TrainLoader
        self.TestLoader = TestLoader
        self.Optimizer = Optimizer
        self.Schedular = Schedular
        self.log = log
        self.tblog = tblog
        self.Analyser = utils.MolecularAnalyser(cfg.data.real_size, cfg.data.nms)
        self.ConfusionMatrixCounter = utils.ConfusionMatrixCounter()
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
                                                )
        
        self.best = np.inf
        
    def fit(self):
        for epoch in range(self.cfg.setting.epoch):
            self.log.info(f"Start training epoch: {epoch}...")
            epoch_start_time = time.time()
            train_loss, train_grad, train_confusion = self.train_one_epoch(epoch)
            logstr = f"\n============== Summary Train | Epoch {epoch:2d} ==============\nloss: {train_loss:.2e} | grad: {train_grad:.2e} | used time: {(time.time() - epoch_start_time)/60:4.1f} mins"
            
            logstr += f"\n=================   Element - '{e}'    =================\n(Overall)  AP: {train_confusion[i,:,4].mean():.2f} | AR: {train_confusion[i,:,3].mean():.2f} | ACC: {train_confusion[i,:,5].mean():.2f} | SUC: {train_confusion[i,:,6].mean():.2f}"
            for j, (low, high) in enumerate(zip(self.cfg.data.split[:-1], self.cfg.data.split[1:])):
                logstr += f"\n({low:.1f}-{high:.1f}A) AP: {train_confusion[i,j,4]:.2f} | AR: {train_confusion[i,j,3]:.2f} | ACC: {train_confusion[i,j,5]:.2f} | SUC: {train_confusion[i,j,6]:.2f}\nTP: {train_confusion[i,j,0]:10.0f} | FP: {train_confusion[i,j,1]:10.0f} | FN: {train_confusion[i,j,2]:10.0f}"
            self.log.info(logstr)
            
            test_loss, test_confusion = self.test_one_epoch(epoch)
            
            logstr = f"\n============= Summary Test | Epoch {epoch:2d} ================\nloss: {test_loss:.2e} | used time: {(time.time() - epoch_start_time)/60:4.1f} mins | {'Model saved' if (test_loss < self.best) and (self.rank == 0) else 'Model not saved'}"
            
            for i, e in enumerate(utils.const.ion_order.split()):
                if e in self.cfg.data.ion_type:
                    logstr += f"\n=================   Element - '{e}'    =================\n(Overall)  AP: {test_confusion[i,:,4].mean():.2f} | AR: {test_confusion[i,:,3].mean():.2f} | ACC: {test_confusion[i,:,5].mean():.2f} | SUC: {test_confusion[i,:,6].mean():.2f}"
                    for j, (low, high) in enumerate(zip(self.cfg.data.split[:-1], self.cfg.data.split[1:])):
                        logstr += f"\n({low:.1f}-{high:.1f}A) AP: {test_confusion[i,j,4]:.2f} | AR: {test_confusion[i,j,3]:.2f} | ACC: {test_confusion[i,j,5]:.2f} | SUC: {test_confusion[i,j,6]:.2f}\nTP: {test_confusion[i,j,0]:10.0f} | FP: {test_confusion[i,j,1]:10.0f} | FN: {test_confusion[i,j,2]:10.0f}"
            
            self.save_model(epoch, test_loss)
            self.log.info(logstr)
        
    def train_one_epoch(self, epoch, log_every: int = 25) -> tuple[torch.Tensor]:
        self.model.train()
        self.LostStat.reset()
        self.LostConfidenceStat.reset()
        self.LostPositionStat.reset()
        self.LostRotationStat.reset()
        self.LostVAEStat.reset()
        self.GradStat.reset()
        self.RotStat.reset()
        self.ConfusionMatrixCounter.reset()
        for i, (filename, box4a, box8a, temp) in enumerate(self.TrainLoader):
            out, mu, var = self.model(box4a, temp)
            out = out_transform(out)
            loss_wc, loss_pos, loss_r, loss_vae = self.Criterion(out, box8a, mu, var)
            loss = loss_wc + loss_pos + loss_r + loss_vae
            self.Optimizer.zero_grad()
            loss.backward()
            self.Optimizer.step()
            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.criterion.clip_grad, error_if_nonfinite=False)
            self.LostStat.add(loss)
            self.LostConfidenceStat.add(loss_wc)
            self.LostPositionStat.add(loss_pos)
            self.LostRotationStat.add(loss_r)
            self.LostVAEStat.add(loss_vae)
            self.GradStat.add(grad)
            confusion_matrix, MS = self.Analyser(out, box8a)
            self.ConfusionMatrixCounter(confusion_matrix)
            self.RotStat.add(MS)
            if self.rank == 0 and i % log_every == 0:
                self.log.info(f"Epoch {epoch:2d} | Iter {i:5d}/{len(self.TrainLoader):5d} | Loss {loss:.2e} | Grad {grad:.2e}")
                self.tblog.add_scalar("Train/Loss", loss, epoch * len(self.TrainLoader) + i)
                self.tblog.add_scalar("Train/LossWC", loss_wc, epoch * len(self.TrainLoader) + i)
                self.tblog.add_scalar("Train/LossPos", loss_pos, epoch * len(self.TrainLoader) + i)
                self.tblog.add_scalar("Train/LossRot", loss_r, epoch * len(self.TrainLoader) + i)
                self.tblog.add_scalar("Train/LossVAE", loss_vae, epoch * len(self.TrainLoader) + i)
                self.tblog.add_scalar("Train/Grad", grad, epoch * len(self.TrainLoader) + i)
                self.tblog.add_scalar("Train/LR", self.Schedular.get_last_lr()[0], epoch * len(self.TrainLoader) + i)
                
                self.tblog.add_scalars(f"Train/AP {0:.1f}-{4:.1f}A", {"H2O": self.ConfusionMatrixCounter.AP[0,0]})
                self.tblog.add_scalars(f"Train/AR {0:.1f}-{4:.1f}A", {"H2O": self.ConfusionMatrixCounter.AR[0,0]})
                self.tblog.add_scalars(f"Train/ACC {0:.1f}-{4:.1f}A", {"H2O": self.ConfusionMatrixCounter.ACC[0,0]})
                self.tblog.add_scalars(f"Train/SUC {0:.1f}-{4:.1f}A", {"H2O": self.ConfusionMatrixCounter.SUC[0,0]})
                
        self.Schedular.step()
        return [self.LostStat.calc(),self.LostConfidenceStat.calc(), self.LostPositionStat.calc(), self.LostRotationStat.calc(), self.LostVAEStat.calc()], self.GradStat.calc(), self.ConfusionMatrixCounter.calc()
    
    @torch.no_grad()
    def test_one_epoch(self, epoch, log_every: int = 25) -> tuple[torch.Tensor]:
        self.model.eval()
        self.LostStat.reset()
        self.LostConfidenceStat.reset()
        self.LostPositionStat.reset()
        self.LostRotationStat.reset()
        self.LostVAEStat.reset()
        self.ConfusionMatrixCounter.reset()
        for i, (filename, box4a, box8a, temp) in enumerate(self.TestLoader):
            out, mu, var = self.model(box4a, temp)
            out = out_transform(out)
            loss_wc, loss_pos, loss_r, loss_vae = self.Criterion(out, box8a, mu, var)
            loss = loss_wc + loss_pos + loss_r + loss_vae
            confusion_matrix = self.Analyser(out, box8a)
            self.LostStat.add(loss)
            self.LostConfidenceStat.add(loss_wc)
            self.LostPositionStat.add(loss_pos)
            self.LostRotationStat.add(loss_r)
            self.LostVAEStat.add(loss_vae)
            self.ConfusionMatrixCounter(confusion_matrix)
            if self.rank == 0 and i % log_every == 0:
                self.log.info(f"Epoch {epoch:2d} | Iter {i:5d}/{len(self.TestLoader):5d} | Loss {loss:.2e}")
                self.tblog.add_scalar("Test/Loss", loss, epoch * len(self.TestLoader) + i)
                self.tblog.add_scalar("Test/LossWC", loss_wc, epoch * len(self.TestLoader) + i)
                self.tblog.add_scalar("Test/LossPos", loss_pos, epoch * len(self.TestLoader) + i)
                self.tblog.add_scalar("Test/LossRot", loss_r, epoch * len(self.TestLoader) + i)
                self.tblog.add_scalar("Test/LossVAE", loss_vae, epoch * len(self.TestLoader) + i)
                
        return self.LostStat.calc(), self.ConfusionMatrixCounter.calc()
    
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
    TestDataset = dataset.AFMDataset(cfg.data.test_path, transform=None)
    
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
                                              )
        
    TestLoader = torch.utils.data.DataLoader(test_data,
                                             batch_size=cfg.setting.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.setting.num_workers,
                                             pin_memory=cfg.setting.pin_memory,
                                             )
    
    return TrainLoader, TestLoader


@hydra.main(config_path="conf", config_name="config", version_base=None) # hydra will automatically relocate the working dir.
def main(cfg):
    rank = 0
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

if __name__ == "__main__":
    main()
    
