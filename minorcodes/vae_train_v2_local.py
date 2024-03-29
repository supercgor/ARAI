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

class Trainer():
    def __init__(self, rank, cfg, model, TrainLoader, TestLoader, TrainDataset, TestDataset, Optimizer, Schedular, log, tblog):
        self.work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
        self.cfg = cfg
        self.rank = rank
        self.device =  torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device(f"cpu")    
        self.model = model.to(self.device)
        self.Analyser = utils.parallelAnalyser(cfg.dataset.real_size, split = self.cfg.dataset.split).to(self.device)
        self.Criterion = utils.conditionVAELoss(wc = cfg.criterion.cond_weight,
                                                wpos_weight = cfg.criterion.pos_weight,
                                                wpos = cfg.criterion.xyz_weight,
                                                wr = cfg.criterion.rot_weight,
                                                wvae = cfg.criterion.vae_weight
                                                ).to(self.device)
        self.TrainDataset = TrainDataset
        self.TestDataset = TestDataset
        self.TrainLoader = TrainLoader
        self.TestLoader = TestLoader
        self.Optimizer = Optimizer
        self.Schedular = Schedular
        
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
            loss, grad, cms, rot = self.train_one_epoch(epoch, log_every = self.cfg.setting.log_every)
            logstr = f"\n============== Summary Train | Epoch {epoch:2d} ==============\nloss: {loss[0]:.2e} | grad: {grad:.2e} | used time: {(time.time() - epoch_start_time)/60:4.1f} mins"
            logstr += f"\nLoss | Confidence {loss[1]:.2e} | Position {loss[2]:.2e} | Rotation {loss[3]:.2e} | VAE {loss[4]:.2e}"
            logstr += f"\n=================   Element - 'H2O'   ================="
            logstr += f"\n(Overall)  AP: {cms.AP[0].mean():.2f} | AR: {cms.AP[0].mean():.2f} | ACC: {cms.ACC[0].mean():.2f} | SUC: {cms.SUC[0].mean():.2f} | Mmean: {rot:.2e}"
            for i, (low, high) in enumerate(zip(self.cfg.dataset.split[:-1], self.cfg.dataset.split[1:])):
                logstr += f"\n({low:.1f}-{high:.1f}A) AP: {cms.AP[0,i]:.2f} | AR: {cms.AR[0,i]:.2f} | ACC: {cms.ACC[0,i]:.2f} | SUC: {cms.SUC[0,i]:.2f}\nTP: {cms.TP[0,i]:10.0f} | FP: {cms.FP[0,i]:10.0f} | FN: {cms.FN[0,i]:10.0f}"
            self.log.info(logstr)
            
            loss, cms, rot = self.test_one_epoch(epoch, log_every = self.cfg.setting.log_every)
            
            logstr = f"\n============= Summary Test | Epoch {epoch:2d} ================\nloss: {loss[0]:.2e} | used time: {(time.time() - epoch_start_time)/60:4.1f} mins | {'Model saved' if (loss[0] < self.best) and (self.rank == 0) else 'Model not saved'}"
            logstr += f"\n=================   Element - 'H2O'   ================="
            logstr += f"\n(Overall)  AP: {cms.AP[0].mean():.2f} | AR: {cms.AP[0].mean():.2f} | ACC: {cms.ACC[0].mean():.2f} | SUC: {cms.SUC[0].mean():.2f} | Mmean: {rot:.2e}"
            for i, (low, high) in enumerate(zip(self.cfg.dataset.split[:-1], self.cfg.dataset.split[1:])):
                logstr += f"\n({low:.1f}-{high:.1f}A) AP: {cms.AP[0,i]:.2f} | AR: {cms.AR[0,i]:.2f} | ACC: {cms.ACC[0,i]:.2f} | SUC: {cms.SUC[0,i]:.2f}\nTP: {cms.TP[0,i]:10.0f} | FP: {cms.FP[0,i]:10.0f} | FN: {cms.FN[0,i]:10.0f}"
                
            self.save_model(epoch, loss[0])
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
        self.ConfusionCounter.reset()
        for i, (filenames, inps, targs, embs) in enumerate(self.TrainLoader):
            inps = inps.to(self.device, non_blocking = True)
            targs = targs.to(self.device, non_blocking = True)
            embs = embs.to(self.device, non_blocking = True)
            preds, mu, logvar = self.model(inps, embs)
            loss_conf, loss_pos, loss_r, loss_vae = self.Criterion(preds, targs, mu, logvar)
            
            predsp = torch.cat([preds[...,(0,)].sigmoid(), preds[...,1:]], dim=-1)
            predsp = torch.where(predsp[...,(0,)] < 0.5, torch.zeros_like(predsp), predsp)
            
            mup, logvarp = self.model.forward(predsp, None, encoder = True)
            vae_add = self.Criterion.wvae * self.Criterion.vaeloss(mup, logvarp)
            loss_vae = (loss_vae + vae_add)/2
            
            loss = loss_conf + loss_pos + loss_r + loss_vae
            self.Optimizer.zero_grad()
            
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.criterion.clip_grad, error_if_nonfinite=False)
            
            self.Optimizer.step()
            
            self.LostStat.add(loss)
            self.LostConfidenceStat.add(loss_conf)
            self.LostPositionStat.add(loss_pos)
            self.LostRotationStat.add(loss_r)
            self.LostVAEStat.add(loss_vae)
            self.GradStat.add(grad)
    
            CMS = self.Analyser(preds, targs)
            self.ConfusionCounter.add(CMS[...,:3])
            self.RotStat.add(CMS[...,3].mean(dim=[1,2]))
            
            if self.rank == 0 and i % log_every == 0:
                # conf, pos, r = utils.functional.box2orgvec(targs[0].detach().cpu(), 0.0, 2.0, (25.0, 25.0, 8.0), False, False)
                # targ = utils.functional.makewater(pos, r)
                # utils.xyz.write(f"{self.work_dir}/{filenames[0]}_targ.xyz", np.array([["O", "H", "H"]], dtype=np.str_).repeat(len(targ), axis=0), targ)
                # conf, pos, r = utils.functional.box2orgvec(inps[0].detach().cpu(), 0.0, 2.0, (25.0, 25.0, 4.0), False, False)
                # inp = utils.functional.makewater(pos, r)
                # utils.xyz.write(f"{self.work_dir}/{filenames[0]}_inp.xyz", np.array([["O", "H", "H"]], dtype=np.str_).repeat(len(inp), axis=0), inp)
                self.log.info(f"Epoch {epoch:2d} | Iter {i:5d}/{len(self.TrainLoader):5d} | loss {loss:.2e} | grad {grad:.2e} | conf {loss_conf:.2e} | pos {loss_pos:.2e} | rot {loss_r:.2e} | vae {loss_vae:.2e}")
                self.tblog.add_scalar("Train/Loss", loss, epoch * len(self.TrainLoader) + i)
                self.tblog.add_scalar("Train/LossConf", loss_conf, epoch * len(self.TrainLoader) + i)
                self.tblog.add_scalar("Train/LossPos", loss_pos, epoch * len(self.TrainLoader) + i)
                self.tblog.add_scalar("Train/LossRot", loss_r, epoch * len(self.TrainLoader) + i)
                self.tblog.add_scalar("Train/LossVAE", loss_vae, epoch * len(self.TrainLoader) + i)
                self.tblog.add_scalar("Train/Grad", grad, epoch * len(self.TrainLoader) + i)
                
        self.Schedular.step()
        losses = [self.LostStat.calc(),self.LostConfidenceStat.calc(), self.LostPositionStat.calc(), self.LostRotationStat.calc(), self.LostVAEStat.calc()]
        grad = self.GradStat.calc()
        cms = self.ConfusionCounter.calc()
        rot = self.RotStat.calc()
        
        return losses, grad, cms, rot
    
    @torch.no_grad()
    def test_one_epoch(self, epoch, log_every: int = 25) -> tuple[torch.Tensor]:
        self.model.eval()
        self.LostStat.reset()
        self.LostConfidenceStat.reset()
        self.LostPositionStat.reset()
        self.LostRotationStat.reset()
        self.LostVAEStat.reset()
        self.ConfusionCounter.reset()
        self.RotStat.reset()
        for i, (filenames, inps, targs, embs) in enumerate(self.TestLoader):
            inps = inps.to(self.device, non_blocking = True)
            targs = targs.to(self.device, non_blocking = True)
            embs = embs.to(self.device, non_blocking = True)
            preds, mu, logvar = self.model(inps, embs)
            loss_conf, loss_pos, loss_r, loss_vae = self.Criterion(preds, targs, mu, logvar)
            
            predsp = torch.cat([preds[...,(0,)].sigmoid(), preds[...,1:]], dim=-1)
            predsp = torch.where(predsp[...,(0,)] < 0.5, torch.zeros_like(predsp), predsp)
                        
            loss = loss_conf + loss_pos + loss_r + loss_vae
                        
            self.LostStat.add(loss)
            self.LostConfidenceStat.add(loss_conf)
            self.LostPositionStat.add(loss_pos)
            self.LostRotationStat.add(loss_r)
            self.LostVAEStat.add(loss_vae)
            
            CMS = self.Analyser(preds, targs)
            self.ConfusionCounter.add(CMS[...,:3])
            self.RotStat.add(CMS[...,3])
            if self.rank == 0 and i % log_every == 0:
                self.log.info(f"Epoch {epoch:2d} | Iter {i:5d}/{len(self.TestLoader):5d} | loss {loss:.2e}")
                self.tblog.add_scalar("Test/Loss", loss, epoch * len(self.TestLoader) + i)
                self.tblog.add_scalar("Test/LossWC", loss_conf, epoch * len(self.TestLoader) + i)
                self.tblog.add_scalar("Test/LossPos", loss_pos, epoch * len(self.TestLoader) + i)
                self.tblog.add_scalar("Test/LossRot", loss_r, epoch * len(self.TestLoader) + i)
                self.tblog.add_scalar("Test/LossVAE", loss_vae, epoch * len(self.TestLoader) + i)
                
        losses = [self.LostStat.calc(),self.LostConfidenceStat.calc(), self.LostPositionStat.calc(), self.LostRotationStat.calc(), self.LostVAEStat.calc()]
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
    
    net.input_transform = inp_transform
    net.output_transform = out_transform
    
    log.info(f"Network parameters: {sum([p.numel() for p in net.parameters()])}")
    
    if cfg.model.checkpoint is None:
            log.info("Start a new model")
    else:
        loaded = utils.model_load(net, cfg.model.checkpoint, True)
        log.info(f"Load parameters from {cfg.model.checkpoint}")
    
    size = cfg.model.params.in_size
    size = [size[1],size[2],size[0]]
    TrainDataset = dataset.ZVarAFM(cfg.dataset.train_path, box_size = size)
    TestDataset = dataset.ZVarAFM(cfg.dataset.test_path, box_size = size)
    
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
    TrainLoader = torch.utils.data.DataLoader(train_data, 
                                              batch_size=cfg.setting.batch_size, 
                                              shuffle=True, 
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


@hydra.main(config_path="config", config_name="vae44_local", version_base=None) # hydra will automatically relocate the working dir.
def main(cfg):
    rank = 0
    model, TrainDataset, TestDataset, Optimizer, Schedular, log, tblog = load_train_objs(rank, cfg)
    TrainLoader, TestLoader = prepare_dataloader(TrainDataset, TestDataset, cfg)
    trainer = Trainer(rank, cfg, model, TrainLoader, TestLoader, TrainDataset, TestDataset, Optimizer, Schedular, log, tblog)
    trainer.fit()

if __name__ == "__main__":
    main()
    
