# torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py

import os
import re
import hydra
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf, DictConfig
from torch.distributed import get_rank
from dataset.sampler import z_sampler
from functools import partial

import dataset
import model
import utils

user = os.environ.get('USER') == "supercgor"
config_name = "unetv3_local" if user else "unetv3_wm"

class Trainer():
    def __init__(self, rank, cfg, model, TestLoader, log, tblog):
        self.work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
        self.cfg = cfg
        try:
            self.rank = get_rank()
        except RuntimeError:
            self.rank = 0
            
        self.gpu_id = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.gpu_id)
        self.TestLoader = TestLoader
        self.Analyser = utils.parallelAnalyser(cfg.dataset.real_size, split = self.cfg.dataset.split, threshold = 0.8).to(self.gpu_id)
        
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
        
    def fit(self, test  = False):
        epoch_start_time = time.time()
        if test:
            loss, cms, rot = self.test_one_epoch(0, log_every = self.cfg.setting.log_every)
            
            logstr = f"\n============= Summary Test | Epoch {0:2d} ================\nloss: {loss:.2e} | used time: {(time.time() - epoch_start_time)/60:4.1f} mins | {'Model saved' if (loss < self.best) and (self.rank == 0) else 'Model not saved'}"
            logstr += f"\n=================   Element - 'H2O'   ================="
            logstr += f"\n(Overall)  AP: {cms.AP[0].mean():.2f} | AR: {cms.AR[0].mean():.2f} | ACC: {cms.ACC[0].mean():.2f} | SUC: {cms.SUC[0].mean():.2f} | Mmean: {rot:.2e}"
            for i, (low, high) in enumerate(zip(self.cfg.dataset.split[:-1], self.cfg.dataset.split[1:])):
                logstr += f"\n({low:.1f}-{high:.1f}A) AP: {cms.AP[0,i]:.2f} | AR: {cms.AR[0,i]:.2f} | ACC: {cms.ACC[0,i]:.2f} | SUC: {cms.SUC[0,i]:.2f}\nTP: {cms.TP[0,i]:10.0f} | FP: {cms.FP[0,i]:10.0f} | FN: {cms.FN[0,i]:10.0f}"
            self.log.info(logstr)
        else:
            self.pred_one_epoch(0, log_every = self.cfg.setting.log_every)
        
    
    @torch.no_grad()
    def test_one_epoch(self, epoch, log_every: int = 100) -> tuple[torch.Tensor]:
        self.model.eval()
        self.LostStat.reset()
        self.GradStat.reset()
        self.RotStat.reset()
        self.ConfusionCounter.reset()
        for i, (filenames, inps, embs, targs, labels) in enumerate(self.TestLoader):
            print(*filenames, end="\r")
            inps = inps.to(self.gpu_id, non_blocking = True)
            embs = embs.to(self.gpu_id, non_blocking = True)
            targs = targs.to(self.gpu_id, non_blocking = True)
            preds = self.model(inps, embs)
            loss= self.Criterion(preds, targs)
            
            self.LostStat.add(loss)
    
            CMS = self.Analyser(preds, targs)
            self.ConfusionCounter.add(CMS[...,:3])
            self.RotStat.add(CMS[...,3].mean(dim=[1,2]))
            
            if True:
                for filename, pred, targs in zip(filenames, preds, targs):
                    _, pos, rot = utils.library.box2orgvec(pred, utils.library.inverse_sigmoid(0.5), 2.0, self.cfg.dataset.real_size, sort = True, nms = True)
                    rot = rot.reshape(-1, 9)[:,:6]
                    pos = np.concatenate([pos, rot], axis = -1) # N, 9
                    pos = utils.library.encodeWater(pos)
                    utils.xyz.write(f"{self.work_dir}/{filename}_pred.xyz", np.tile(np.array(["O", "H", "H"]),(pos.shape[0],1)), pos.reshape(-1, 3, 3))
                    O_pos, H_pos = torch.split_with_sizes(pos, [3, 6], dim = 1)
                    O_pos = O_pos.reshape(-1, 3) / torch.tensor([25.0, 25.0, 3.0])
                    H_pos = H_pos.reshape(-1, 3) / torch.tensor([25.0, 25.0, 3.0])
                    utils.poscar.save(f"{self.work_dir}/{filename}_pred.poscar", [25.0, 25.0, 3.0], ["O", "H"], [len(O_pos), len(H_pos)], torch.cat([O_pos, H_pos], dim = 0), ZXYformat=False)
                    _, pos, rot = utils.library.box2orgvec(targs, 0.5, 2.0, self.cfg.dataset.real_size, sort = False, nms = False)
                    rot = rot.reshape(-1, 9)[:,:6]
                    pos = np.concatenate([pos, rot], axis = -1) # N, 9
                    pos = utils.library.encodeWater(pos)
                    utils.xyz.write(f"{self.work_dir}/{filename}_targ.xyz", np.tile(np.array(["O", "H", "H"]),(pos.shape[0],1)), pos.reshape(-1, 3, 3))
                    
            if self.rank == 0 and i % log_every == 0:
                self.log.info(f"Epoch {epoch:2d} | Iter {i:5d}/{len(self.TestLoader):5d} | loss {loss:.2e}")
                self.tblog.add_scalar("Train/Loss", loss, epoch * len(self.TestLoader) + i)
                
        loss = self.LostStat.calc()
        cms = self.ConfusionCounter.calc()
        rot = self.RotStat.calc()
        
        return loss, cms, rot
    
    @torch.no_grad()
    def pred_one_epoch(self, epoch, log_every: int = 100) -> tuple[torch.Tensor]:
        self.model.eval()
        for i, (filenames, inps) in enumerate(self.TestLoader):
            print(*filenames, end="\r")
            inps = inps.to(self.gpu_id, non_blocking = True)
            preds = self.model(inps, None)
            if True:
                for filename, pred in zip(filenames, preds):
                    _, pos, rot = utils.library.box2orgvec(pred, utils.library.inverse_sigmoid(0.5), 2.0, self.cfg.dataset.real_size, sort = True, nms = True)
                    rot = rot.reshape(-1, 9)[:,:6]
                    pos = np.concatenate([pos, rot], axis = -1) # N, 9
                    pos = utils.library.encodeWater(pos)
                    utils.xyz.write(f"{self.work_dir}/{filename}_pred.xyz", np.tile(np.array(["O", "H", "H"]),(pos.shape[0],1)), pos.reshape(-1, 3, 3))
                    O_pos, H_pos = pos[...,:3], pos[...,3:]
                    O_pos = O_pos.reshape(-1, 3) / [25.0, 25.0, 3.0]
                    H_pos = H_pos.reshape(-1, 3) / [25.0, 25.0, 3.0]
                    utils.poscar.save(f"{self.work_dir}/{filename}_pred.poscar", [25.0, 25.0, 3.0], ["O", "H"], [len(O_pos), len(H_pos)], np.concatenate([O_pos, H_pos], axis = 0), ZXYformat=False)

        return
    
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
    
    net.apply_transform(inp_transform, out_transform)
    
    log.info(f"Network parameters: {sum([p.numel() for p in net.parameters()])}")
    
    if cfg.model.checkpoint is None:
        if input("Continue? (y/n): ").lower().strip() not in ["y", "yes", "true", "1", "t"]:
            exit()
    else:
        loaded_keys = utils.model_load(net, cfg.model.checkpoint, True)
        log.info(f"Load parameters from {cfg.model.checkpoint}")
    
    #transform = torch.nn.Sequential(dataset.PixelShift(), dataset.Cutout(),  dataset.ColorJitter(),  dataset.Noisy(),  dataset.Blur())
    transform = []
        
    TestDataset = dataset.AFMDataset_V2(cfg.dataset.test_path, useLabel=False, useEmb=False, useZ=cfg.dataset.image_size[0], transform=transform, key_filter= key_filter, sampler=partial(z_sampler, is_rand=False))

    return net, TestDataset, log, tblog

def prepare_dataloader(test_data, cfg: DictConfig):
    TestLoader = torch.utils.data.DataLoader(test_data,
                                             batch_size=cfg.setting.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.setting.num_workers,
                                             pin_memory=cfg.setting.pin_memory,
                                             )
    return TestLoader

def key_filter(key):
    #return True
    return True if "HDA" in key or "ss" in key else False
    #return re.match(r"T\d{1,3}_\d{1,5}", key) is not None

@hydra.main(config_path="config", config_name=config_name, version_base=None) # hydra will automatically relocate the working dir.
def main(cfg):
    rank = "cuda" if torch.cuda.is_available() else "cpu"
    model, TestDataset, log, tblog = load_train_objs(rank, cfg)
    TestLoader = prepare_dataloader(TestDataset, cfg)
    trainer = Trainer(rank, cfg, model, TestLoader, log, tblog)
    trainer.fit()

if __name__ == "__main__":
    main()
    
