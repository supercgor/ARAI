import os
import hydra
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig 

import dataset
import model
import utils

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
    
class Trainer():
    def __init__(self, rank, cfg, model, Testdata, TestLoader, log, tblog):
        self.work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
        self.cfg = cfg
        self.rank = rank
        self.device =  torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device(f"cpu")
        self.model = model.to(self.device)
        self.Analyser = utils.parallelAnalyser(cfg.dataset.real_size).to(self.device)
        self.Criterion = utils.conditionVAELoss(wc = cfg.criterion.cond_weight,
                                                wpos_weight = cfg.criterion.pos_weight,
                                                wpos = cfg.criterion.xyz_weight,
                                                wr = cfg.criterion.rot_weight,
                                                wvae = cfg.criterion.vae_weight
                                                ).to(self.device)
        
        self.TestData = Testdata
        self.TestLoader = TestLoader
        
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
        epoch_start_time = time.time()
        
        if False:
            self.test_one_epoch(0)
        else:
            self.pred2_one_epoch(0)
       
    @torch.no_grad()
    def test_one_epoch(self, epoch, log_every: int = 25) -> tuple[torch.Tensor]:
        all_data = pd.DataFrame()
        self.model.eval()
        self.LostStat.reset()
        self.LostConfidenceStat.reset()
        self.LostPositionStat.reset()
        self.LostRotationStat.reset()
        self.LostVAEStat.reset()
        self.ConfusionCounter.reset()
        self.RotStat.reset()
        for i, (filenames, inps, targs, embs) in enumerate(self.TestLoader):
            inps = inps.to(self.device)
            targs = targs.to(self.device)
            embs = embs.to(self.device)
            preds, mu, logvar = self.model(inps, embs)
            loss_wc, loss_pos, loss_r, loss_vae = self.Criterion(preds, targs, mu, logvar)
            loss = loss_wc + loss_pos + loss_r + loss_vae
            self.LostStat.add(loss)
            self.LostConfidenceStat.add(loss_wc)
            self.LostPositionStat.add(loss_pos)
            self.LostRotationStat.add(loss_r)
            self.LostVAEStat.add(loss_vae)
            feats = torch.cat([mu, logvar], dim=-1).flatten(1).detach().cpu().numpy()
            names = []
            for i in range(len(filenames)):
                fn = filenames[i]
                fn = fn.split("_")
                name = ""
                if "d" in fn:
                    name += f"d,{fn[3]},{fn[-1]}"
                else:
                    name += f"h,{fn[2]},{fn[-1]}"
                names.append(name)
            print(names)
            df_batch_predictions = pd.DataFrame(feats, index=names)
            
            all_data = pd.concat([all_data, df_batch_predictions])
            
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
            
        all_data.to_csv('all_output.csv')
        losses = [self.LostStat.calc(),self.LostConfidenceStat.calc(), self.LostPositionStat.calc(), self.LostRotationStat.calc(), self.LostVAEStat.calc()]
        cms = self.ConfusionCounter.calc()
        rot = self.RotStat.calc()
        
        return losses, cms, rot
        
    @torch.no_grad()
    def pred_one_epoch(self, epoch, log_every: int = 25) -> tuple[torch.Tensor]:
        self.model.eval()
        for i, (filenames, inps, targs, embs) in enumerate(self.TestLoader):
            inps = inps.to(self.device)
            embs = embs.to(self.device)
            
            out = [inps]
            preds = inps
            for j in range(self.cfg.pred_loop):
                preds, mu, logvar = self.model(preds, embs)
                preds = torch.stack([utils.library.box2box(pred, real_size=(25.0, 25.0, 4.0), threshold=0.0, nms=True, sort=True, cutoff=2.0) for pred in preds], dim = 0)
                #out.insert(0, preds)
                out.append(preds)
                            
            preds = torch.cat(out, dim=3)# B X Y Z*L 10
            
            for b in range(preds.shape[0]):
                pred = preds[b]
                filename = filenames[b]
                pred = pred.detach().cpu()
                conf, pos, r = utils.library.box2orgvec(pred, 0.0, 2.0, (25.0, 25.0, 4.0 * (self.cfg.pred_loop+1)), True, True)
                pred = utils.library.encodeWater(np.concatenate([pos, r], axis = -1))
                # pred = utils.library.makewater(pos, r)
                utils.xyz.write(f"{self.work_dir}/{filename}.xyz", np.array([["O", "H", "H"]], dtype=np.str_).repeat(len(pred), axis=0), pred)
                # conf, pos, r = utils.functional.box2orgvec(targs[b].detach().cpu(), 0.0, 2.0, (25.0, 25.0, 8.0), False, False)
                # targ = utils.functional.makewater(pos, r)
                # utils.xyz.write(f"{self.work_dir}/{filename}_targ.xyz", np.array([["O", "H", "H"]], dtype=np.str_).repeat(len(targ), axis=0), targ)
                # conf, pos, r = utils.functional.box2orgvec(inps[b].detach().cpu(), 0.0, 2.0, (25.0, 25.0, 4.0), False, False)
                # inp = utils.functional.makewater(pos, r)
                # utils.xyz.write(f"{self.work_dir}/{filename}_inp.xyz", np.array([["O", "H", "H"]], dtype=np.str_).repeat(len(inp), axis=0), inp)
                
            if self.rank == 0 and i % log_every == 0:
                self.log.info(f"Epoch {epoch:2d} | Iter {i:5d}/{len(self.TestLoader):5d}")
            
    @torch.no_grad()
    def pred2_one_epoch(self, epoch, log_every: int = 25) -> tuple[torch.Tensor]:
        self.model.eval()
        for i in range(len(self.TestData)):
            preds = []
            for j, zinp in enumerate(range(2, 16, 4)):
                filename, inp, _, emb = self.TestData.get(i, zinp)
                if j == 0:
                    preds.append(inp)
                else:
                    inp = pred
                inp = inp.to(self.device)
                emb = emb.to(self.device)
                pred, mu, nu = self.model(inp[None,...], emb[None,...])
                pred = utils.library.box2box(pred[0], real_size=(25.0, 25.0, 4.0), threshold=0.0, nms=True, sort=True, cutoff=2.0)
                preds.append(pred)
                            
            preds = torch.cat(preds, dim=-2)# B X Y Z*L 10

            preds = preds.detach().cpu()
            conf, pos, r = utils.library.box2orgvec(preds, 0.0, 2.0, (25.0, 25.0, 4.0 * (j + 2)), True, True)
            r = r.view(-1, 9)[:, :6]
            pred = utils.library.encodeWater(np.concatenate([pos, r], axis = -1)).reshape(-1, 3, 3)
            utils.xyz.write(f"{self.work_dir}/{filename}.xyz", np.array([["O", "H", "H"]], dtype=np.str_).repeat(len(pred), axis=0), pred)
            
            if self.rank == 0 and i % log_every == 0:
                self.log.info(f"Epoch {epoch:2d} | Iter {i:5d}/{len(self.TestLoader):5d}")
            
                
def load_train_objs(rank, cfg: DictConfig):
    work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    
    log = utils.get_logger(f"Rank {rank}")
    tblog = SummaryWriter(f"{work_dir}/runs")
    
    net = getattr(model, cfg.model.net)(**cfg.model.params)
    
    net.input_transform = inp_transform
    net.output_transform = out_transform
    
    log.info(f"Network parameters: {sum([p.numel() for p in net.parameters()])}")
    
    if cfg.model.checkpoint is None:
        raise ValueError("No checkpoint is provided.")
    else:
        loaded = utils.model_load(net, cfg.model.checkpoint, True)
        log.info(f"Load parameters from {cfg.model.checkpoint}")
            
    #TestDataset = dataset.AFMGenDataset(cfg.dataset.test_path, transform=None)
    TestDataset = dataset.ZVarAFM(cfg.dataset.test_path, (25, 25, 4))
    
    return net, TestDataset, log, tblog

def prepare_dataloader(test_data, cfg: DictConfig): 
    TestLoader = torch.utils.data.DataLoader(test_data,
                                             batch_size=cfg.setting.batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=cfg.setting.pin_memory,
                                             )
    
    return TestLoader


@hydra.main(config_path="config", config_name="vae44_local", version_base=None) # hydra will automatically relocate the working dir.
def main(cfg):
    rank = 0
    model, TestDataset, log, tblog = load_train_objs(rank, cfg)
    TestLoader = prepare_dataloader(TestDataset, cfg)
    trainer = Trainer(rank, cfg, model, TestDataset, TestLoader, log, tblog)
    trainer.fit()

if __name__ == "__main__":
    main()
    
