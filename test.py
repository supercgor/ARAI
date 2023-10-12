# torchrun --standalone --nnodes=1 --nproc-per-node=2 test.py

import os
import hydra
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf, DictConfig

import dataset
import model
import utils
from utils import poscar
    
class Trainer():
    def __init__(self, 
                 work_dir: str,
                 cfg: DictConfig,
                 model: torch.nn.Module,
                 TestLoader,
                 log,
                 tblog,
                 gpu_id: int,
                ) -> None:
        self.work_dir = work_dir
        self.cfg = cfg
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.save_paths = []
        self.TestLoader = TestLoader
        self.log = log
        self.tblog = tblog
        #Analyser = torch.jit.script(utils.Analyser(cfg.data.real_size, cfg.data.nms, split = OmegaConf.to_object(cfg.data.split))).to(device)
        self.Analyser = utils.Analyser(cfg.data.real_size, cfg.data.nms, split = OmegaConf.to_object(cfg.data.split)).to(self.gpu_id)
        self.ConfusionMatrixCounter = utils.ConfusionMatrixCounter()
        self.LostStat = utils.metStat()
        
        self.Criterion = utils.BoxClsLoss(wxy = cfg.criterion.xy_weight, 
                                     wz = cfg.criterion.z_weight, 
                                     wcls = cfg.criterion.cond_weight, 
                                     ion_weight=cfg.criterion.pos_weight
                                     ).to(self.gpu_id)
        
        self.best = np.inf
        
    def fit(self):
        self.log.info(f"Start testing {self.cfg.test_path}...")
        epoch_start_time = time.time()
        if self.cfg.label:
            test_loss, test_confusion = self.test_one_epoch(0)
            logstr = f"\n============= Summary Test | Epoch 0 ================\nloss: {test_loss:.2e} | used time: {(time.time() - epoch_start_time)/60:4.1f} mins | {'Model saved' if (test_loss < self.best) and (self.rank == 0) else 'Model not saved'}"
            for i, e in enumerate(utils.const.ion_order.split()):
                if e in self.cfg.data.ion_type:
                    logstr += f"\n=================   Element - '{e}'    =================\n(Overall)  AP: {test_confusion[i,:,4].mean():.2f} | AR: {test_confusion[i,:,3].mean():.2f} | ACC: {test_confusion[i,:,5].mean():.2f} | SUC: {test_confusion[i,:,6].mean():.2f}"
                    for j, (low, high) in enumerate(zip(self.cfg.data.split[:-1], self.cfg.data.split[1:])):
                        logstr += f"\n({low:.1f}-{high:.1f}A) AP: {test_confusion[i,j,4]:.2f} | AR: {test_confusion[i,j,3]:.2f} | ACC: {test_confusion[i,j,5]:.2f} | SUC: {test_confusion[i,j,6]:.2f}\nTP: {test_confusion[i,j,0]:10.0f} | FP: {test_confusion[i,j,1]:10.0f} | FN: {test_confusion[i,j,2]:10.0f}"
        else:
            logstr = f"\n============= Summary Test | Epoch 0 ================\n used time: {(time.time() - epoch_start_time)/60:4.1f}"
            self.valid_one_epoch(0)
            
        self.log.info(logstr)
        
      
    @torch.no_grad()
    def valid_one_epoch(self, epoch, log_every: int = 100, save_result_every: int | None = None) -> tuple[torch.Tensor]:
        self.model.eval()
        for i, (filenames, afm) in enumerate(self.TestLoader):
            afm = afm.to(self.gpu_id, non_blocking = True)
            pred_type, pred_pos, mu, var = self.model(afm)
            pred_pos, pred_ion_num = self.Analyser.process_pred(pred_type, pred_pos)
            if save_result_every is not None and i % save_result_every == 0:
                for filename, predpos, pred_ionnum in zip(filenames, pred_pos, pred_ion_num):
                    poscar.save(f"{self.work_dir}/{filename}.poscar", self.cfg.data.real_size, self.cfg.data.ion_type, pred_ionnum, predpos)
            if i % log_every == 0:
                self.log.info(f"Iter {i:5d}/{len(self.TestLoader):5d}")
        return

    @torch.no_grad()
    def test_one_epoch(self, epoch, log_every: int = 100, save_result_every: int | None = None) -> tuple[torch.Tensor]:
        self.model.eval()
        self.LostStat.reset()
        self.ConfusionMatrixCounter.reset()
        for i, (filenames, afm, targ_type, targ_pos) in enumerate(self.TestLoader):
            afm = afm.to(self.gpu_id, non_blocking = True)
            targ_type = targ_type.to(self.gpu_id, non_blocking = True)
            targ_pos = targ_pos.to(self.gpu_id, non_blocking = True)
            with torch.autocast(device_type='cuda'):
                pred_type, pred_pos, mu, var = self.model(afm)
                loss = self.Criterion(pred_type, pred_pos, targ_type, targ_pos)
            confusion_matrix = self.Analyser(pred_type, pred_pos, targ_type, targ_pos)
            self.LostStat.add(loss)
            self.ConfusionMatrixCounter(confusion_matrix)
            pred_pos, pred_ion_num = self.Analyser.process_pred(pred_type, pred_pos)
            if save_result_every is not None and i % save_result_every == 0:
                for filename, predpos, pred_ionnum in zip(filenames, pred_pos, pred_ion_num):
                    poscar.save(f"{self.work_dir}/{filename}.poscar", self.cfg.data.real_size, self.cfg.data.ion_type, pred_ionnum, predpos)
            if i % log_every == 0:
                self.log.info(f"Epoch {epoch:2d} | Iter {i:5d}/{len(self.TestLoader):5d} | Loss {loss:.2e}")
                self.tblog.add_scalar("Test/Loss", loss, epoch * len(self.TestLoader) + i)
                
        return self.LostStat.calc(), self.ConfusionMatrixCounter.calc()
                
def load_objs(cfg: DictConfig):
    work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    
    log = utils.get_logger(f"Rank 0")
    tblog = SummaryWriter(f"{work_dir}/runs")
    
    net = getattr(model, cfg.model.net)(**cfg.model.params)
    
    log.info(f"Network parameters: {sum([p.numel() for p in net.parameters()])}")
    
    try:
        missing = utils.model_load(net, cfg.checkpoint, True)
        log.info(f"Load parameters from {cfg.checkpoint}")
        if len(missing) > 0:
            log.info(f"Missing keys: {missing}")
    except:
        raise FileNotFoundError(f"Checkpoint {cfg.checkpoint} not found")
    
    if cfg.transform is None:
        transform = None
    else:
        transform = torch.nn.Sequential(dataset.Resize(tuple(cfg.data.image_size[1:])),
                                        dataset.PixelShift(),
                                        dataset.Cutout(),
                                        dataset.ColorJitter(),
                                        dataset.Noisy(),
                                        dataset.Blur()
                                        )
        
    TestDataset = dataset.AFMDataset(cfg.data.test_path, useLabel=True, useZ=cfg.data.image_size[0], transform=transform)
        
    return net, TestDataset, log, tblog

def prepare_dataloader(test_data, cfg: DictConfig):        
    TestLoader = torch.utils.data.DataLoader(test_data,
                                             batch_size=cfg.setting.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.setting.num_workers,
                                             pin_memory=cfg.setting.pin_memory,
                                             )
    
    return TestLoader


@hydra.main(config_path="conf", config_name="config", version_base=None) # hydra will automatically relocate the working dir.
def main(cfg):
    model, TestDataset, log, tblog = load_objs(cfg)
    TestLoader = prepare_dataloader(TestDataset, cfg)
    trainer = Trainer(hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir'],
                      cfg,
                      model,
                      TestLoader,
                      log,
                      tblog,
                      )
    trainer.fit()

if __name__ == "__main__":
    main()
    
