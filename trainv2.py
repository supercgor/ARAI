import os
import hydra
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import dataset
import model
import utils

from omegaconf import OmegaConf, DictConfig


@hydra.main(config_path="conf", config_name="config", version_base=None) # hydra will automatically relocate the working dir.
def main(cfg: DictConfig) -> None:
    def train():
        LostStat.reset()
        GradStat.reset()
        ConfusionMatrixCounter.reset()
        for i, (filename, afm, targ_type, targ_pos) in enumerate(TrainLoader):
            afm = afm.to(device, non_blocking = True)
            targ_type = targ_type.to(device, non_blocking = True)
            targ_pos = targ_pos.to(device, non_blocking = True)
            pred_type, pred_pos, mu, var = net(afm)
            loss = Criterion(pred_type, pred_pos, targ_type, targ_pos)
            Optimizer.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.criterion.clip_grad, error_if_nonfinite=True)
            LostStat.add(loss)
            GradStat.add(grad)
            confusion_matrix = Analyser(pred_type, pred_pos, targ_type, targ_pos)
            ConfusionMatrixCounter(confusion_matrix)
            Optimizer.step()
            if local_rank == 0 and i % 1 == 0:
                log.info(f"Epoch {epoch} | Iter {i} | Loss {loss:.2e} | Grad {grad:.2e}")
        Schedular.step()
        return LostStat.calc(), GradStat.calc(), ConfusionMatrixCounter.calc()
            
    @torch.no_grad()
    def valid():
        LostStat.reset()
        ConfusionMatrixCounter.reset()
        for i, (filename, afm, targ_type, targ_pos) in enumerate(TestLoader):
            print(i)
            afm = afm.to(device, non_blocking = True)
            targ_type = targ_type.to(device, non_blocking = True)
            targ_pos = targ_pos.to(device, non_blocking = True)
            pred_type, pred_pos, mu, var = net(afm)
            loss = Criterion(pred_type, pred_pos, targ_type, targ_pos)
            confusion_matrix = Analyser(pred_type, pred_pos, targ_type, targ_pos)
            LostStat.add(loss)
            ConfusionMatrixCounter(confusion_matrix)
        return LostStat.calc(), ConfusionMatrixCounter.calc()
    
    work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    log = utils.get_logger("Train")
    tblog = SummaryWriter(work_dir)
    save_paths = []
    local_rank = 0
    
    net = getattr(model, cfg.model.net)(**cfg.model.params)
    log.info(f"Network parameters: {sum([p.numel() for p in net.parameters()])}")

    if cfg.model.checkpoint is None:
        log.info("Start a new model")
    else:
        missing = utils.model_load(net, cfg.model.checkpoint, True)
        log.info(f"Load parameters from {cfg.model.checkpoint}")
        log.info(f"Missing keys: {missing}")
    
    if len(cfg.setting.device) >= 1:
        device = torch.device(f"cuda:{cfg.setting.device[0]}")
    else:
        device = torch.device("cpu")
        
    net = net.to(device)
    if len(cfg.setting.device) >= 2:
        pass
    
    transform = torch.nn.Sequential(
        dataset.Resize(tuple(cfg.data.image_size[1:])),
        dataset.PixelShift(),
        dataset.Cutout(),
        dataset.ColorJitter(),
        dataset.Noisy(),
        dataset.Blur()
    )
    
    TrainDataset = dataset.AFMDataset(cfg.data.train_path, useLabel=True, useZ=cfg.data.image_size[0], transform=transform)
    TestDataset = dataset.AFMDataset(cfg.data.test_path, useLabel=True, useZ=cfg.data.image_size[0], transform=transform)
    
    TrainLoader = torch.utils.data.DataLoader(TrainDataset, 
                                              batch_size=cfg.setting.batch_size, 
                                              shuffle=True, 
                                              num_workers=cfg.setting.num_workers, 
                                              pin_memory=cfg.setting.pin_memory
                                              )
    
    TestLoader = torch.utils.data.DataLoader(TestDataset,
                                             batch_size=cfg.setting.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.setting.num_workers,
                                             pin_memory=cfg.setting.pin_memory
                                             )
    
    #Analyser = torch.jit.script(utils.Analyser(cfg.data.real_size, cfg.data.nms, split = OmegaConf.to_object(cfg.data.split))).to(device)
    Analyser = utils.Analyser(cfg.data.real_size, cfg.data.nms, split = OmegaConf.to_object(cfg.data.split)).to(device)

    
    ConfusionMatrixCounter = utils.ConfusionMatrixCounter()
    LostStat = utils.metStat()
    GradStat = utils.metStat()
    
    Criterion = utils.BoxClsLoss(wxy = cfg.criterion.xy_weight, 
                                 wz = cfg.criterion.z_weight, 
                                 wcls = cfg.criterion.cond_weight, 
                                 ion_weight=cfg.criterion.pos_weight
                                 ).to(device)
    
    Optimizer = torch.optim.AdamW(net.parameters(), 
                                  lr=cfg.criterion.lr, 
                                  weight_decay=cfg.criterion.weight_decay
                                  )
    
    Schedular = getattr(torch.optim.lr_scheduler, cfg.criterion.schedular.name)(Optimizer, **cfg.criterion.schedular.params)
    
    best = np.inf
    
    for epoch in range(cfg.setting.epoch):
        log.info(f"({local_rank})Start training epoch: {epoch}...")
        epoch_start_time = time.time()
        train_loss, train_grad, train_confusion = train()
        
        logstr = f"""
        ============ ({local_rank})Summary Train | Epoch {epoch:2d} ==============
        loss: {train_loss:.2e} | grad: {train_grad:.2e} | used time: {(time.time() - epoch_start_time)/60:4.1f} mins
        """
        
        for i, e in enumerate(utils.ion_order.split()):
            if e in cfg.data.ion_order:
                logstr += f"""
        =================   Element - '{e}'    =================
        (Overall)  AP: {train_confusion[i,:,4].mean():.2f} | AR: {train_confusion[i,:,3].mean():.2f} | ACC: {train_confusion[i,:,5].mean():.2f} | SUC: {train_confusion[i,:,6].mean():.2f}
        """
                for j, (low, high) in enumerate(zip(cfg.data.split[:-1], cfg.data.split[1:])):
                    logstr += f"""
        ({low:.1f}-{high:.1f}A) AP: {train_confusion[i,j,4]:.2f} | AR: {train_confusion[i,j,3]:.2f} | ACC: {train_confusion[i,j,5]:.2f} | SUC: {train_confusion[i,j,6]:.2f}
        TP: {train_confusion[i,j,0]:10d} | FP: {train_confusion[i,j,1]:10d} | FN: {train_confusion[i,j,2]:10d}
        """
        log.info(logstr)
        
        test_loss, test_confusion = valid()
        
        logstr = f"""
        ============ ({local_rank})Summary Test | Epoch {epoch:2d} ==============
        loss: {test_loss:.2e} | used time: {(time.time() - epoch_start_time)/60:4.1f} mins | {'Model saved' if (test_loss < best) and (local_rank == 0) else 'Model not saved'}
        """
        
        for i, e in enumerate(utils.ion_order.split()):
            if e in cfg.data.ion_order:
                logstr += f"""
        =================   Element - '{e}'    =================
        (Overall)  AP: {test_confusion[i,:,4].mean():.2f} | AR: {test_confusion[i,:,3].mean():.2f} | ACC: {test_confusion[i,:,5].mean():.2f} | SUC: {test_confusion[i,:,6].mean():.2f}
        """
                for j, (low, high) in enumerate(zip(cfg.data.split[:-1], cfg.data.split[1:])):
                    logstr += f"""
        ({low:.1f}-{high:.1f}A) AP: {test_confusion[i,j,4]:.2f} | AR: {test_confusion[i,j,3]:.2f} | ACC: {test_confusion[i,j,5]:.2f} | SUC: {test_confusion[i,j,6]:.2f}
        TP: {test_confusion[i,j,0]:10d} | FP: {test_confusion[i,j,1]:10d} | FN: {test_confusion[i,j,2]:10d}
        """
        
        log.info(logstr)
        
        if local_rank == 0:
            # log
            tblog.add_scalar("Train/Loss", train_loss, epoch)
            tblog.add_scalar("Valid/Loss", test_loss, epoch)
            tblog.add_scalar("Train/Grad", train_grad, epoch)
            
            # save model
            if test_loss < best:
                best = test_loss
                path = f"{work_dir}/unet_CP{epoch:02d}_LOSS{test_loss:.4f}.pkl"
                if len(save_paths) >= cfg.model.max_save:
                    os.remove(save_paths.pop(0))
                utils.model_save(net, path)
                save_paths.append(path)
            
if __name__ == "__main__":
    main()