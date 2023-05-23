import os
import time
import numpy as np
from tqdm import tqdm
import json

import torch
from torch import nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from network.basic import basicParallel

from utils.metrics import *
from utils.criterion import Criterion
from utils.tools import metStat, output_target_to_imgs, fill_dict
from datasets.dataset import make_dataset
from utils.loader import Loader
from utils.logger import Logger
from utils.loader import poscarLoader

class Test():
    def __init__(self, cfg):
        self.cfg = cfg
        self.cuda = cfg.setting.device != []
        
        self.load_dir, self.work_dir = Loader(cfg, make_dir=False)

        self.logger = Logger(path=f"{self.work_dir}",log_name="test.log",elem=cfg.data.elem_name,split=cfg.setting.split)
        
        self.tb_writer = SummaryWriter(log_dir=f"{self.work_dir}/runs/Test")
        
        self.model = model[cfg.model.net](inp_size=cfg.model.inp_size, out_size=cfg.model.out_size, hidden_channels=cfg.model.channels, out_feature= False)
        
        log = f"Load model parameters from {self.load_dir}/{cfg.model.best}"
        
        self.logger.info(log)

        self.model = self.model.eval()

        if self.cuda():
            self.model.cuda()
            self.analyzer = Analyzer(cfg).cuda()
            self.ana2 = Analyzer2().cuda
        
        else:
            self.analyzer = Analyzer(cfg)
            self.ana2 = Analyzer2()
        
        self.pd_loader = make_dataset('test', cfg)
        
        self.pl = poscarLoader(path = f"{cfg.path.data_root}/{cfg.data.dataset}", model_name=cfg.model.checkpoint)
            
    @torch.no_grad()
    def test(self, npy = True):
        len_loader = len(self.pd_loader)
        it_loader = iter(self.pd_loader)
        
        log_dic = {'loss': metStat()}
        self.logger.info(f'Start Testing.')
        
        start_time = time.time()
                
        pbar = tqdm(total=len_loader, desc=f"{self.cfg.model.net} - Test", position=0, leave=True, unit='it')
        
        i = 0
        while i < len_loader:
            inputs, targets, filenames = next(it_loader)

            if self.cuda:
                inputs = inputs.cuda(non_blocking = True)
                targets = targets.cuda(non_blocking = True)
                
            predictions = self.model(inputs)
            info = self.analyzer(predictions, targets)
            count_info = self.analyzer.count(info)
            
            for key in count_info:
                if key in log_dic:
                    log_dic[key].add(count_info[key])
                else:
                    log_dic[key] = count_info[key]
            
            
            for filename, x in zip(filenames, predictions):
                self.pl.save4npy(f"{filename}A.poscar", x, conf = self.cfg.model.threshold)
                if npy:
                    if not os.path.exists(f"{self.cfg.path.data_root}/{self.cfg.data.dataset}/npy/{self.cfg.model.checkpoint}"):
                        os.mkdir(f"{self.cfg.path.data_root}/{self.cfg.data.dataset}/npy/{self.cfg.model.checkpoint}")
                    torch.save(x.cpu().numpy(), f"{self.cfg.path.data_root}/{self.cfg.data.dataset}/npy/{self.cfg.model.checkpoint}/{filename}.npy")
                

            pbar.update(1)
        # -------------------------------------------

        pbar.update(1)
        pbar.close()
        # analyzer.rdf.save()
  
        self.logger.info(f"Spend time: {time.time() - start_time:.2f}s")
        
        self.logger.info(f'End prediction')