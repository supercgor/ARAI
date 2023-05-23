import os
import time
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils.metrics import *
from datasets.dataset import make_dataset
from utils.loader import Loader, modelLoader, poscarLoader
from utils.logger import Logger

class Pred():
    def __init__(self, cfg):
        self.cfg = cfg
        self.cuda = cfg.setting.device != []
        
        self.load_dir, self.work_dir = Loader(cfg, make_dir=False)

        self.logger = Logger(path=f"{self.work_dir}",log_name="test.log",elem=cfg.data.elem_name,split=cfg.setting.split)
        
        self.tb_writer = SummaryWriter(log_dir=f"{self.work_dir}/runs/Pred")
        
        self.main_network = modelLoader(self.load_dir, self.work_dir, model_keeps=0, cuda=self.cuda)
        
        log = self.main_network.load(net=cfg.model.net, inp_size=cfg.model.inp_size, out_size=cfg.model.out_size, hidden_channels=cfg.model.channels, out_feature= False)
        
        self.model = self.main_network.model.eval()

        self.logger.info(log)
        
        self.analyzer = Analyzer(cfg).cuda() if self.cuda else Analyzer(cfg)
        
        self.pd_loader = make_dataset('predict', cfg)
        
        self.pl = poscarLoader(path = "/home/supercgor/gitfile/ARAI/datasets/data/exp", model_name=cfg.model.checkpoint)
            
    @torch.no_grad()
    def predict(self, npy = False):
        len_loader = len(self.pd_loader)
        it_loader = iter(self.pd_loader)
        
        self.logger.info(f'Start prediction.')
        
        start_time = time.time()
                
        pbar = tqdm(total=len_loader, desc=f"{self.cfg.model.net} - Pred", position=0, leave=True, unit='it')
        
        i = 0
        while i < len_loader:
            inputs, filenames = next(it_loader)

            if self.cuda:
                inputs = inputs.cuda(non_blocking= True)
            
            predictions = self.model(inputs)
            
            for filename, x in zip(filenames, predictions):
                self.pl.save4npy(f"{filename}.poscar", x, conf = self.cfg.model.threshold)
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