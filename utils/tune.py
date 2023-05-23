import time
from tqdm import tqdm
from collections import OrderedDict
from itertools import chain
import math


import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import model
from model.op import grad_reverse


from datasets.dataset import make_dataset
from utils.logger import Logger
from utils.loader import Loader
from utils.metrics import metStat, analyse_cls
from utils.criterion import modelLoss, wassersteinLoss, grad_penalty
from utils.schedular import Scheduler
from demo.plot import out2img

class Tuner():
    def __init__(self, cfg):
        self.cfg = cfg
        
        assert cfg.setting.device != [], "No device is specified!"

        self.load_dir, self.work_dir = Loader(cfg, make_dir=True)

        self.logger = Logger(path=self.work_dir,
                             elem=cfg.data.elem_name, 
                             split=cfg.setting.split,
                             log_name = "tune.log")

        self.tb_writer = SummaryWriter(
            log_dir=f"{self.work_dir}/runs/Tuner_{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

        log = []
        self.net = model.CombineModel().cuda()

        try:
            paths = {"fea": f"{self.load_dir}/{cfg.model.fea}",
                    "neck": f"{self.load_dir}/{cfg.model.neck}", 
                    "head": f"{self.load_dir}/{cfg.model.head}"}
            match_list = self.net.load(paths, pretrained=True)
            match_list = "\n".join(match_list)
            log.append(f"Load parameters from {self.load_dir}")
            log.append(f"\n{match_list}")
        except (FileNotFoundError, IsADirectoryError):
            self.net.init()
            log.append(
                f"No network is loaded, start a new model: {self.net.name}")

        self.disc = model.NLayerDiscriminator().cuda()
        
        try:
            path = f"{self.load_dir}/{cfg.model.disc}"
            match_list = self.disc.load(path, pretrained=True)
            match_list = "\n".join(match_list)
            log.append(f"Load parameters from {self.load_dir}")
            log.append(f"\n{match_list}")
        except (FileNotFoundError, IsADirectoryError):
            self.disc.init()
            log.append(
                f"No discriminator is loaded, start a new model: {self.disc.name}")

        if len(cfg.setting.device) >= 2:
            self.net.parallel(devices_ids=cfg.setting.device)
            self.disc = model.utils.basicParallel(self.disc, device_ids=cfg.setting.device)
        
        self.net.train()
        self.disc.train()
        
        self.analyse = analyse_cls(threshold=cfg.model.threshold).cuda()
        self.LOSS = modelLoss(threshold = cfg.model.threshold, pos_w = cfg.criterion.pos_weight).cuda()
        
        #self.DLOSS = nn.BCEWithLogitsLoss()

        self.OPT = torch.optim.AdamW(self.net.parameters(), lr=self.cfg.setting.lr, weight_decay= 5e-3)
        self.DOPT = torch.optim.Adam(self.disc.parameters(), lr=self.cfg.setting.lr, betas=(0, 0.9))
        self.SCHEDULER = Scheduler(self.OPT, warmup_steps=10, decay_factor=1000)
        self.DSCHEDULER = Scheduler(self.DOPT, warmup_steps=10, decay_factor=1000)
        
        for l in log:
            self.logger.info(l)

    def fit(self):
        # --------------------------------------------------

        self.train_loader = make_dataset('train', self.cfg)

        self.valid_loader = make_dataset('valid', self.cfg)

        self.dann_loader = make_dataset('dann', self.cfg)
     
        self.best = {'loss': metStat(mode="min")}
        self.best_met = 9999

        # --------------------------------------------------
        self.logger.info(f'Start tunning.')

        for epoch in range(1, self.cfg.setting.epochs + 1):
            epoch_start_time = time.time()

            log_train_dic = self.train(epoch)

            if True:  # Can add some condition
                log_valid_dic = self.valid(epoch)
            else:
                log_valid_dic = {}

            self.logger.epoch_info(epoch, log_train_dic, log_valid_dic)

            for dic_name, dic in zip(["Train", "Valid"], [log_train_dic, log_valid_dic]):
                for key, MET in dic.items():
                    if key != 'MET':
                        self.tb_writer.add_scalar(
                            f"EPOCH/{dic_name} {key}", MET.value, epoch)
                    else:
                        # log Metrics dict
                        for e in MET.elems:
                            self.tb_writer.add_scalars(
                                f"EPOCH/{dic_name} {e} COUNT", {f"{met_name} {l}": MET[e, i, met_name] for i,l in enumerate(MET.split) for met_name in ["TP", "FP", "FN", "T", "P"]}, epoch)
                            self.tb_writer.add_scalars(
                                f"EPOCH/{dic_name} {e} AP/AR", {f"{met_name} {l}": MET[e, i, met_name] for i,l in enumerate(MET.split) for met_name in ["AP", "AR"]}, epoch)
                            
            # ---------------------------------------------
            # Saver here

            self.save(epoch, log_valid_dic)

            self.logger.info(
                f"Spend time: {time.time() - epoch_start_time:.2f}s")

            self.logger.info(f'End training epoch: {epoch:0d}')

        # --------------------------------------------------

        self.logger.info(f'End tunning.')


    @staticmethod
    def get_dict():
        T_dict = OrderedDict(
            Grad =metStat(mode="mean"),
            Loss =metStat(mode="mean"),
            LossD =metStat(mode="mean"),
            GradD = metStat(mode="mean"),
        )
        
        return T_dict
    
    def train(self, epoch):
        # -------------------------------------------
        # for some reasons, daaccu >= accu
        accu = self.cfg.setting.batch_accumulation
        daccu = self.cfg.setting.disc_accumulation
        
        iter_times = len(self.dann_loader) // daccu
        
        source_repeat = 5 # len(self.train_loader) // len(self.dann_loader)
        assert source_repeat >= 1, "get less data in source that in target"
        
        it_loader = iter(self.train_loader)
        it_dloader = iter(self.dann_loader)

        # -------------------------------------------

        T_dict = self.get_dict()

        pbar = tqdm(total=iter_times - 1, desc=f"Epoch {epoch} - Tune", position=0, leave=True, unit='it')

        i = 0
        while i < iter_times:
            step = (epoch-1) * iter_times + i
            # source domain training
            for t in range(source_repeat):
                self.OPT.zero_grad()
                
                imgs, gt_box, _ = next(it_loader)
                imgs = imgs.cuda(non_blocking=True)
                gt_box = gt_box.cuda(non_blocking=True)
                
                pd_box, fake_fea = self.net(imgs, require_fea=True)
                
                fake = self.disc(grad_reverse(fake_fea))
                
                match = self.analyse(pd_box, gt_box)
                
                loss_g = self.LOSS(pd_box, gt_box)
                
                if t == 0:
                    self.DOPT.zero_grad()
                    loss = loss_g + fake.mean()
                else:
                    loss = loss_g
                    
                loss.backward()
                
                if t == 0:
                    
                    imgs, _ = next(it_dloader)
                    imgs = imgs.cuda(non_blocking=True)
                    _, real_fea = self.net(imgs, require_fea=True)
                    
                    real = self.disc(grad_reverse(real_fea))
                    
                    loss_d = fake.mean().item() - real.mean()
                    
                    loss_d.backward()
                    
                    loss_gp = 10 * grad_penalty(self.disc, real_fea, fake_fea)
                    loss_gp.backward()
                    
                    grad_d = nn.utils.clip_grad_norm_(self.disc.parameters(), 500, error_if_nonfinite = True)
                    self.DOPT.step()
                    
                    T_dict['LossD'].add(loss_d)
                    T_dict['GradD'].add(grad_d)
                
                grad_g = nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.setting.clip_grad, error_if_nonfinite = True)
                
                self.OPT.step()
                
                T_dict['Loss'].add(loss_g)
                T_dict['Grad'].add(grad_g)
                
                pbar.set_postfix(Lg= T_dict['Loss'].last,
                             Gg= T_dict['Grad'].last,
                             Ld= T_dict['LossD'].last,
                             Gd= T_dict['GradD'].last)
                    
            self.SCHEDULER.step()
            self.DSCHEDULER.step()
            
            self.tb_writer.add_scalars(f"TRAIN", {key: value.last for key, value in T_dict.items()}, step)
           
            if step % 100 == 0:
                self.tb_writer.add_images("Train/In IMG", imgs[0].permute(1,0,2,3), step)
                self.tb_writer.add_image("Train/OUT BOX", out2img(pd_box, gt_box), step)
                
            self.tb_writer.add_scalar(f"TRAIN/LR_rate", self.OPT.param_groups[0]['lr'], step)
            
            for e in ["O","H"]:
                for j,l in enumerate(match.split):
                    self.tb_writer.add_scalars(f"TRAIN/{e} {l}", {key: match[e,j,key] for key in ["AP", "AR"]}, step)
            
            i += 1
            pbar.update(1)
    
        # -------------------------------------------
        pbar.update(1)
        pbar.close()
        return {**T_dict, "MET": self.analyse.summary()}

    @torch.no_grad()
    def valid(self, epoch):
        # -------------------------------------------

        T_dict = self.get_dict()
        len_loader = len(self.valid_loader) // 10 if epoch % 5 == 0 else len(self.valid_loader)
        it_loader = iter(self.valid_loader)

        self.net.eval()

        T_dict = self.get_dict()

        pbar = tqdm(total=len_loader - 1,
                    desc=f"Epoch {epoch} -  Valid", position=0, leave=True, unit='it')

        i = 0
        while i < len_loader:
            step = (epoch-1) * len_loader + i
            imgs, gt_box, filenames = next(it_loader)

            imgs = imgs.cuda(non_blocking=True)
            gt_box = gt_box.cuda(non_blocking=True)
            pd_box = self.net(imgs)

            loss = self.LOSS(pd_box, gt_box)
                        
            match = self.analyse(pd_box, gt_box)
            T_dict['Loss'].add(loss)

            pbar.set_postfix(Loss= T_dict['Loss'].last)
            
            i += 1
            pbar.update(1)

            # log Valid dict
            self.tb_writer.add_scalar(f"VALID/Loss", T_dict['Loss'].last, step)
        # -------------------------------------------

        pbar.update(1)
        pbar.close()

        return {**T_dict, "MET": self.analyse.summary()}


    def save(self, epoch, log_dic):
        met = 0
        if epoch < 5:
            return 
        if log_dic["Loss"].n > 0:
            met += log_dic["Loss"]()

        logger = self.logger

        if met < self.best_met:
            self.best_met = met

            log = []
            try:
                name = f"CP{epoch:02d}_Tune_LOSS{log_dic['Loss']:.4f}.pkl"
                self.net.save(path = self.work_dir, name = name)
                self.disc.save(path = f"{self.work_dir}/DISC_{name}")
                log.append(f"Saved a new net: {name}")
            except AttributeError:
                pass

            for i in log:
                logger.info(i)

        else:
            logger.info(f"No model was saved")