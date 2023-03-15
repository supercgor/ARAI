import os
import time
import torch
import numpy as np
from torch import nn

from .analyze_data import *
from .criterion import Criterion
from .tools import condense, ReverseLayerF
from .dataset import make_dataset, AFMPredictDataset
from .loader import Loader, modelLoader
from .logger import Logger

from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    def __init__(self, cfg, dann=False):
        self.cfg = cfg

        self.load_dir, self.work_dir = Loader(cfg)

        self.logger = Logger(
            path=f"{self.work_dir}", 
            log_name="train.log",
            elem = cfg.data.elem_name,
            split = cfg.setting.split)

        self.tb_writer = SummaryWriter(log_dir=f"{self.work_dir}/tb_log/")

        self.main_network = modelLoader(
            self.work_dir, keeps=cfg.setting.max_save)

        log = self.main_network.load(self.load_dir,
                                     best=cfg.model.best,
                                     net=cfg.model.net,
                                     cuda = True,
                                     inp_size=cfg.model.inp_size,
                                     out_size=cfg.model.out_size,
                                     hidden_channels=cfg.model.channels,
                                     out_feature = True)
        
        self.model = self.main_network.model
        
        self.logger.info(log)
        
        # ---------------- domain adaptation network ----------------0
        
        self.domain_network = modelLoader(self.work_dir, keeps = 1)
        
        log = self.domain_network.load(net = "SqueezeNet",
                                       cuda = True,
                                       in_channels = 64, 
                                       sample_size = cfg.model.inp_size[1] // 4, 
                                       sample_duration = cfg.model.inp_size[0], 
                                       num_classes = 2)
        
        self.classicfier = self.domain_network.model
        
        self.class_loss = nn.CrossEntropyLoss(reduction = "mean")
        
        self.logger.info(log)
        
        self.analyzer = Analyzer(cfg).cuda()

    def fit(self):
        # --------------------------------------------------
        start_time = time.time()

        cfg = self.cfg
        logger = self.logger
        tbw = self.tb_writer

        self.train_loader = make_dataset('train', cfg)
        
        domain_dataset = AFMPredictDataset(
            f"{cfg.path.data_root}/lots_exp/HDA",
            cfg.data.elem_name,
            file_list="domain1.filelist",
            model_inp=cfg.model.inp_size,
            model_out=cfg.model.out_size)
        
        self.domain_loader = torch.utils.data.DataLoader(
            domain_dataset,
            batch_size=cfg.setting.batch_size // 2,
            num_workers=cfg.setting.num_workers,
            pin_memory=cfg.setting.pin_memory)
        
        self.valid_loader = make_dataset('valid', cfg)

        self.criterion = Criterion(cfg, cfg.setting.local_epoch).cuda()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.setting.learning_rate)

        self.best_ACC = [0.0, 0.0]

        self.best_LOSS = 999

        # --------------------------------------------------
        logger.info(f'Start training.')

        for epoch in range(1, cfg.setting.epochs + 1):
            epoch_start_time = time.time()

            log_train_dic = self.train(epoch)

            if True:  # Can add some condition
                log_valid_dic = self.valid(epoch)
            else:
                log_valid_dic = {}

            logger.epoch_info(epoch, log_train_dic, log_valid_dic)
            for key in ['loss', 'grad']:
                tbw.add_scalar(f"epoch/train/{key}", log_train_dic[key], epoch)
            for key in log_train_dic['count']:
                tbw.add_scalar(
                    f"epoch/train/{key}", log_train_dic['count'][key], epoch)
            for key in ['loss']:
                tbw.add_scalar(f"epoch/valid/{key}", log_valid_dic[key], epoch)
            for key in log_valid_dic['count']:
                tbw.add_scalar(
                    f"epoch/valid/{key}", log_valid_dic['count'][key], epoch)
            # ---------------------------------------------
            # Saver here

            self.save(epoch, log_valid_dic)

            logger.info(f"Spend time: {time.time() - epoch_start_time:.2f}s")

            logger.info(f'End training epoch: {epoch:0d}')

        # --------------------------------------------------

        logger.info(f'End training.')

    def train(self, epoch):
        # -------------------------------------------

        model = self.model
        class_model = self.classicfier
        criterion = self.criterion
        analyzer = self.analyzer
        optimizer = self.optimizer
        cfg = self.cfg
        tbw = self.tb_writer
        len_dataloader = min(len(self.train_loader), len(self.domain_loader))
        
        it_train_loader = iter(self.train_loader)
        it_domain_loader = iter(self.domain_loader)
        
        # -------------------------------------------

        model.train()
        log_dic = {'loss': [], 'grad': [], 'count': []}

        i = 0
        while i < len_dataloader:
            if cfg.setting.show:
                t = time.time()
                print(f'\r{i}/{len_dataloader}', end='')
            
            p = float(i + epoch * len_dataloader) / cfg.setting.epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            inputs, targets, domain, _ = next(it_train_loader)
            
            inputs = inputs.cuda(non_blocking = True)
            
            targets = targets.cuda(non_blocking = True)
            
            domain = domain.cuda(non_blocking = True)
            
            predictions, feature = model(inputs)
            
            reverse_feature = ReverseLayerF.apply(feature, alpha)
            
            class_pred = class_model(reverse_feature)
            
            loss = criterion(predictions, targets)

            loss_class = self.class_loss(class_pred, domain)
            
            info = analyzer(predictions, targets)

            loss_local = criterion.loss_local(epoch, info)

            loss = loss + loss_class + loss_local
            
            inputs, domain, _ = next(it_domain_loader)

            inputs = inputs.cuda(non_blocking = True)
                        
            domain = domain.cuda(non_blocking = True)
            
            _, feature = model(inputs)
            
            reverse_feature = ReverseLayerF.apply(feature, alpha)
            
            class_pred = class_model(feature)
            
            loss_class = self.class_loss(class_pred, domain)
            
            loss = loss + loss_class
            
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.setting.clip_grad, error_if_nonfinite=True)

            optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            log_dic['count'].append(analyzer.count(info))
            log_dic['loss'].append(loss)
            log_dic['grad'].append(grad_norm)

            for key in ['loss', 'grad']:
                tbw.add_scalar(
                    f"train/step.{key}", log_dic[key][-1], (epoch-1) * len_dataloader + i)

            if i == 0:
                batch, _, Z, X, Y = inputs.shape
                input_img = torch.reshape(inputs, (batch, Z, 1, X, Y))
                input_img = make_grid(input_img[0])
                tbw.add_image(f"train/epoch.input_img",
                              input_img, global_step=epoch)
                batch, X, Y, Z, _ = predictions.shape
                out_img = predictions[0, ..., (3, 7)]
                out_img = torch.permute(out_img, (2, 3, 0, 1))  # -> C, H, W
                out_img = torch.max(out_img, dim=0)[0]
                out_img = torch.reshape(out_img, (2, 1, X, Y))
                out_img = out_img > 0
                tar_img = targets[0, ..., (3, 7)]
                tar_img = torch.permute(tar_img, (2, 3, 0, 1))  # -> C, H, W
                tar_img = torch.max(tar_img, dim=0)[0]
                tar_img = torch.reshape(tar_img, (2, 1, X, Y))
                out_img = torch.cat([tar_img, tar_img, out_img], dim=1)
                tbw.add_images(f"train/epoch.out_img", out_img,
                               global_step=epoch, dataformats="NCHW")

            if cfg.setting.show:
                print(
                    f' time: {(time.time() - t):.2f}, loss: {loss.item():.4f}', end='')
            
            i += 1
        # -------------------------------------------
        log_dic = condense(log_dic)

        for key in log_dic['count'].keys():
            if key[-3:] in ["ACC", "SUC"]:
                log_dic['count'][key] = torch.mean(log_dic['count'][key])
            else:
                log_dic['count'][key] = torch.sum(log_dic['count'][key])

        for key in log_dic.keys():
            if key != "count":
                log_dic[key] = torch.mean(log_dic[key])

        return log_dic

    @torch.no_grad()
    def valid(self, epoch):
        # -------------------------------------------

        model = self.model
        analyzer = self.analyzer
        criterion = self.criterion
        cfg = self.cfg
        tbw = self.tb_writer

        # -------------------------------------------

        model.eval()  # 切换模型为预测模式
        log_dic = {'loss': [], 'count': []}

        for i, (inputs, targets, _, _) in enumerate(self.valid_loader):

            if cfg.setting.show:
                t = time.time()
                print(f'\r{i}/{len(self.valid_loader)}', end='')

            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            predictions, _ = model(inputs)

            loss = criterion(predictions, targets)

            info = analyzer(predictions, targets)

            loss_local = criterion.loss_local(epoch, info)

            loss = loss + loss_local

            log_dic['count'].append(analyzer.count(info))
            log_dic['loss'].append(loss)

            if i == 0:
                batch, _, Z, X, Y = inputs.shape
                input_img = torch.reshape(inputs, (batch, Z, 1, X, Y))
                input_img = make_grid(input_img[0])
                tbw.add_image(f"valid/epoch.input_img",
                              input_img, global_step=epoch)
                batch, X, Y, Z, _ = predictions.shape
                out_img = predictions[0, ..., (3, 7)]
                out_img = torch.permute(out_img, (2, 3, 0, 1))  # -> C, H, W
                out_img = torch.max(out_img, dim=0)[0]
                out_img = torch.reshape(out_img, (2, 1, X, Y))
                out_img = out_img > 0
                tar_img = targets[0, ..., (3, 7)]
                tar_img = torch.permute(tar_img, (2, 3, 0, 1))  # -> C, H, W
                tar_img = torch.max(tar_img, dim=0)[0]
                tar_img = torch.reshape(tar_img, (2, 1, X, Y))
                out_img = torch.cat([tar_img, tar_img, out_img], dim=1)
                tbw.add_images(f"valid/epoch.out_img", out_img,
                               global_step=epoch, dataformats="NCHW")

            if cfg.setting.show:
                print(
                    f' time: {(time.time() - t):.2f}, loss: {loss.item():.4f}', end='')

        # -------------------------------------------
        log_dic = condense(log_dic)

        for key in log_dic['count'].keys():
            if key[-3:] in ["ACC", "SUC"]:
                log_dic['count'][key] = torch.mean(log_dic['count'][key])
            else:
                log_dic['count'][key] = torch.sum(log_dic['count'][key])

        for key in log_dic.keys():
            if key != "count":
                log_dic[key] = torch.mean(log_dic[key])

        return log_dic

    def save(self, epoch, log_dic):

        cfg = self.cfg
        log_count = log_dic['count']
        log_loss = log_dic['loss']
        model = self.model
        logger = self.logger
        elem = cfg.data.elem_name
        split = cfg.setting.split
        split = [f"{split[i]}-{split[i+1]}" for i in range(len(split)-1)]

        # -------------------------------------------

        save = False
        ele_ACC = []
        for ele in elem:
            ele_ACC.append(min(log_count[f"{ele}-{i}-ACC"] for i in split))
        if (ele_ACC[0] > self.best_ACC[0]) and (ele_ACC[1] > self.best_ACC[1]):
            save = True
        elif (ele_ACC[0] < self.best_ACC[0]) and (ele_ACC[1] > self.best_ACC[1]):
            save = True
        else:
            save = False

        if log_loss < self.best_LOSS:
            save = True

        if save:
            self.best_ACC = ele_ACC
            self.best_LOSS = log_loss
            model_name = f"CP{epoch:02d}_"
            model_name += "_".join([f"{ele}{ACC.item():.4f}" for ele,
                                   ACC in zip(elem, ele_ACC)])
            model_name += f"_{self.best_LOSS:.6f}.pkl"

            self.ml.save_model(model_name)
            self.ml.save_info(cfg)

            logger.info(f"Saved a new model: {model_name}")
        else:
            logger.info(f"No model was saved")
