import os
import time
import torch

from .model_saver import ModelSaver
from .log import Logger
from .checker import *
from .analyze_data import *
from .criterion import Criterion
from .tools import condense
from .const import MODEL_SAVE_DIR
from .dataset import make_dataset

from network.unet3d_model import UNet3D


class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        cfg = self.cfg
        
        self.logger = Logger(cfg)

        model = model_load(UNet3D, 
                           cfg.MODEL.CHANNELS, 
                           cfg.DATA.Z, 
                           cfg.TRAIN.CHECKPOINT,
                           logger = self.logger)

        self.device_num = len(cfg.TRAIN.DEVICE)

        self.model, self.device = gpu_condition(model, cfg.TRAIN.DEVICE, logger = self.logger)

        self.analyzer = Analyzer(cfg).cuda()
        
    def fit(self):
        # --------------------------------------------------
        start_time = time.time()
        
        cfg = self.cfg
        logger = self.logger
        
        self.saver = ModelSaver(cfg.TRAIN.MAX_SAVE)

        self.train_loader = make_dataset('train', cfg)
        self.valid_loader = make_dataset('valid', cfg)

        self.criterion = Criterion(cfg,cfg.TRAIN.CRITERION.LOCAL).cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.TRAIN.LR)

        self.best_ACC = [0.0,0.0]
        
        self.best_LOSS = 999

        # --------------------------------------------------
        logger.info(f'Start training.')

        for epoch in range(1, cfg.TRAIN.EPOCHS + 1):
            epoch_start_time = time.time()

            log_train_dic = self.train(epoch)
            
            if True: # Can add some condition
                log_valid_dic = self.valid(epoch)
            else:
                log_valid_dic = {}
            
            logger.epoch_info(epoch,log_train_dic,log_valid_dic)
            
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
        criterion = self.criterion
        analyzer = self.analyzer
        optimizer = self.optimizer
        cfg = self.cfg

        # -------------------------------------------

        model.train()
        log_dic = {'loss':[],'grad':[],'count':[]}
        
        for i, (inputs, targets, filename, _) in enumerate(self.train_loader):
            
            if cfg.TRAIN.SHOW:
                t = time.time()
                print(f'\r{i}/{len(self.train_loader)}', end='')

            inputs = inputs.cuda(non_blocking= True)
            targets = targets.cuda(non_blocking= True)
            
            predictions = model(inputs)
            
            loss = criterion(predictions, targets)

            info = analyzer(predictions, targets)
                        
            loss_local = criterion.loss_local(epoch, info)
            
            loss = loss + loss_local

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.TRAIN.CLIP_GRAD, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            log_dic['count'].append(analyzer.count(info))
            log_dic['loss'].append(loss)
            log_dic['grad'].append(grad_norm)

            if cfg.TRAIN.SHOW:
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

    @torch.no_grad()
    def valid(self, epoch):
        # -------------------------------------------
        
        model = self.model
        analyzer = self.analyzer
        criterion = self.criterion
        cfg = self.cfg

        # -------------------------------------------

        model.eval() # 切换模型为预测模式
        log_dic = {'loss':[], 'count':[]}
        
        for i, (inputs, targets, _, _) in enumerate(self.valid_loader):
            
            if cfg.TRAIN.SHOW:
                t = time.time()
                print(f'\r{i}/{len(self.valid_loader)}', end='')

            inputs = inputs.cuda(non_blocking= True)
            targets = targets.cuda(non_blocking= True)
            
            predictions = model(inputs)
            
            loss = criterion(predictions, targets)

            info = analyzer(predictions, targets)
                        
            loss_local = criterion.loss_local(epoch, info)
            
            loss = loss + loss_local

            log_dic['count'].append(analyzer.count(info))
            log_dic['loss'].append(loss)

            if cfg.TRAIN.SHOW:
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

    @torch.no_grad()
    def test(self):
        # -------------------------------------------
        
        logger = self.logger
        model = self.model
        analyzer = self.analyzer
        criterion = self.criterion
        cfg = self.cfg
        test_loader = make_dataset('test')
        save_dir = os.path.join(cfg.DATA.TEST_PATH, cfg.PRED.SAVE_DIR)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        # -------------------------------------------

        logger.info(f'Start testing.')
        
        start_time = time.time()
        
        model.eval() # 切换模型为预测模式
        
        log_dic = {'loss':[], 'count':[]}
        
        # -------------------------------------------
        
        for i, (inputs, targets, filenames, _) in enumerate(test_loader):
            
            if cfg.TRAIN.SHOW:
                t = time.time()
                print(f'\r{i}/{len(self.test_loader)}', end='')

            inputs = inputs.cuda(non_blocking= True)
            targets = targets.cuda(non_blocking= True)
            
            predictions = model(inputs)
            
            loss = criterion(predictions, targets)

            info = analyzer(predictions, targets)
                        
            loss_local = criterion.loss_local(999, info)
            
            loss = loss + loss_local

            log_dic['count'].append(analyzer.count(info))
            log_dic['loss'].append(loss)
                        
            analyzer.to_poscar(predictions, filenames, save_dir)

            if cfg.TRAIN.SHOW:
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
                
        logger.test_info(log_dic)
        
        # -------------------------------------------
        analyzer.rdf.save()
        
        logger.info(f"Spend time: {time.time() - start_time:.2f}s")
        
        logger.info(f'End testing')

    @torch.no_grad()
    def predict(self):
        # -------------------------------------------
        
        logger = self.logger
        model = self.model
        cfg = self.cfg
        rdf = RDF(cfg)
        analyzer = self.analyzer
        predict_loader = make_dataset('predict', cfg)

        # -------------------------------------------

        logger.info(f'Start prediction.')
        
        start_time = time.time()

        model.eval() # 切换模型为预测模式
        
        # -------------------------------------------
        
        for i, (inputs, filenames) in enumerate(predict_loader):
            
            if cfg.TRAIN.SHOW:
                t = time.time()
                print(f'\r{i}/{len(predict_loader)}', end='')

            inputs = inputs.cuda(non_blocking= True)
            
            predictions = model(inputs)
            
            save_dir = os.path.join(cfg.PRED.PATH, cfg.PRED.SAVE_DIR)
            
            analyzer.to_poscar(predictions, filenames, save_dir)
            
            if cfg.TRAIN.SHOW:
                print(f' time: {(time.time() - t):.2f}', end='')
        
        # -------------------------------------------
  
        analyzer.rdf.save()
  
        logger.info(f"Spend time: {time.time() - start_time:.2f}s")
        
        logger.info(f'End prediction')
        

    def save(self, epoch, log_dic):
        
        cfg = self.cfg
        log_count = log_dic['count']
        log_loss = log_dic['loss']
        model = self.model
        logger = self.logger
        elem = cfg.DATA.ELE_NAME
        split = cfg.OTHER.SPLIT
        split = [f"{split[i]}-{split[i+1]}" for i in range(len(split)-1)]

        # -------------------------------------------
        
        save = False
        ele_ACC = []
        for ele in elem:
            ele_ACC.append(sum(log_count[f"{ele}-{i}-ACC"] for i in split))
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
            model_name += "_".join([f"{ele}{ACC.item():.4f}" for ele, ACC in zip(elem, ele_ACC)])
            model_name += f"_{self.best_LOSS:.6f}.pkl"
            model_path = os.path.join(MODEL_SAVE_DIR, model_name)
            
            self.saver.save_new_model(model, model_path, self.device_num > 1)
            
            logger.info(f"Saved a new model: {model_name}")
        else:
            logger.info(f"No model was saved")


