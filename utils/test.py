import time
import torch

from .analyze_data import *
from .criterion import Criterion
from .dataset import make_dataset
from .loader import Loader, poscarLoader
from .tools import condense

class Test():
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.ml, self.model, self.logger = Loader(cfg, make_dir = False)
        
        self.pl = poscarLoader(f"{cfg.path.data_root}/{cfg.path.dataset}", model_name = cfg.path.checkpoint, lattice=cfg.DATA.REAL_SIZE, elem=cfg.DATA.ELE_NAME)
        self.analyzer = Analyzer(cfg).cuda()

    @torch.no_grad()
    def test(self):
        # -------------------------------------------
        
        logger = self.logger
        model = self.model
        analyzer = self.analyzer
        cfg = self.cfg
        criterion = Criterion(cfg,cfg.TRAIN.CRITERION.LOCAL).cuda()
        test_loader = make_dataset('test', cfg)
 
        # -------------------------------------------

        logger.info(f'Start testing.')
        
        start_time = time.time()
        
        model.eval() # 切换模型为预测模式
        
        log_dic = {'loss':[], 'count':[]}
        
        # -------------------------------------------
        
        for i, (inputs, targets, filenames, _) in enumerate(test_loader):
            
            if cfg.TRAIN.SHOW:
                t = time.time()
                print(f'\r{i}/{len(test_loader)}', end='')

            inputs = inputs.cuda(non_blocking= True)
            targets = targets.cuda(non_blocking= True)
            
            predictions = model(inputs)
            
            loss = criterion(predictions, targets)

            info = analyzer(predictions, targets)
                                    
            log_dic['count'].append(analyzer.count(info))
            log_dic['loss'].append(loss)
            
            for filename, x, t in zip(filenames, predictions, targets):
                self.pl.save4npy(f"{filename}.poscar", x)
                self.pl.save4npy(f"{filename}_ref.poscar", t, NMS=False)
                
            if cfg.TRAIN.SHOW:
                print("Finish!")
        
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
        # analyzer.rdf.save()
        
        logger.info(f"Spend time: {time.time() - start_time:.2f}s")
        
        logger.info(f'End testing')