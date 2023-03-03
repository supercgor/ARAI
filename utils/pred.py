import time
import torch

from utils.analyze_data import *
from utils.dataset import make_dataset
from utils.loader import Loader, poscarLoader

class Pred():
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.ml, self.model, self.logger = Loader(cfg)

        self.pl = poscarLoader(f"{cfg.path.data_root}/{cfg.path.dataset}", model_name = cfg.path.checkpoint, lattice=cfg.DATA.REAL_SIZE, elem=cfg.DATA.ELE_NAME)

        self.analyzer = Analyzer(cfg).cuda()

    @torch.no_grad()
    def predict(self):
        # -------------------------------------------
        
        logger = self.logger
        model = self.model
        cfg = self.cfg
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
            
            for filename, x in zip(filenames, predictions):
                self.pl.save4npy(f"{filename}.poscar", x, conf = 0)
                if not os.path.exists(f"{cfg.path.data_root}/{cfg.path.dataset}/npy/{cfg.path.checkpoint}"):
                    os.mkdir(f"{cfg.path.data_root}/{cfg.path.dataset}/npy/{cfg.path.checkpoint}")
                torch.save(x.cpu().numpy(), f"{cfg.path.data_root}/{cfg.path.dataset}/npy/{cfg.path.checkpoint}/{filename}.npy")
                
            if cfg.TRAIN.SHOW:
                print("Finish!")

        # -------------------------------------------
  
        # analyzer.rdf.save()
  
        logger.info(f"Spend time: {time.time() - start_time:.2f}s")
        
        logger.info(f'End prediction')