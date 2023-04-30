from utils import set_seed
from config.config import get_config
from utils.train import Trainer
import torch

def run(mode):
    
    # debug
    cfg = get_config()
    
    if mode == 'train':
        trainer = Trainer(cfg)
        trainer.fit()

if __name__ == '__main__':
    set_seed(1)
    torch.set_printoptions(precision=6,sci_mode=False)
    run(mode = "train") 

#test