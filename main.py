from utils import set_seed
from utils.train import Trainer
from utils.cyctune import Tuner
import torch

import os
user = os.environ.get("USER") == "supercgor"
if user:
    from config.config import get_config
else:
    from config.wm import get_config

def run(mode):
    
    # debug
    cfg = get_config()
    
    if mode == 'train':
        trainer = Trainer(cfg)
        trainer.fit()
    if mode == 'tune':
        Tuner(cfg).fit()
    if mode == 'diffuse':
        Diffuse(cfg).fit()

if __name__ == '__main__':
    set_seed(1)
    torch.set_printoptions(precision=6,sci_mode=False)
    run(mode = "tune") 

#test