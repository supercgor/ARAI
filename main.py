from utils.tools import Parser, set_seed
from config import get_config
from train import Trainer
from test import Test
from pred import Pred
from tune import Tuner

# 后边正常写你的代码

# ---------------------------------------

def run(mode):
    
    parser = Parser()
    options, _ = parser.parse_args()
    # debug
    cfg = get_config(vars(options))
    
    if mode == 'train':
        trainer = Trainer(cfg)
        trainer.fit()
    elif mode == 'test':
        trainer = Test(cfg)
        trainer.test()
    elif mode == 'predict':
        trainer = Pred(cfg)
        trainer.predict()
    elif mode == 'tune':
        trainer = Tuner(cfg)
        trainer.fit()

if __name__ == '__main__':
    set_seed(1)
    run(mode = "train")

#test