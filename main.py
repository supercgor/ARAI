from utils.tools import Parser, set_seed
from config import get_config
from utils.train import Trainer
from utils.test import Test
from utils.pred import Pred

# ---------------------------------------

def run():
    
    parser = Parser()
    options, _ = parser.parse_args()
    # debug
    mode = "train"
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
    
if __name__ == '__main__':
    set_seed(1)
    run()

#test