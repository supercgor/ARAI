from utils.tools import Parser
from config import get_config
from utils.train import Trainer
from utils.test import Test
from utils.pred import Pred

# ---------------------------------------

def run():
    
    parser = Parser()
    options, args = parser.parse_args()
    
    # debug
    options.mode = 'train'

    cfg = get_config(options)

    
    
    if options.mode == 'train':
        trainer = Trainer(cfg)
        trainer.fit()
    elif options.mode == 'test':
        trainer = Test(cfg)
        trainer.test()
    elif options.mode == 'predict':
        trainer = Pred(cfg)
        trainer.predict()
    
if __name__ == '__main__':
    run()

#test