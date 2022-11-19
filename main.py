from utils.tools import Parser
from config_9A import get_config
from utils.train import Trainer

# ---------------------------------------

def run():
    
    parser = Parser()
    options, args = parser.parse_args()
    
    # debug
    options.mode = 'train'

    cfg = get_config(options)

    trainer = Trainer(cfg)
    
    if options.mode == 'train':
        trainer.fit()
    elif options.mode == 'test':
        trainer.test()
    elif options.mode == 'predict':
        trainer.predict()
    
if __name__ == '__main__':
    run()

#test