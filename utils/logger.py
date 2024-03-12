import os
import logging
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

def get_logger():
    logger = logging.getLogger()
    return logger
    
def get_tensorboard_logger(log_dir):
    return SummaryWriter(log_dir=os.path.join(log_dir, 'runs'))

def log_to_csv(path, **kwargs):
    df = pd.DataFrame([kwargs])
    if not os.path.isfile(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode='a', header=False, index=False)
