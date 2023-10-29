from .logger import get_logger
from .model import model_save, model_load, model_structure
import torch

import random
import numpy as np
from .metrics import Analyser, parallelAnalyser, metStat, ConfusionCounter, ConfusionRotate
from .criterion import *
from . import const, library, poscar, xyz
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    return

set_seed(0)