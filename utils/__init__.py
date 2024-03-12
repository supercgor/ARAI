from .logger import *
from .model import model_save, model_load, model_structure
import torch

from hydra.core import hydra_config as hc
import random
import numpy as np
from .metrics import parallelAnalyser, metStat, ConfusionCounter, ConfusionRotate, MetricCollection
from .confusion import *
from .criterion import *
from . import const, lib, poscar, xyz
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    return

def workdir():
    return hc.HydraConfig.get().runtime.output_dir
# set_seed(0)