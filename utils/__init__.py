#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filename: __init__.py
# modified: 2022-11-10

__version__ = "0.0.0"
__date__    = "2022.11.05"
__author__  = "Xu_Group"

import torch
from torch import nn
import random
import numpy as np
from .metrics import *

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    return