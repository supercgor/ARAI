#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filename: const.py
# modified: 2022-11-06

from datetime import datetime
from utils.tools import mkdir, absp
import os

DATE = datetime.now().strftime("%y-%m-%d-%H")

MODEL_DIR = absp("../checkpoints/")
LOG_DIR = absp("../log/")
PROJ_DIR = absp("../")
MODEL_SAVE_DIR = os.path.join(MODEL_DIR,DATE)

mkdir(LOG_DIR)
mkdir(MODEL_DIR)
mkdir(MODEL_SAVE_DIR)