from yacs.config import CfgNode as CN
import os
from utils.tools import absp

_C = CN()
# -----------------------------------------------------------------------------
# Load settings
# -----------------------------------------------------------------------------
_C.path = CN()
# the root of all file
_C.path.root = '/home/supercgor/gitfile/'
# datasets path
_C.path.data_root = '/home/supercgor/gitfile/data'
# checkpoints path
_C.path.check_root = '/home/supercgor/gitfile/data/model'
# use dataset
_C.path.dataset = 'bulkice'
# Train file list
_C.path.train_filelist = 'train.filelist'
# Val file list
_C.path.valid_filelist = 'valid.filelist'
# Test file list
_C.path.test_filelist = 'test.filelist'
# Test file list
_C.path.pred_filelist = 'pred.filelist'
# checkpoint name
_C.path.checkpoint = "3A_ref"
# Save file dir
_C.path.save_dir = "None"
# OVITO
_C.path.ovito = "None"

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Input image size
_C.DATA.IMG_SIZE = 128
# Number of input images
_C.DATA.MAX_Z = 10
# Out put layers
_C.DATA.Z = 4
# Element names
_C.DATA.ELE_NAME = ('O', 'H')
# Real box size
_C.DATA.REAL_SIZE = [25,25,3]
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 2
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 0

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.CHANNELS = 32

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# 0 for using one GPU or list for Parallel device idx 
_C.TRAIN.DEVICE = [0,1]
# Training epochs
_C.TRAIN.EPOCHS = 50
# learning rate
_C.TRAIN.LR = 1e-4
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 10.0
# Max number of models to save
_C.TRAIN.MAX_SAVE = 3
# Show the progress
_C.TRAIN.SHOW = True
# Criterion
_C.TRAIN.CRITERION = CN()
# Factor to increase the loss of positive sample
_C.TRAIN.CRITERION.POS_WEIGHT = (5.0, 5.0)
# Weight of confidence
_C.TRAIN.CRITERION.WEIGHT_CONFIDENCE = 1.0
# Weight of offset in x-axis and y-axis
_C.TRAIN.CRITERION.WEIGHT_OFFSET_XY = 0.5
# Weight of offset in z-axis
_C.TRAIN.CRITERION.WEIGHT_OFFSET_Z = 0.5
# Reduction of offset
_C.TRAIN.CRITERION.REDUCTION = 'mean'
# Enable local loss after epoch
_C.TRAIN.CRITERION.LOCAL = 999
# Decay para
_C.TRAIN.CRITERION.DECAY = [0.9,0.7,0.5,0.3,0.1,0.05]

# -----------------------------------------------------------------------------
# Other settings
# -----------------------------------------------------------------------------
_C.OTHER = CN()
# Threshold
_C.OTHER.THRESHOLD = 0
# NMS
_C.OTHER.NMS = True
# Split space
_C.OTHER.SPLIT = [0.0,3.0]

# -----------------------------------------------------------------------------
# Test settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Show the progress
_C.TEST.SHOW = True
# Number of the best and the worst prediction
_C.TEST.N = 5

_C.freeze()

def get_config(options = None):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    cfg = _C.clone()
    cfg.defrost()
    if options is not None:
        if options.batch_size is not None:
            cfg.DATA.BATCH_SIZE = options.batch_size
        if options.model is not None:
            cfg.path.checkpoint = options.model
        if options.dataset is not None:
            cfg.path.dataset = options.dataset
        if options.train_filelist is not None:
            cfg.path.train_filelist = options.train_filelist
        if options.valid_filelist is not None:
            cfg.path.valid_filelist = options.valid_filelist
        if options.test_filelist is not None:
            cfg.path.test_filelist = options.test_filelist
        if options.pred_filelist is not None:
            cfg.path.pred_filelist = options.pred_filelist
        if options.worker is not None:
            cfg.DATA.NUM_WORKERS = options.worker
        if options.gpu is not None:
            gpu = list(map(int,options.gpu.split(",")))
            cfg.TRAIN.DEVICE = gpu
        if options.local_epoch is not None:
            cfg.TRAIN.CRITERION.LOCAL = options.local_epoch
        if options.epoch is not None:
            cfg.TRAIN.EPOCH = options.epoch

    cfg.freeze()
    return cfg
