from yacs.config import CfgNode as CN
import os
from utils.tools import absp

_C = CN()
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Input image size
_C.DATA.IMG_SIZE = 128
# Number of input images
_C.DATA.MAX_Z = 10
# Out put layers
_C.DATA.Z = 9
# Element names
_C.DATA.ELE_NAME = ('O', 'H')
# Real box size
_C.DATA.REAL_SIZE = [25,25,9]
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 32
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 12

# absp root is utils, so use 2 ..
_root_path = absp('../../AFM_3d')
# Train data path
_C.DATA.TRAIN_PATH = _root_path
# Val data path
_C.DATA.VAL_PATH = _root_path
# Test data path
_C.DATA.TEST_PATH = _root_path
# Train file list
_C.DATA.TRAIN_FILE_LIST = os.path.join(_root_path,'all_fileList', 'train.filelist')
# Val file list
_C.DATA.VAL_FILE_LIST = os.path.join(_root_path,'all_fileList', 'valid.filelist')
# Test file list
_C.DATA.TEST_FILE_LIST = os.path.join(_root_path,'all_fileList', 'test.filelist')
# Label path
_C.DATA.LABEL_PATH = os.path.join(_root_path, 'label')

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
_C.TRAIN.EPOCHS = 95
# learning rate
_C.TRAIN.LR = 1e-4
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 10.0
# Max number of models to save
_C.TRAIN.MAX_SAVE = 3
# Show the progress
_C.TRAIN.SHOW = True
# Checkpoint path
_C.TRAIN.CHECKPOINT = "3D_babe.pkl"
# Criterion
_C.TRAIN.CRITERION = CN()
# Factor to increase the loss of positive sample
_C.TRAIN.CRITERION.POS_WEIGHT = (5.0, 5.0)
# Weight of confidence
_C.TRAIN.CRITERION.WEIGHT_CONFIDENCE = 1.5
# Weight of offset in x-axis and y-axis
_C.TRAIN.CRITERION.WEIGHT_OFFSET_XY = 0.5
# Weight of offset in z-axis
_C.TRAIN.CRITERION.WEIGHT_OFFSET_Z = 0.5
# Reduction of offset
_C.TRAIN.CRITERION.REDUCTION = 'mean'
# Enable local loss after epoch
_C.TRAIN.CRITERION.LOCAL = 5
# Decay para
_C.TRAIN.CRITERION.DECAY = [0.9,0.7,0.5,0.3]

# -----------------------------------------------------------------------------
# Other settings
# -----------------------------------------------------------------------------
_C.OTHER = CN()
# Threshold
_C.OTHER.THRESHOLD = 0
# NMS
_C.OTHER.NMS = True
# Split space
_C.OTHER.SPLIT = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]

# -----------------------------------------------------------------------------
# Predict settings
# -----------------------------------------------------------------------------
_C.PRED = CN()
# Prediction data root file
_C.PRED.PATH = _root_path
# Prediction file list
_C.PRED.FILE_LIST = "pre.FileList"
# Prediction result save file dir
_C.PRED.SAVE_DIR = "result"


# -----------------------------------------------------------------------------
# Test settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Path to save test result
_C.TEST.SAVE_PATH = r""
# Show the progress
_C.TEST.SHOW = True
# Whether to open OVITO
_C.TEST.OVITO = False
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
            cfg.TRAIN.CHECKPOINT = absp("../checkpoints/" + options.model)
        if options.dataset is not None:
            _root_path = absp("../../" + options.dataset.split("/")[0])
            if len(options.dataset.split("/")) >= 2:
                _filelist_path = options.dataset.split("/")[1]
            else:
                _filelist_path = ""
            if len(options.dataset.split("/")) >= 3:
                _filelist_name = options.dataset.split("/")[2]
            else:
                _filelist_name = ""
            cfg.DATA.LABEL_PATH = os.path.join(_root_path, 'label')
            if options.mode == "train":
                cfg.DATA.TRAIN_PATH = _root_path
                cfg.DATA.TRAIN_FILE_LIST = os.path.join(_filelist_path, 'train.filelist')
                cfg.DATA.VAL_PATH = _root_path
                cfg.DATA.VAL_FILE_LIST = os.path.join(_filelist_path, 'valid.filelist')
            elif options.mode == "test":
                cfg.DATA.TEST_PATH = _root_path
                if _filelist_name == "":
                    _filelist_name = 'test.filelist'
                cfg.DATA.TEST_FILE_LIST = os.path.join(_filelist_path, _filelist_name)
                print(cfg.DATA.TEST_FILE_LIST)
            elif options.mode == "predict":
                cfg.PRED.PATH = _root_path
                if _filelist_name == "":
                    _filelist_name = 'pre.filelist'
                cfg.PRED.FILE_LIST = os.path.join(_filelist_path, _filelist_name)
                print(cfg.PRED.FILE_LIST)

            
        if options.log_name is not None:
            cfg.OTHER.LOG = options.log_name
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
