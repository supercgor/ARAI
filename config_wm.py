from utils.tools import CfgNode as CN

_C = CN()
# -----------------------------------------------------------------------------
# Load settings
# -----------------------------------------------------------------------------
_C.path = CN()
# the root of all file
_C.path.root = '/gpfs/share/home/2000012508/ML2023'
# datasets path
_C.path.data_root = '/gpfs/share/home/2000012508/ML2023/data'
# checkpoints path
_C.path.check_root = '/gpfs/share/home/2000012508/ML2023/data/model'
# Save file dir
_C.path.save_dir = ""
# OVITO
_C.path.ovito = ""


# -----------------------------------------------------------------------------
# Default settings
# -----------------------------------------------------------------------------
_C.setting = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.setting.batch_size = 16
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.setting.pin_memory = True
# Number of data loading threads
_C.setting.num_workers = 12
# 0 for using one GPU or list for Parallel device idx
_C.setting.device = [0,1]
# Training epochs
_C.setting.epochs = 50
# Enable local loss after epoch
_C.setting.local_epoch = 999
# learning rate
_C.setting.learning_rate = 1e-4
# Clip gradient norm
_C.setting.clip_grad = 10.0
# Max number of models to save
_C.setting.max_save = 3
# Show the progress
_C.setting.show = True
# Split space
_C.setting.split = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.model = CN()
# checkpoint name
_C.model.checkpoint = ""
# use net
_C.model.net = "UNet3D"
# the init channels number
_C.model.channels = 32
# the size of input and output. (Z, X, Y)
_C.model.inp_size = (10, 128, 128)
_C.model.out_size = (9, 32, 32)
# Threshold
_C.model.threshold = 0.0
# NMS
_C.model.nms = True

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.data = CN()
# use dataset
_C.data.dataset = 'afm3d'
# How many images will be used
_C.data.img_use = 10
# Element names
_C.data.elem_name = ('O', 'H')
# Real box size (Z, X, Y)
_C.data.real_size = [9, 25, 25]
# file list
_C.data.train_filelist = 'prism_train.filelist'
_C.data.valid_filelist = 'prism_valid.filelist'
_C.data.test_filelist = 'test.filelist'
_C.data.pred_filelist = 'pred.filelist'

# -----------------------------------------------------------------------------
# Criterion settings
# -----------------------------------------------------------------------------
# Criterion
_C.criterion = CN()
# Factor to increase the loss of positive sample
_C.criterion.pos_weight = (5.0, 5.0)
# Weight of confidence
_C.criterion.weight_confidence = 1.0
# Weight of offset in x-axis and y-axis
_C.criterion.weight_offset_xy = 0.5
# Weight of offset in z-axis
_C.criterion.weight_offset_z = 0.5
# Reduction of offset
_C.criterion.reduction = 'mean'
# Decay para
_C.criterion.decay = [0.9, 0.7, 0.5, 0.3, 0.1, 0.05]

# -----------------------------------------------------------------------------
# Don't fill any thing here, this part is automatically filled
# -----------------------------------------------------------------------------
_C.model.best = ""
_C.setting.tag = ""

_C.freeze()


def get_config(options={}):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    cfg = _C.clone()
    cfg.defrost()
    cfg.merge_from_dict(options)

    return cfg
