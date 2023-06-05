from . import CfgNode as CN

_C = CN()
# -----------------------------------------------------------------------------
# Load settings
# -----------------------------------------------------------------------------
_C.path = CN()
_C.path.data_root = '../data' # datasets path
_C.path.check_root = './model/pretrain' # checkpoints path


# -----------------------------------------------------------------------------
# Default settings
# -----------------------------------------------------------------------------
_C.setting = CN()
_C.setting.batch_accumulation = 1
_C.setting.batch_size = 1 # Batch size for a single GPU, could be overwritten by command line argument
_C.setting.pin_memory = True # Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.setting.num_workers = 6 # Number of data loading threads
_C.setting.device = (0,) # 0 for using one GPU or list for Parallel device idx, no cpu: []
_C.setting.epochs = 50 # Training epochs
_C.setting.lr = 3.0e-4 # learning rate
_C.setting.clip_grad = 15.0 # Clip gradient norm
_C.setting.max_save = 3 # Max number of models to save
_C.setting.split = (0.0, 3.0) # Split space

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.model = CN()
_C.model.checkpoint = "tune_UNet_strong_baseline" # checkpoint name
# use net
_C.model.fea = "tune_unet_CP07_LOSS0.0549.pkl"
_C.model.reg = "tune_reg_CP07_LOSS0.0549.pkl"
_C.model.cyc = ""
_C.model.channels = 32 # the init channels number
_C.model.inp_size = (16, 128, 128) # the size of input. (Z, X, Y)
_C.model.out_size = (4, 32, 32) # the size of output (Z, X, Y)
_C.model.threshold = 0.5 # Threshold
_C.model.nms = True # NMS

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.data = CN()
_C.data.dataset = 'bulkice' # use dataset "smallice", "bulkice", "bulkexp", ...
_C.data.img_use = 10 # How many images will be used
_C.data.elem_name = ('O', 'H') # Element names, have to be place in order
_C.data.real_size = (3, 25, 25) # Real box size (Z, X, Y)

# -----------------------------------------------------------------------------
# Criterion settings
# -----------------------------------------------------------------------------
# Criterion
_C.criterion = CN()
_C.criterion.pos_weight = (5.0, 5.0) # Factor to increase the loss of positive sample
_C.criterion.w_c = 1.0 # Weight of confidence
_C.criterion.w_xy = 0.5 # Weight of offset in x-axis and y-axis
_C.criterion.w_z = 0.5 # Weight of offset in z-axis

_C.freeze()


def get_config(options={}):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    cfg = _C.clone()
    cfg.defrost()
    cfg.merge_from_dict(options)

    return cfg
