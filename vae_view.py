import os, hydra, time, h5py, torch, re
import tqdm
import pandas as pd
import numpy as np
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf, DictConfig
import model
import utils
from ase import io
from ase.visualize import view
from dataset.dataset import PointGridDataset

user = os.environ.get('USER') == "supercgor"
config_name = "vae44_local" if user else "vae44_wm"
log_every = 1 if user else 25
    
class Trainer():
    def __init__(self, rank, cfg, model, Testdata, log, tblog):
        self.work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
        self.cfg = cfg
        self.rank = 0
        self.gpu_id = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.gpu_id)

        self.Analyser = utils.parallelAnalyser(real_size=(25,25,16), split = self.cfg.dataset.split).to(self.gpu_id)
 
        self.TestData = Testdata
            
        self.log = log
        self.tblog = tblog
        self.save_paths = []
        self.best = np.inf

    @torch.no_grad()
    def pred(self, epoch, log_every: int = 25, repeat = 20, enc = False):
        self.model.eval()
        pbar = tqdm.tqdm(self.TestData)
        all_mu = []
        for i, (filename, inp) in enumerate(pbar):
            pbar.set_description(f"Iter {i:5d}/{len(self.TestData):5d} | {filename}")
            condition = inp[None,:,:,:2].to(self.gpu_id, non_blocking = True)
            if enc:
                x = inp[None, :, :, 2:].to(self.gpu_id, non_blocking = True)
                out, mu, _ = self.model(x, condition)
                all_mu.append(mu)

        all_mu = torch.cat(all_mu, dim=0)
        all_mu = all_mu.cpu().numpy()
        np.save(f"{self.work_dir}/mu.npy", all_mu)
        
@hydra.main(config_path="config", config_name=config_name, version_base=None) # hydra will automatically relocate the working dir.
def main(cfg):
    
    work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']

    model_load_path = "./outputs/save_params/20240304_pretrain_start5_kl8.pkl"
    # dataset_load_path = "../data/bulkexp/result/unet_tune_v1_repair_2"
    dataset_load_path = "../data/ice_16A_R_hup_low_T_test.hdf5"
    # dataset_load_path = "../data/bulkexp/result/labeled"
    

    if model_load_path is not None:
        cfg.model.checkpoint = model_load_path
    elif cfg.model.checkpoint is None:
        raise ValueError("No checkpoint is provided.")
    if dataset_load_path is not None:
        cfg.dataset.test_path = dataset_load_path
        
    
    log = utils.get_logger(f"Rank 0")
    tblog = SummaryWriter(f"{work_dir}/runs")
    
    net = getattr(model, cfg.model.net)(**cfg.model.params)
    missing = utils.model_load(net, cfg.model.checkpoint, True)
    log.info(f"Load parameters from {cfg.model.checkpoint}")
    print(missing)
    
    log.info(f"Network parameters: {sum([p.numel() for p in net.parameters()])}")
        
    # TestDataset = PointGridDataset(cfg.dataset.test_path, grid_size = [25, 25, 4], cell_size = [25.0, 25.0, 4.0], flip = [False, False, False], noise_position=0.1)    
    # TestDataset = PointGridDataset(cfg.dataset.test_path, grid_size = [25, 25, 8], position_offset = [0, 0, 0], reflect=[False, False, True], random_transform=False)
    TestDataset = PointGridDataset(cfg.dataset.test_path, grid_size = [25, 25, 8], random_transform=False)

    # TestDataset = Point_Grid_Dataset_hdf("../data/ice_16A_R_hup_test.hdf5", grid_size = [25, 25, 16], noise_position= 0, z_off= 0, remove_ratio=0, extra_noise_for_first_layer=0, key_filter=lambda x: "T260" in x)

    trainer = Trainer(0, cfg, net, TestDataset, log, tblog)
    
    trainer.pred(0, enc=True)
    

if __name__ == "__main__":
    main()
    
