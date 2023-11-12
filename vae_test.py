import os, hydra, time, h5py, torch
import pandas as pd
import numpy as np
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf, DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, get_rank, destroy_process_group, is_initialized
import model
import utils
from dataset.dataset import Point_Grid_Dataset_hdf, Point_Grid_Dataset_folder

user = os.environ.get('USER') == "supercgor"
config_name = "vae44_local" if user else "vae44_wm"
model_load_path = "./outputs/vae_CP32_L0.4382.pkl"
dataset_load_path = "../data/bulkexp/result/unet_tune_v1_repair_2"
log_every = 1 if user else 25



def key_filter(key):
    import re
    return True
    #return True if "HDA" in key or "ss" in key else False
    return re.match(r"T\d{1,3}_\d{1,5}", key) is not None or "icehup" in key

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if torch.cuda.device_count() > 1:
        init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        pass
    
class Trainer():
    def __init__(self, rank, cfg, model, Testdata, TestLoader, log, tblog):
        self.work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
        self.cfg = cfg
        if is_initialized():
            self.rank = get_rank()
        else:
            self.rank = 0
        self.gpu_id = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.gpu_id)
        if is_initialized():
            DDP(self.model, device_ids=[self.gpu_id]) 
            
        self.pos_weight = torch.tensor([self.cfg.criterion.pos_weight]).to(self.gpu_id)
        self.Analyser = utils.parallelAnalyser(real_size=(25,25,16), split = self.cfg.dataset.split).to(self.gpu_id)
 
        self.TestData = Testdata
        self.TestLoader = TestLoader
        
        self.ConfusionCounter = utils.ConfusionRotate()
        self.LostStat = utils.metStat()
        self.LostConfidenceStat = utils.metStat()
        self.LostPositionStat = utils.metStat()
        self.LostRotationStat = utils.metStat()
        self.LostVAEStat = utils.metStat()
        self.GradStat = utils.metStat()
        self.RotStat = utils.metStat()
            
        self.log = log
        self.tblog = tblog
        self.save_paths = []
        self.best = np.inf
        
    def fit(self):
        epoch_start_time = time.time()
        
        if False:
            self.test_one_epoch(0)
        else:
            self.pred2_one_epoch(0)
       
    @torch.no_grad()
    def test_one_epoch(self, epoch, log_every: int = 25) -> tuple[torch.Tensor]:
        all_data = pd.DataFrame()
        self.model.eval()
        self.LostStat.reset()
        self.LostConfidenceStat.reset()
        self.LostPositionStat.reset()
        self.LostRotationStat.reset()
        self.LostVAEStat.reset()
        self.ConfusionCounter.reset()
        self.RotStat.reset()
        for i, (filenames, inps, targs, embs) in enumerate(self.TestLoader):
            inps = inps.to(self.device)
            targs = targs.to(self.device)
            embs = embs.to(self.device)
            preds, mu, logvar = self.model(inps, embs)
            loss_wc, loss_pos, loss_r, loss_vae = self.Criterion(preds, targs, mu, logvar)
            loss = loss_wc + loss_pos + loss_r + loss_vae
            self.LostStat.add(loss)
            self.LostConfidenceStat.add(loss_wc)
            self.LostPositionStat.add(loss_pos)
            self.LostRotationStat.add(loss_r)
            self.LostVAEStat.add(loss_vae)
            feats = torch.cat([mu, logvar], dim=-1).flatten(1).detach().cpu().numpy()
            names = []
            for i in range(len(filenames)):
                fn = filenames[i]
                fn = fn.split("_")
                name = ""
                if "d" in fn:
                    name += f"d,{fn[3]},{fn[-1]}"
                else:
                    name += f"h,{fn[2]},{fn[-1]}"
                names.append(name)
            print(names)
            df_batch_predictions = pd.DataFrame(feats, index=names)
            
            all_data = pd.concat([all_data, df_batch_predictions])
            
            CMS = self.Analyser(preds, targs)
            self.ConfusionCounter.add(CMS[...,:3])
            self.RotStat.add(CMS[...,3])
            if self.rank == 0 and i % log_every == 0:
                self.log.info(f"Epoch {epoch:2d} | Iter {i:5d}/{len(self.TestLoader):5d} | Loss {loss:.2e}")
                self.tblog.add_scalar("Test/Loss", loss, epoch * len(self.TestLoader) + i)
                self.tblog.add_scalar("Test/LossWC", loss_wc, epoch * len(self.TestLoader) + i)
                self.tblog.add_scalar("Test/LossPos", loss_pos, epoch * len(self.TestLoader) + i)
                self.tblog.add_scalar("Test/LossRot", loss_r, epoch * len(self.TestLoader) + i)
                self.tblog.add_scalar("Test/LossVAE", loss_vae, epoch * len(self.TestLoader) + i)
            
        all_data.to_csv('all_output.csv')
        losses = [self.LostStat.calc(),self.LostConfidenceStat.calc(), self.LostPositionStat.calc(), self.LostRotationStat.calc(), self.LostVAEStat.calc()]
        cms = self.ConfusionCounter.calc()
        rot = self.RotStat.calc()
        
        return losses, cms, rot
        
    @torch.no_grad()
    def pred_one_epoch(self, epoch, log_every: int = 25) -> tuple[torch.Tensor]:
        self.model.eval()
        for i, (filenames, inps, targs, embs) in enumerate(self.TestLoader):
            inps = inps.to(self.device)
            embs = embs.to(self.device)
            
            out = [inps]
            preds = inps
            for j in range(self.cfg.pred_loop):
                preds, mu, logvar = self.model(preds, embs)
                preds = torch.stack([utils.library.box2box(pred, real_size=(25.0, 25.0, 4.0), threshold=0.0, nms=True, sort=True, cutoff=2.0) for pred in preds], dim = 0)
                out.insert(0, preds)
                #out.append(preds)
                            
            preds = torch.cat(out, dim=3)# B X Y Z*L 10
            
            for b in range(preds.shape[0]):
                pred = preds[b]
                filename = filenames[b]
                pred = pred.detach().cpu()
                conf, pos, r = utils.library.box2orgvec(pred, 0.0, 2.0, (25.0, 25.0, 4.0 * (self.cfg.pred_loop+1)), True, True)
                r = r.view(-1, 9)[:, :6]
                #pred[...,2] = 16.0 - pred[...,2]
                pred = utils.library.encodeWater(np.concatenate([pos, r], axis = -1)).reshape(-1, 3, 3)
                # pred = utils.library.makewater(pos, r)
                utils.xyz.write(f"{self.work_dir}/{filename}.xyz", np.array([["O", "H", "H"]], dtype=np.str_).repeat(len(pred), axis=0), pred)
                # conf, pos, r = utils.functional.box2orgvec(targs[b].detach().cpu(), 0.0, 2.0, (25.0, 25.0, 8.0), False, False)
                # targ = utils.functional.makewater(pos, r)
                # utils.xyz.write(f"{self.work_dir}/{filename}_targ.xyz", np.array([["O", "H", "H"]], dtype=np.str_).repeat(len(targ), axis=0), targ)
                # conf, pos, r = utils.functional.box2orgvec(inps[b].detach().cpu(), 0.0, 2.0, (25.0, 25.0, 4.0), False, False)
                # inp = utils.functional.makewater(pos, r)
                # utils.xyz.write(f"{self.work_dir}/{filename}_inp.xyz", np.array([["O", "H", "H"]], dtype=np.str_).repeat(len(inp), axis=0), inp)
                
            if self.rank == 0:
                self.log.info(f"Epoch {epoch:2d} | Iter {i:5d}/{len(self.TestLoader):5d}")
            
    @torch.no_grad()
    def pred2_one_epoch(self, epoch, log_every: int = 25) -> tuple[torch.Tensor]:
        self.model.eval()
        for i in range(len(self.TestData)):
            filename, inp = self.TestData[i]
        
            pred, mu, nu = self.model(inp.to(self.gpu_id).unsqueeze(0))

            pred = pred.squeeze(0).detach().cpu()  
            pred[...,:inp.shape[2],:] = inp
            conf, pos, r = utils.library.box2orgvec(pred, 0.5, 2.0, (25.0, 25.0, 16.0), True, True)
            r = r.view(-1, 9)[:, :6]

            pos[...,2] = 16.0 - pos[...,2]
            
            pred = utils.library.encodeWater(np.concatenate([pos, r], axis = -1)).reshape(-1, 3, 3)
            utils.xyz.write(f"{self.work_dir}/{filename}.xyz", np.array([["O", "H", "H"]], dtype=np.str_).repeat(len(pred), axis=0), pred)
            
            if self.rank == 0 and i % log_every == 0:
                self.log.info(f"Epoch {epoch:2d} | Iter {i:5d}/{len(self.TestLoader):5d}")
         
def inp_transform(inp: torch.Tensor):
    # B X Y Z C -> B C Z X Y
    inp = inp.permute(0, 4, 3, 1, 2)
    return inp

def out_transform(inp: torch.Tensor):
    # B C Z X Y -> B X Y Z C
    inp = inp.permute(0, 3, 4, 2, 1).sigmoid()
    conf, pos, rotx, roty = torch.split(inp, [1, 3, 3, 3], dim = -1)
    pos = pos.sigmoid()
    c1 = rotx / torch.norm(rotx, dim=-1, keepdim=True)    
    c2 = roty - (c1 * roty).sum(-1, keepdim=True) * c1
    c2 = c2 / torch.norm(c2, dim=-1, keepdim=True)
    return torch.cat([conf, pos, c1, c2], dim=-1)
            
                
def load_train_objs(rank, cfg: DictConfig):
    work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    
    log = utils.get_logger(f"Rank {rank}")
    tblog = SummaryWriter(f"{work_dir}/runs")
    
    net = getattr(model, cfg.model.net)(**cfg.model.params)
    
    net.apply_transform(inp_transform, out_transform)

    log.info(f"Network parameters: {sum([p.numel() for p in net.parameters()])}")
    
    if cfg.model.checkpoint is None:
        raise ValueError("No checkpoint is provided.")
    else:
        missing = utils.model_load(net, cfg.model.checkpoint, True)
        log.info(f"Load parameters from {cfg.model.checkpoint}")
        print(missing)
            
    TestDataset = Point_Grid_Dataset_folder(cfg.dataset.test_path, label_size = None, flipz = 4)    
    
    return net, TestDataset, log, tblog

def prepare_dataloader(test_data, cfg: DictConfig): 
    TestLoader = DataLoader(test_data,
                            batch_size=cfg.setting.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=cfg.setting.pin_memory,
                            )
    
    return TestLoader


@hydra.main(config_path="config", config_name=config_name, version_base=None) # hydra will automatically relocate the working dir.
def main(cfg):
    if user:
        rank = 0
    else:
        if "LOCAL_RANK" in os.environ:
            rank = int(os.environ["LOCAL_RANK"])
        else:
            rank = 0
        world_size = torch.cuda.device_count()
        ddp_setup(rank, world_size)
    
    if model_load_path is not None:
        cfg.model.checkpoint = model_load_path
    if dataset_load_path is not None:
        cfg.dataset.test_path = dataset_load_path
    model, TestDataset, log, tblog = load_train_objs(rank, cfg)
    TestLoader = prepare_dataloader(TestDataset, cfg)
    trainer = Trainer(rank, cfg, model, TestDataset, TestLoader, log, tblog)
    trainer.fit()
    
    if is_initialized():
        destroy_process_group()

if __name__ == "__main__":
    main()
    
