import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import einops

from model import UNetModel
from model.diffusion import GaussianDiffuser
from model.utils import basicParallel

from datasets.dataset import make_dataset
from datasets.poscar import poscar
from utils.logger import Logger
from utils.loader import Loader
from utils.metrics import metStat, analyse_cls
from utils.schedular import Scheduler
from demo import plot as vis

from collections import OrderedDict
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc("image", cmap = "gray")

class Diffuse():
    def __init__(self, cfg):
        self.cfg = cfg
        
        assert cfg.setting.device != [], "No device is specified!"
        
        self.load_dir, self.work_dir = Loader(cfg, make_dir=True)
        
        self.logger = Logger(path=self.work_dir,
                             elem=cfg.data.elem_name, split=cfg.setting.split)
        
        self.tb_writer = SummaryWriter(
            log_dir=f"{self.work_dir}/runs/Difuse_{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        
        log = []
        
        
        self.net = CombineModel().cuda()
        
        try:
            paths = {"fea": f"{self.load_dir}/{cfg.model.fea}",
                    "neck": f"{self.load_dir}/{cfg.model.neck}", 
                    "head": f"{self.load_dir}/{cfg.model.head}"}
            match_list = self.net.load(paths, pretrained=True)
            match_list = "\n".join(match_list)
            log.append(f"Feature: Load parameters from {self.load_dir}")
            log.append(f"\n{match_list}")
        except FileNotFoundError:
            raise f"No network has been found in {self.load_dir}, stop training"
        
        self.denoise = UNetModel(image_size = (8, 32, 32), 
                             in_channels = 8,
                             out_channels = 8,
                             num_res_blocks = 2,
                             model_channels = 32,
                             channel_mult = (1, 2, 4, 8),
                             attention_resolutions = (8, 4),
                             num_heads = 4,
                             dims = 3,
                             reference_channels = 32).cuda()
        
        try:
            path = f"{self.load_dir}/{cfg.model.denoise}"
            match_list = self.denoise.load(path, pretrained = True)
            match_list = "\n".join(match_list)
            log.append(f"Denoise: Load parameters from {self.load_dir}")
            log.append(f"\n{match_list}")
        except (FileNotFoundError, IsADirectoryError):
            # self.denoise.init()
            log.append(f"No network is loaded, start a new model: {self.denoise.name}")

        if len(cfg.setting.device) >= 2:
            self.net.parallel(devices_ids=cfg.setting.device)
            self.denoise = basicParallel(self.denoise, device_ids=cfg.setting.device)
        
        self.analyse = analyse_cls(threshold=cfg.model.threshold).cuda()
        
        self.net.eval()
        
        self.diffusion = GaussianDiffuser(T=256, schedule='linear').cuda()
        
        self.OPT = torch.optim.Adam(self.denoise.parameters(), lr = 3e-4)
        
        self.SCHEDULER = Scheduler(self.OPT, warmup_steps=0, decay_factor=90000)
        
        for l in log:
            self.logger.info(l)
            
    def fit(self):
        self.train_loader = make_dataset('train', self.cfg)
        self.best = {'loss': metStat(mode="min")}
        self.best_met = 9999

        # --------------------------------------------------
        self.logger.info(f'Start diffusing.')

        for epoch in range(1, self.cfg.setting.epochs + 1):
            epoch_start_time = time.time()

            log_train_dic = self.train(epoch)

            self.save(epoch, log_train_dic)

            self.logger.info(
                f"Spend time: {time.time() - epoch_start_time:.2f}s")

            self.logger.info(f'End training epoch: {epoch:0d}')

        # --------------------------------------------------

        self.logger.info(f'End tunning.')
        
    @staticmethod
    def get_dict():
        T_dict = OrderedDict(
            Grad =metStat(mode="mean"),
            Loss =metStat(mode="mean"),
        )
        
        return T_dict
    
    def train(self, epoch):
        accu = self.cfg.setting.denoise_accumulation
        max_iter = len(self.train_loader) // accu
        it_loader = iter(self.train_loader)
        
        T_dict = self.get_dict()
        
        pbar = tqdm(total=max_iter - 1,
                    desc=f"Epoch {epoch} - Diffuse", position=0, leave=True, unit='it')
        
        i = 0
        
        self.denoise.train()
        
        while i < max_iter:
            step = (epoch-1) * max_iter + i
            self.OPT.zero_grad()
            for t in range(accu):
                imgs, gt_box, filenames = next(it_loader)
                imgs = imgs.cuda(non_blocking=True)
                gt_box = gt_box.cuda(non_blocking=True)
                
                x0 = einops.rearrange(gt_box, "B Z X Y E C -> B (E C) Z X Y").cuda(non_blocking=True)

                with torch.no_grad():
                    ref = self.net(imgs, require_head=False)
                
                t = torch.randint(1, (self.diffusion.T+1).item(), (x0.shape[0],), device=x0.device)
                
                xt, epsilon = self.diffusion.sample(x0, t)

                out = self.denoise(xt, t, ref = ref)
                
                loss = F.mse_loss(out, epsilon)
                #print(out)
                T_dict['Loss'].add(loss)
                loss.backward()
                
            grad = nn.utils.clip_grad_norm_(self.denoise.parameters(), 9999, error_if_nonfinite = True)
            
            self.OPT.step()
            self.SCHEDULER.step()
            
            T_dict['Grad'].add(grad)
                
            pbar.set_postfix(L= T_dict['Loss'].last, G= T_dict['Grad'].last)
                
            if step > 499 and step % 100 == 0:
                self.tb_writer.add_images("Train/In IMG", imgs[0].permute(1,0,2,3), step)
                self.denoise.eval()
                pd_box = self.diffusion.inverse(self.denoise, shape=(8, 8, 32, 32), prior=ref[(0,), ...])
                self.denoise.train()
                self.tb_writer.add_image("Train/OUT BOX", vis.label2img(pd_box.clamp(0,1), format="BEZXY"), step)
                
                pd_box.clamp_(0,1)
                pd_box = einops.rearrange(pd_box, "B (E C) Z X Y -> B Z X Y E C", C = 4)
                
                pos = poscar.box2pos(pd_box[0], threshold = 0.5, nms = False, sort = False)
                poscar.pos2poscar(f"{self.work_dir}/{filenames[0]}-{step}.poscar", pos)
                
                match = self.analyse(pd_box, gt_box)
                
                for e in ["O","H"]:
                    for j,l in enumerate(match.split):
                        self.tb_writer.add_scalars(f"TRAIN/{e} {l}", {key: match[e,j,key] for key in ["AP", "AR"]}, step)
                    
            self.tb_writer.add_scalar(f"TRAIN/LR_rate", self.OPT.param_groups[0]['lr'], step)

            self.tb_writer.add_scalars(f"TRAIN", {key: value.last for key, value in T_dict.items()}, step)
            
            i += 1
            pbar.update(1)
                
        # -------------------------------------------
        pbar.update(1)
        pbar.close()
        return {**T_dict, "MET": self.analyse.summary()}
    
    def save(self, epoch, log_dic):
        met = 0
        if log_dic["Loss"].n > 0:
            met += log_dic["Loss"]()

        logger = self.logger

        if met < self.best_met:
            self.best_met = met

            log = []
            try:
                name = f"CP{epoch:02d}_LOSS{log_dic['Loss']:.4f}.pkl"
                self.denoise.save(path = f"{self.work_dir}/DIFFUSE_{name}")
                log.append(f"Saved a new net: {name}")
            except AttributeError:
                pass

            for i in log:
                logger.info(i)

        else:
            logger.info(f"No model was saved")