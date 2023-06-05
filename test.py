import time
import os
import tqdm
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import UNetModel, Regression
from model.utils import basicParallel
from itertools import chain

from datasets import AFMDataset
from utils import *
from utils.metrics import metStat, analyse_cls
from utils.criterion import modelLoss
from utils.schedular import Scheduler

if os.environ.get("USER") == "supercgor":
    from config.config import get_config
else:
    from config.wm import get_config
    
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser = argparse.ArgumentParser(description='Testing the model')
parser.add_argument("-d", "--dataset", help="Specify the dataset to use", default="../data/bulkice")
parser.add_argument("-f", "--filelist", help="Specify the filelist to use", default="test.filelist")
parser.add_argument("-l", "--label", help="Specify whether to use label", default=False)
parser.add_argument("-n", "--npy", help="Specify whether to save npy", default=True)

args = parser.parse_args()

class Trainer():
    def __init__(self):
        cfg = get_config()
        self.cfg = cfg
        self.label = str2bool(args.label)
        self.npy = str2bool(args.npy)
        assert cfg.setting.device != [], "No device is specified!"

        self.data_path = args.dataset
        os.makedirs(f"{self.data_path}/result", exist_ok=True)
        os.makedirs(f"{self.data_path}/npy", exist_ok=True)
        os.makedirs(f"{self.data_path}/result/{cfg.model.checkpoint}", exist_ok=True)
        os.makedirs(f"{self.data_path}/npy/{cfg.model.checkpoint}", exist_ok=True)

        self.work_dir = f"{cfg.path.check_root}/{cfg.model.checkpoint}"
        self.logger = Logger(path=f"{self.data_path}/result/{cfg.model.checkpoint}",elem=cfg.data.elem_name,split=cfg.setting.split)

        self.tb_writer = SummaryWriter(log_dir=f"{self.work_dir}/runs/test")

        self.net = UNetModel(image_size=(16, 128, 128),
                             in_channels=1,
                             model_channels=32,
                             out_channels=32,
                             num_res_blocks=2,
                             attention_resolutions=(8,),
                             dropout=0.0,
                             channel_mult=(1, 2, 4, 4),
                             dims=3,
                             num_heads=4,
                             time_embed=None,
                             use_checkpoint=False).cuda()

        self.reg = Regression(in_channels=32, out_channels=8).cuda()

        log = []

        load = {"net": f"{cfg.path.check_root}/{cfg.model.checkpoint}/{cfg.model.fea}",
                "reg": f"{cfg.path.check_root}/{cfg.model.checkpoint}/{cfg.model.reg}"}

        for key, name in load.items():
            if len(cfg.setting.device) >= 2:
                setattr(self, key,
                        basicParallel(getattr(self, key), device_ids=cfg.setting.device))
            if name == "":
                continue
            try:
                getattr(self, key).load(name, pretrained=False)
                log.append(f"Loade model {key} from {name}")
            except (FileNotFoundError, IsADirectoryError):
                log.append(f"Model {key} loading warning, {name}")
        for l in log:
            self.logger.info(l)

        self.model = nn.Sequential(self.net, self.reg)
        self.model = self.model.eval()

        self.analyse = analyse_cls(threshold=cfg.model.threshold).cuda()
        self.LOSS = modelLoss(pos_w=cfg.criterion.pos_weight).cuda()

        pred_data = AFMDataset(self.data_path,
                               self.cfg.data.elem_name,
                               file_list=args.filelist,
                               img_use=self.cfg.data.img_use,
                               model_inp=self.cfg.model.inp_size,
                               model_out=self.cfg.model.out_size,
                               label=self.label)

        self.pred_loader = DataLoader(pred_data,
                                      batch_size=1,
                                      num_workers=self.cfg.setting.num_workers,
                                      pin_memory=self.cfg.setting.pin_memory,
                                      shuffle=False)

    @torch.no_grad()
    def fit(self):
        self.logger.info(f"Start Prediction")
        start_time = time.time()

        iter_times = len(self.pred_loader)
        it_loader = iter(self.pred_loader)

        pbar = tqdm.tqdm(
            total=iter_times - 1, desc=f"{self.cfg.model.checkpoint} - Test", position=0, leave=True, unit='it')

        loss = metStat()
        i = 0
        while i < iter_times:
            if self.label:
                imgs, gt_box, filenames = next(it_loader)
            else:
                imgs, filenames = next(it_loader)

            imgs = imgs.cuda()

            pd_box = self.model(imgs)

            if self.label:
                gt_box = gt_box.cuda()
                loss.add(self.LOSS(pd_box, gt_box))
                match = self.analyse(pd_box, gt_box)

            for filename, x in zip(filenames, pd_box):
                points_dict = poscar.box2pos(x,
                               real_size=self.cfg.data.real_size,
                               threshold=self.cfg.model.threshold)
                poscar.pos2poscar(f"{self.data_path}/result/{self.cfg.model.checkpoint}/{filename}.poscar", points_dict)
                if self.npy:
                    os.makedirs(
                        f"{self.data_path}/npy/{self.cfg.model.checkpoint}", exist_ok=True)
                    torch.save(x.cpu().numpy(
                    ), f"{self.data_path}/npy/{self.cfg.model.checkpoint}/{filename}.npy")

            pbar.set_postfix(file=filenames[0])
            pbar.update(1)
            i += 1

        pbar.update(1)
        pbar.close()

        if self.label:
            self.logger.epoch_info(
                0, {"loss": loss, "MET": self.analyse.summary()})

        self.logger.info(f"Spend time: {time.time() - start_time:.2f}s")
        self.logger.info(f'End prediction')
        
if __name__ == "__main__":
    Trainer().fit()
