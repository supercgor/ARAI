import time
import os
import tqdm
from collections import OrderedDict
import numpy as np

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import build_basic_model
from model.utils import model_load

from datasets import AFMDataset
from utils import *
from utils.metrics import metStat, analyse_cls
from utils.criterion import modelLoss

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
parser.add_argument("-d", "--dataset", help="Specify the dataset to use", default="../data/bulkexp")
parser.add_argument("-f", "--filelist", help="Specify the filelist to use", default="test.filelist")
parser.add_argument("-l", "--label", help="Specify whether to use label", default=False)
parser.add_argument("-n", "--npy", help="Specify whether to save npy", default=True)
parser.add_argument("-c", "--checkpoint", help="Specify the checkpoint to use", default="model/pretrain/unet_v0/unet_CP17_LOSS0.0331.pkl")

args = parser.parse_args()

class Trainer():
    def __init__(self):
        cfg = get_config()
        self.cfg = cfg
        self.label = str2bool(args.label)
        self.npy = str2bool(args.npy)

        self.data_path = args.dataset
        self.model_name = args.checkpoint.split('/')[-2]
        os.makedirs(f"{self.data_path}/result", exist_ok=True)
        os.makedirs(f"{self.data_path}/npy", exist_ok=True)
        os.makedirs(f"{self.data_path}/result/{self.model_name}", exist_ok=True)
        os.makedirs(f"{self.data_path}/npy/{self.model_name}", exist_ok=True)

        self.work_dir = "/".join(args.checkpoint.split("/")[:-1])
        self.logger = Logger(path=f"{self.data_path}/result/{self.model_name}",elem=cfg.data.elem_name,split=cfg.setting.split)

        self.tb_writer = SummaryWriter(log_dir=f"{self.work_dir}/runs/test")

        self.net = build_basic_model(cfg).cuda().eval()

        model_load(self.net, args.checkpoint, True)

        self.analyse = analyse_cls(real_size = cfg.data.real_size,
                                   lattent_size=cfg.model.out_size,
                                   split=cfg.setting.split,
                                   threshold=cfg.model.threshold).cuda()
        self.LOSS = modelLoss(pos_w=cfg.criterion.pos_weight).cuda()

        pred_data = AFMDataset(self.data_path,
                               self.cfg.data.elem_name,
                               file_list = args.filelist,
                               transform = None,
                               img_use = None,
                               random_layer=False,
                               model_inp = self.cfg.model.inp_size,
                               model_out = self.cfg.model.out_size,
                               label = self.label)

        self.pred_loader = DataLoader(pred_data,
                                      batch_size=1,
                                      num_workers=self.cfg.setting.num_workers,
                                      pin_memory=self.cfg.setting.pin_memory,
                                      shuffle=False)

    @torch.no_grad()
    def fit(self):
        self.logger.info(f"Start Prediction")
        self.logger.info(f"dataset: {args.dataset}, filelist: {args.filelist}, checkpoint: {args.checkpoint}")
        start_time = time.time()

        iter_times = len(self.pred_loader)
        it_loader = iter(self.pred_loader)

        pbar = tqdm.tqdm(
            total=iter_times - 1, desc=f"{args.checkpoint} - Test", position=0, leave=True, unit='it')

        loss = metStat()
        i = 0
        loss_count = {}
        file_count = {}
        
        while i < iter_times:
            try:
                if self.label:
                    imgs, gt_box, filenames = next(it_loader)
                else:
                    imgs, filenames = next(it_loader)
            except StopIteration:
                break
            imgs = imgs.cuda()

            pd_box = self.net(imgs)

            if self.label:
                gt_box = gt_box.cuda()
                B  = gt_box.shape[0]
                losses = [self.LOSS(pd_box[(b,),...], gt_box[(b,),...]) for b in range(B)]
                matches = [self.analyse(pd_box[(b,),...], gt_box[(b,),...]) for b in range(B)]
                loss.add(self.LOSS(pd_box, gt_box))

            for i, (filename, x) in enumerate(zip(filenames, pd_box)):
                points_dict = poscar.box2pos(x,
                               real_size=self.cfg.data.real_size,
                               threshold=self.cfg.model.threshold)
                poscar.pos2poscar(f"{self.data_path}/result/{self.model_name}/{filename}.poscar", points_dict)
                if self.npy:
                    np.save(f"{self.data_path}/npy/{self.model_name}/{filename}.npy", x.cpu().numpy())
                    
                if self.label:
                    loss_count[filename] = losses[i].item()
                    file_count[filename] = matches[i].get_met

            pbar.set_postfix(file=filenames[0])
            pbar.update(1)
            i += 1

        pbar.update(1)
        pbar.close()

        if self.label:
            self.logger.epoch_info(
                0, {"loss": loss, "MET": self.analyse.summary()})
            np.savez(f"{self.data_path}/result/{self.model_name}/{args.filelist}_loss.npz", **loss_count)
            np.savez(f"{self.data_path}/result/{self.model_name}/{args.filelist}_MET.npz", **file_count)

        self.logger.info(f"Spend time: {time.time() - start_time:.2f}s")
        self.logger.info(f'End prediction')
        
if __name__ == "__main__":
    Trainer().fit()
