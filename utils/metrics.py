import numpy as np
import numba

import torch
from torch import nn, Tensor

from . import poscar
from typing import Iterable

class Analyser(nn.Module):
    """
    Analyser module for the model prediction. This module is jit-compatible
    """
    def __init__(self, 
                 real_size: tuple[float] = (3.0, 25.0, 25.0), 
                 nms: bool = True, 
                 cutoff: tuple[float] = (1.0, 0.75),
                 sort: bool = True, 
                 split: list[float] = [0.0, 3.0]):
        super().__init__()
        self.register_buffer("real_size", torch.as_tensor(real_size))
        self._nms = nms
        self.cutoff = cutoff
        self._sort = sort
        self.split = split
        self.split[-1] += 1e-5
        self.S = len(split) - 1
    
    def forward(self, pred_clses: Tensor, pred_boxes: Tensor, targ_clses: Tensor, targ_boxes: Tensor) -> Tensor:
        """
        _summary_

        Args:
            pred_clses (Tensor): B C D H W
            pred_boxes (Tensor): B 3 D H W
            targ_clses (Tensor): B D H W 
            targ_boxes (Tensor): B D H W 3

        Returns: 
            Tensor: B C-1 S (TP, FP, FN)
        """
        with torch.no_grad():
            B, C, D, H, W = pred_clses.shape
            pred_clses = pred_clses.permute(0, 2, 3, 4, 1)
            pred_clses, pred_arges = pred_clses.max(dim = -1)
            pred_boxes = pred_boxes.permute(0, 2, 3, 4, 1).sigmoid() # B D H W 3
            confusion_matrix = torch.zeros((B, C - 1, self.S, 3), dtype = torch.long, device = pred_clses.device) # B C-1 (TP, FP, FN)
            for b in range(B):
                pred_cls = pred_clses[b]
                pred_arg, pred_box, cls_mask = poscar.boxncls2pos_torch(pred_arges[b], pred_boxes[b])
                targ_cls, targ_box, _ = poscar.boxncls2pos_torch(targ_clses[b], targ_boxes[b])
                for c in range(1, C):
                    mask = pred_arg == c
                    pred_ions = pred_box[mask] * self.real_size
                    if self._sort:
                        mid = pred_cls[cls_mask[0], cls_mask[1], cls_mask[2]][mask]
                        mid = mid.argsort(descending = True)
                        pred_ions = pred_ions[mid]
                    if self._nms:
                        pred_ions = self.nms(pred_ions, self.cutoff[c-1])
                    targ_ions = targ_box[targ_cls == c] * self.real_size
                    confusion_matrix[b, c-1] = self.match(pred_ions, targ_ions, cutoff= self.cutoff[c-1], split= self.split)
        return confusion_matrix

    @torch.jit.export
    def process_pred(self, pred_clses: Tensor, pred_boxes: Tensor, nms: bool | None = None, sort: bool | None = None) -> tuple[list[Tensor], list[list[int]]]:
        _nms = self._nms if nms is None else nms
        _sort = self._sort if sort is None else sort
        with torch.no_grad():
            B, C, D, H, W = pred_clses.shape
            pred_clses = pred_clses.permute(0, 2, 3, 4, 1)
            pred_clses, pred_arges = pred_clses.max(dim = -1)
            pred_boxes = pred_boxes.permute(0, 2, 3, 4, 1).sigmoid() # B D H W 3
            batch_pred_pos = []
            batch_ion_num = []
            for b in range(B):
                pred_pos = []
                ion_num = []
                pred_cls = pred_clses[b]
                pred_arg, pred_box, cls_mask = poscar.boxncls2pos_torch(pred_arges[b], pred_boxes[b])
                for c in range(1, C):
                    mask = pred_arg == c
                    pred_ions = pred_box[mask] * self.real_size
                    if _sort:
                        mid = pred_cls[cls_mask[0], cls_mask[1], cls_mask[2]][mask]
                        mid = mid.argsort(descending = True)
                        pred_ions = pred_ions[mid]
                    if _nms:
                        pred_ions = self.nms(pred_ions, self.cutoff[c-1])
                    pred_pos.append(pred_ions)
                    ion_num.append(pred_ions.shape[0])
                batch_pred_pos.append(torch.cat(pred_pos, dim = 0))
                batch_ion_num.append(ion_num)
        return batch_pred_pos, batch_ion_num
        
    
    @staticmethod
    def nms(pos: Tensor, cutoff: float) -> Tensor:
        """
        _summary_

        Args:
            pos (Tensor): N 3

        Returns:
            Tensor: N 3
        """
        DIS = torch.cdist(pos, pos)
        DIS = DIS < cutoff
        DIS = (torch.triu(DIS, diagonal= 1)).float()
        while True:
            N = pos.shape[0]
            restrain_tensor = DIS.sum(0)
            restrain_tensor -= ((restrain_tensor != 0).float() @ DIS)
            SELECT = restrain_tensor == 0
            DIS = DIS[SELECT][:, SELECT]
            pos = pos[SELECT]
            if N == pos.shape[0]:
                break
        return pos
        
    @staticmethod
    def match(pred: Tensor, targ: Tensor, cutoff: float, split: list[float]) -> Tensor:
        """
        _summary_

        Args:
            pred (Tensor): _description_
            targ (Tensor): _description_
            cutoff (float): _description_
            split (tuple[float]): _description_

        Returns:
            Tuple[list[Tensor], list[Tensor], list[Tensor]]: _description_
        """
        OUTS = torch.empty((len(split) - 1, 3), dtype = torch.long, device = pred.device)
        DIS = torch.cdist(pred, targ)
        SELECT_T = (DIS  < cutoff).sum(0) != 0
        for i in range(len(split) - 1):
            LOW, HIGH = split[i], split[i+1]
            layer_P, layer_T = (pred[:, 0] >= LOW) & (pred[:, 0] < HIGH), (targ[:, 0] >= LOW) & (targ[:, 0] < HIGH)
            layer_TP = layer_T & SELECT_T
            P, T, TP = layer_P.sum(), layer_T.sum(), layer_TP.sum()
            FP, FN = P - TP, T - TP
            OUTS[i,0], OUTS[i,1], OUTS[i,2] = TP, FP, FN
        return OUTS

class ConfusionMatrixCounter(object):
    def __init__(self, ):
        self._TP = []
        self._FP = []
        self._FN = []
        self._AR = []
        self._AP = []
        self._ACC = []
        self._SUC = []
    
    def __call__(self, confusion_matrix: np.ndarray) -> None:
        """
        _summary_

        Args:
            confusion_matrix (Tensor): B C-1 S (TP, FP, FN)
        """
        if isinstance(confusion_matrix, Tensor):
            confusion_matrix = confusion_matrix.detach().cpu().numpy()
        TP, FP, FN, AR, AP, ACC, SUC = self._count(confusion_matrix)
        self._TP.append(TP)
        self._FP.append(FP)
        self._FN.append(FN)
        self._AR.append(AR)
        self._AP.append(AP)
        self._ACC.append(ACC)
        self._SUC.append(SUC)

    def reset(self):
        self._TP = []
        self._FP = []
        self._FN = []
        self._AR = []
        self._AP = []
        self._ACC = []
        self._SUC = []
    
    def calc(self) -> tuple[np.ndarray]:
        TP = np.concatenate(self._TP, axis = 0).sum(axis = 0)
        FP = np.concatenate(self._FP, axis = 0).sum(axis = 0)
        FN = np.concatenate(self._FN, axis = 0).sum(axis = 0)
        AR = np.concatenate(self._AR, axis = 0).mean(axis = 0)
        AP = np.concatenate(self._AP, axis = 0).mean(axis = 0)
        ACC = np.concatenate(self._ACC, axis = 0).mean(axis = 0)
        SUC = np.concatenate(self._SUC, axis = 0).mean(axis = 0)
        return np.stack([TP, FP, FN, AR, AP, ACC, SUC], axis = -1)
    
    @staticmethod
    @numba.jit(nopython=True)
    def _count(cm: np.ndarray) -> tuple[np.ndarray]:
        return cm[..., 0], cm[..., 1], cm[..., 2], np.nan_to_num(cm[..., 0] / (cm[..., 0] + cm[..., 2])), np.nan_to_num(cm[..., 0] / (cm[..., 0] + cm[..., 1])), np.nan_to_num(cm[..., 0] / np.sum(cm, axis=-1)), ((cm[..., 1] == 0) & (cm[...,2] == 0)).astype(np.int32)
    
    @property
    def TP(self):
        return self._TP[-1]
    
    @property
    def FP(self):
        return self._FP[-1]
    
    @property
    def FN(self):
        return self._FN[-1]
    
    @property
    def AR(self):
        return self._AR[-1]
    
    @property
    def AP(self):
        return self._AP[-1]
    
    @property
    def ACC(self):
        return self._ACC[-1]
    
    @property
    def SUC(self):
        return self._SUC[-1]

class MolecularAnalyser(nn.Module):
    def __init__(self, real_size: tuple[float] = (25.0, 25.0, 4.0), nms: bool = True, sort: bool = True, threshold: float = 0.5, cutoff: float = 2.0):
        # real_size: x, y, z
        super().__init__()
        self.register_buffer("_real_size", torch.as_tensor(real_size))
        self._nms = nms
        self._sort = sort
        self._threshold = threshold
        self._cutoff = cutoff

    def forward(self, pred: Tensor, targ: Tensor):
        with torch.no_grad():
            pd_confs, pd_poses, pd_rotxs, pd_rotys = pred[..., 0].sigmoid(), pred[..., 1:4], pred[..., 4:7], pred[..., 7:]
            tg_confs, tg_poses, tg_rotxs, tg_rotys = targ[..., 0], targ[..., 1:4], targ[..., 4:7], targ[..., 7:]
            pd_rotzs = torch.cross(pd_rotxs, pd_rotys, dim = -1)
            tg_rotzs = torch.cross(tg_rotxs, tg_rotys, dim = -1)
            pd_Rs = torch.stack([pd_rotxs, pd_rotys, pd_rotzs], dim = -2)
            tg_Rs = torch.stack([tg_rotxs, tg_rotys, tg_rotzs], dim = -2)
            confusion_matrix = torch.empty((pred.shape[0], 1, 1, 3), dtype = torch.long, device = pred.device) # B C-1 (TP, FP, FN)
            MS = torch.zeros((pred.shape[0], 1, 1, 1), dtype = torch.float32, device = pred.device)
            for b in range(pred.shape[0]):
                pd_conf, pd_pos, pd_mask = self.tovec(pd_confs[b], pd_poses[b])
                _, tg_pos, tg_mask = self.tovec(tg_confs[b], tg_poses[b])
                pd_R = pd_Rs[b][pd_mask[0], pd_mask[1], pd_mask[2]]
                tg_R = tg_Rs[b][tg_mask[0], tg_mask[1], tg_mask[2]]
                if self._sort:
                    pd_conf_order = pd_conf.argsort(descending = True)
                    pd_pos = pd_pos[pd_conf_order]
                    pd_R = pd_R[pd_conf_order]
                if self._nms:
                    pd_nms_mask = self.argnms(pd_pos, self._cutoff)
                    pd_pos = pd_pos[pd_nms_mask]
                    pd_R = pd_R[pd_nms_mask]
                pd_pos = pd_pos * self._real_size
                tg_pos = tg_pos * self._real_size
                pd_match_ids, tg_match_ids = self.argmatch(pd_pos, tg_pos, self._cutoff)
                pd_R = pd_R[pd_match_ids]
                tg_R = tg_R[tg_match_ids]
                confusion_matrix[b,...,0] = len(pd_match_ids)
                confusion_matrix[b,...,1] = len(pd_pos) - len(pd_match_ids)
                confusion_matrix[b,...,2] = len(tg_pos) - len(tg_match_ids)
                if pd_R.shape[0] != 0:
                    MS[b,...,0] = torch.mean(torch.clip((pd_R * tg_R).sum(-1, keepdim=True) / (torch.norm(pd_R, dim = -1) * torch.norm(tg_R, dim = -1)), -1, 1).arccos() / torch.pi * 180.0)
            return confusion_matrix, MS
                
    def tovec(self, box_cls: torch.Tensor, box_off: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        _summary_

        Args:
            box_cls (torch.tensor): Z X Y
            box_off (torch.tensor): Z X Y 3

        Returns:
            tuple[torch.tensor]: (N, ), (N, 3), (3, Z * X * Y)
        """
        ZXY = box_cls.shape
        mask = torch.nonzero(box_cls > self._threshold)
        tupmask = mask.T
        box_cls = box_cls[tupmask[0], tupmask[1], tupmask[2]]
        box_off = (box_off[tupmask[0], tupmask[1], tupmask[2]] + mask) / torch.as_tensor(ZXY, dtype = box_off.dtype, device = box_off.device)
        
        return box_cls, box_off, tupmask
                    
    
    @staticmethod
    def argnms(pos: Tensor, cutoff: float) -> Tensor:
        """
        _summary_

        Args:
            pos (Tensor): N 3

        Returns:
            Tensor: N 3
        """
        DIS = torch.cdist(pos, pos)
        DIS = DIS < cutoff
        DIS = (torch.triu(DIS, diagonal= 1)).float()
        args = torch.ones(pos.shape[0], dtype = torch.bool, device = pos.device)
        while True:
            N = pos.shape[0]
            restrain_tensor = DIS.sum(0)
            restrain_tensor -= ((restrain_tensor != 0).float() @ DIS)
            SELECT = restrain_tensor == 0
            DIS = DIS[SELECT][:, SELECT]
            pos = pos[SELECT]
            args[args.nonzero(as_tuple=True)] = SELECT
            if N == pos.shape[0]:
                break
        return args

    @staticmethod
    def argmatch(pred: Tensor, targ: Tensor, cutoff: float) -> tuple[Tensor]:
        # This function is only true when one prediction does not match two targets and one target can match more than two predictions
        # return pred_ind, targ_ind
        dis = torch.cdist(targ, pred)
        dis = (dis < cutoff).nonzero()
        dis = dis[:, (1, 0)]
        unique, idx, counts = torch.unique(dis[...,1], sorted=True, return_inverse=True, return_counts=True)
        ind_sorted = torch.argsort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        if cum_sum.shape[0] != 0:
            cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
        first_indicies = ind_sorted[cum_sum]
        dis = dis[first_indicies]
        
        return dis[...,0], dis[...,1]
    
        

class metStat():
    def __init__(self, value = None, reduction:str = "mean"):
        """To help you automatically find the mean or sum, which allows us to calculate something easily and consuming less space. 
        Args:
            reduction (str): should be 'mean', 'sum', 'max', 'min', 'none'
        """
        self._value = np.array([])
        if reduction == "mean":
            self._reduction = np.mean
        elif reduction == "sum":
            self._reduction = np.sum
        elif reduction == "max":
            self._reduction = np.max
        elif reduction == "min":
            self._reduction = np.min
        else:
            raise ValueError(f"reduction should be 'mean' or 'sum', but got {reduction}")

        if value is not None:
            self.add(value)
    
    def add(self, other):
        if isinstance(other, Iterable) or isinstance(other ,metStat):
            self.extend(other)
        else:
            self.append(other)
        
    def append(self, x: np.ndarray | torch.Tensor | float | int):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray) and x.size != 1:
            raise ValueError(f"Only support scalar input, but got {x}")
        self._value = np.append(self._value, x)
    
    def extend(self, xs: Iterable):
        if isinstance(xs, (metStat, list, tuple)):
            self._value = np.append(self._value, xs._value)
        elif isinstance(xs, np.ndarray):
            xs = xs.view(-1)
            self._value = np.append(self._value, xs)
        elif isinstance(xs, torch.Tensor):
            xs = xs.detach().cpu().view(-1).numpy()
            self._value = np.append(self._value, xs)
        elif isinstance(xs, Iterable):
            for x in xs:
                self._value = np.append(self._value, x)
        else:
            raise TypeError(f"{type(xs)} is not an iterable")
    
    def reset(self):
        self._value = np.array([])
    
    def calc(self) -> float:
        return self._reduction(self._value)
    
    def __call__(self, other):
        self.add(other)
    
    def __repr__(self):
        return f"{self._reduction(self._value)}"
        
    def __str__(self):
        return str(self._reduction(self._value))
    
    def __len__(self):
        return len(self._value)
    
    def __format__(self, code):
        return self._reduction(self._value).__format__(code)
    
    @property
    def value(self):
        return self._value
    
