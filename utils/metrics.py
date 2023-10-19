import numpy as np
import numba
from functools import partial
import time

import torch
from torch import nn, Tensor
from torch import multiprocessing as mp

from . import poscar
from . import functional
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

class parallelAnalyser(nn.Module):
    def __init__(self, real_size: tuple[float] = (25.0, 25.0, 4.0), nms: bool = True, sort: bool = True, threshold: float = 0.5, cutoff: float = 2.0, split: list[float] = [0.0, 4.0, 8.0]):
        super().__init__()
        self.register_buffer("_real_size", torch.as_tensor(real_size, dtype=torch.float))
        self._nms = nms
        self._sort = sort
        self._threshold = threshold
        self._cutoff = cutoff
        self._split = split
        self._split[-1] += 1e-5
    
    @torch.no_grad()
    def forward(self, preds: Tensor, targs: Tensor):
        # B D H W C -> B H W D C
        # preds = preds.permute(0, 2, 3, 1, 4)
        # targs = targs.permute(0, 2, 3, 1, 4)
        
        f = partial(self.eval_one, threshold = self._threshold, cutoff = self._cutoff, real_size = self._real_size, sort = self._sort, nms = self._nms, split = self._split)
        results = [f(pred, targ) for pred, targ in zip(preds.detach(), targs.detach())]
        return torch.stack(results, dim = 0)
             
    @staticmethod
    def eval_one(pred: Tensor, targ: Tensor, threshold: float, cutoff: float, real_size: Tensor, sort: bool, nms: bool, split: list[float]) -> tuple[Tensor]:
        _, pd_pos, pd_R = functional.box2orgvec(pred, functional.inverse_sigmoid(threshold), cutoff, real_size, sort, nms)
        _, tg_pos, tg_R = functional.box2orgvec(targ, 0.5, cutoff, real_size, False, False)
        
        cm = torch.zeros((1, len(split) - 1, 4), dtype = torch.float, device = pred.device) # C S (TP, FP, FN)
            
        pd_match_ids, tg_match_ids = functional.argmatch(pd_pos, tg_pos, cutoff/2)
        
        pd_R = pd_R[pd_match_ids] # N 3 3
        tg_R = tg_R[tg_match_ids] # N 3 3
        
        match_tg_pos = tg_pos[tg_match_ids] # N 3
        
        ang = torch.einsum("bij,bij->bi", pd_R, tg_R) / pd_R.norm(dim=-1) / tg_R.norm(dim=-1)
        ang.clamp_(-1, 1).acos_().div_(torch.pi/ 180.0).nan_to_num_(90.0)
        
        for i, (low, high) in enumerate(zip(split[:-1], split[1:])):
            match_tg_mask = (match_tg_pos[:, 2] >= low) & (match_tg_pos[:, 2] < high) # TP
            pd_mask = (pd_pos[:, 2] >= low) & (pd_pos[:, 2] < high) # P
            tg_mask = (tg_pos[:, 2] >= low) & (tg_pos[:, 2] < high) # T
            cm[0, i, 0] = match_tg_mask.sum() #TP
            cm[0, i, 1] = pd_mask.sum() - match_tg_mask.sum() # FP
            cm[0, i, 2] = tg_mask.sum() - match_tg_mask.sum() # FN
            cm[0, i, 3] = ang.mean().nan_to_num(90.0)         # ANG
            
        return cm
                
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
    
class ConfusionMatrixCounter(object):
    def __init__(self, ):
        self.reset()
    
    def __call__(self, confusion_matrix: np.ndarray) -> None:
        """
        _summary_

        Args:
            confusion_matrix (Tensor): B C-1 S (TP, FP, FN)
        """
        if isinstance(confusion_matrix, Tensor):
            if confusion_matrix.device != torch.device("cpu"):
                confusion_matrix = confusion_matrix.detach().cpu().numpy()
            else:
                confusion_matrix = confusion_matrix.numpy()
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

class ConfusionCounter(object):
    def __init__(self,):
        self._cmc = []
    
    def add(self, cm: torch.Tensor) -> None:
        if cm.device != torch.device("cpu"):
            cm = cm.detach().cpu().numpy()
        else:
            cm = cm.numpy()
        if cm.ndim == 3:
            self._cmc.append(cm)
        elif cm.ndim == 4:
            self._cmc.extend(cm)
        
    def calc(self):
        self._cmc = np.stack(self._cmc, axis = 0)
        return self
    
    def reset(self):
        self._cmc = []
    
    @property
    def TP(self):
        return self._cmc[..., 0].sum(axis = 0)
    
    @property
    def FP(self):
        return self._cmc[..., 1].sum(axis = 0)
    
    @property
    def FN(self):
        return self._cmc[..., 2].sum(axis = 0)
    
    @property
    def AR(self):
        return np.nan_to_num(self._cmc[..., 0] / (self._cmc[..., 0] + self._cmc[..., 2])).mean(axis = 0)
    
    @property
    def AP(self):
        return np.nan_to_num(self._cmc[..., 0] / (self._cmc[..., 0] + self._cmc[..., 1])).mean(axis = 0)
    
    @property
    def ACC(self):
        return np.nan_to_num(self._cmc[..., 0] / np.sum(self._cmc, axis=-1)).mean(axis = 0)
    
    @property
    def SUC(self):
        return ((self._cmc[..., 1] == 0) & (self._cmc[...,2] == 0)).astype(np.float_).mean(axis = 0)
    
class ConfusionRotate(ConfusionCounter):
    def __init__(self):
        super().__init__()
        
    @property
    def ROT(self):
        return self._cmc[..., 3].mean(axis = 0)