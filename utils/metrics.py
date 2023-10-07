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
        self.sort = sort
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
                    if self.sort:
                        mid = pred_cls[cls_mask[0], cls_mask[1], cls_mask[2]][mask]
                        mid = mid.argsort(descending = True)
                        pred_ions = pred_ions[mid]
                    if self._nms:
                        pred_ions = self.nms(pred_ions, self.cutoff[c-1])
                    targ_ions = targ_box[targ_cls == c] * self.real_size
                    confusion_matrix[b, c-1] = self.match(pred_ions, targ_ions, cutoff= self.cutoff[c-1], split= self.split)
        return confusion_matrix

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
        B, C, S, _ = confusion_matrix.shape
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
        return cm[..., 0], cm[..., 1], cm[..., 2], cm[..., 0] / (cm[..., 0] + cm[..., 2]), cm[..., 0] / (cm[..., 0] + cm[..., 1]), cm[..., 0] / np.sum(cm, axis=-1), ((cm[..., 1] == 0) & (cm[...,2] == 0)).astype(np.int32)
        

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
    
