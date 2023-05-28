import torch
import numpy as np
from torch import nn
from datasets import poscar
from typing import Tuple, Dict
from collections.abc import Iterable
import math

class analyse_cls(nn.Module):
    """Analysing the result of the model
    example:
        >>> ana = analyse_cls(...)
        >>> ana(pd, gt)
        >>> out = ana.summary()
        >>> out
    output:
        >>> {"O": "0.0-3.0A" : {"T": 100, "P": 90, "TP": 90, "FP": 10, "FN": 10, "AP": 0.7, "AR": 0.8, "SUC": 0.1}, "H": {...}}
    """
    def __init__(self, elements: Tuple[str | None] = ("O", "H", None), real_size: Tuple[float] = (3, 25, 25), lattent_size: Tuple[float] = (8, 32, 32), ratio: float = 1.0, NMS: bool = True, sort: bool = True, threshold: float = 0.7, split: Tuple[float] = (0, 3.0), radius: Dict[str, float] = {"O": 0.740, "H": 0.528}):
        super(analyse_cls, self).__init__()
        
        self.elements = elements
        self.radius = radius
        self.NMS = NMS
        self.SORT = sort
        threshold = math.log(threshold / (1 - threshold))
        self.THRES = threshold
        self.SPLIT = split
        
        self.register_buffer("real_size", torch.tensor(real_size))
        self.register_buffer("lattent_size", torch.tensor(lattent_size))
        self.register_buffer("out_size", self.lattent_size * ratio)
        self.register_buffer("scale_up", self.out_size / self.real_size)

        # this is the faster matrix to calculate the distance between two points: DIS[|Z1 -Z2|, |X1 - X2|, |Y1 - Y2|] = r(p1,p2) 
        DIS = torch.ones_like(self.out_size).nonzero() # Z * X * Y * 3
        DIS = DIS / self.scale_up
        DIS = torch.sqrt((DIS ** 2).sum(-1)) # Z * X * Y
        self.register_buffer("DIS", DIS)
        
        # Counting Dict
        self.init_count()
        
    def init_count(self):
        elems = [e for e in self.elements if e is not None]
        self.elm = ElemLayerMet(elems = elems, split = self.SPLIT)
                
    def summary(self, reset: bool = True):
        """Return the summary dictionary
        :return: Dict[elements][layername][(T,P,TP,FP,FN,AP,AR,SUC)]
        """
        count = self.elm
        if reset:
            self.init_count()
        return count
    
    @torch.no_grad()
    def forward(self, batch_pd, batch_gt):
        # batch_gt_ind: List[Dict[str, torch.Tensor]]
        for b, (pd, gt) in enumerate(zip(batch_pd, batch_gt)):
            pd = poscar.box2pos(pd, real_size = self.real_size, order = [e for e in self.elements if e is not None], threshold=self.THRES, sort=self.SORT)
            gt = poscar.box2pos(gt, real_size = self.real_size, order = [e for e in self.elements if e is not None], threshold= 0.5, sort = False)

            if self.NMS:
                pd = self.nms(pd)
                pd = self.nms(pd)
            
            match = self.match(pd, gt) # Dict[str, tuple]: (T, P, TP, FP, FN, AP, AR, SUC)
            self.elm = self.elm + match
        return match
        
    def nms(self, pd_pos: Dict[str, torch.Tensor]):
        for e in pd_pos.keys():
            pos = pd_pos[e]
            if pos.nelement() == 0:
                continue
            cutoff = self.radius[e] * 1.4
            DIS = torch.cdist(pos, pos)
            DIS = DIS < cutoff
            DIS = (torch.triu(DIS, diagonal= 1)).float()
            restrain_tensor = DIS.sum(0)
            restrain_tensor -= ((restrain_tensor != 0).float() @ DIS)
            SELECT = restrain_tensor == 0
            pd_pos[e] = pos[SELECT]
            
        return pd_pos
        
    def match(self, pd: Dict[str,torch.Tensor], gt: Dict[str,torch.Tensor]) -> Dict[str, Dict[str, tuple]]:
        # :return: Dict[str, tuple]: "0.0-3.0A" : (T, P, TP, FP, FN, AP, AR, SUC)
        out = np.zeros((len(self.elements) - 1, len(self.SPLIT) - 1, 8))
        for i, e in enumerate(self.elements):
            if e is None:
                continue
            pd_pos = pd[e]
            gt_pos = gt[e]
        
            # if pd_pos.nelement() == 0:
            #     continue
            DIS = torch.cdist(pd_pos, gt_pos)
            SELECT_T = (DIS  < self.radius[e]).sum(0) != 0
            
            for j, (low, up) in enumerate(zip(self.SPLIT[:-1], self.SPLIT[1:])):
                LOW, UP = low * self.scale_up[0], up * self.scale_up[0]
                layer_P, layer_T = (pd_pos[:, 0] >= LOW) & (pd_pos[:, 0] < UP), (gt_pos[:, 0] >= LOW) & (gt_pos[:, 0] < UP)
                layer_TP = layer_T & SELECT_T
                P, T, TP = layer_P.sum().item(), layer_T.sum().item(), layer_TP.sum().item()
                FP, FN = P - TP, T - TP
                AR = 1 if T == 0 else TP / T
                AP = 0 if P == 0 else TP / P
                out[i,j] = [T, P, TP, FP, FN, AR, AP, (P == TP and T == TP)]
        return ElemLayerMet(out)

class ElemLayerMet():
    def __init__(self, init: np.ndarray = ...,
                 elems = ("O", "H"), 
                 split = (0, 3.0), 
                 met = ("T", "P", "TP", "FP", "FN", "AR", "AP", "SUC"),
                 format = ("sum", "sum", "sum", "sum", "sum", "mean", "mean", "mean")):
        self.elems = elems
        self.split = tuple(f"{i:3.1f}-{j:3.1f}A" for i,j in zip(split[:-1], split[1:]))
        self.met = met
        self.keys = (self.elems, self.split, self.met)
        if init is Ellipsis:
            self.sumMet = np.zeros((len(elems), len(split) - 1, len(met)), dtype= np.float128)
            self.num = 0
        else:
            assert init.shape == (len(elems), len(split) - 1, len(met))
            self.sumMet = init
            self.num = 1
        self.format = format
    
    def __getitem__(self, index: int | str | Tuple[int, str] = ...):
        out = self.get_met()
        IND = tuple()
        for i,key in enumerate(index):
            if isinstance(key, str):
                IND += (self.keys[i].index(key), )
            elif isinstance(key, tuple):
                IND += (tuple(self.keys[i].index(k) if isinstance(k, str) else k for k in key),)
            else:
                IND += (key,)
                
        return out[IND]
    
    def get(self, name: str | Tuple[str] = ..., reduce = "none"):
        if isinstance(name, str):
            ind = tuple()
            for type in (self.elems, self.split, self.met):
                if name in type:
                    ind += (type.index(name), )
                else:
                    ind += (slice(len(type)), )
                    
        elif isinstance(name, tuple):
            ind = tuple()
            for type in (self.elems, self.split, self.met):
                for n in name:
                    if n in type:
                        ind += (type.index(n), )
                        break
                else:
                    ind += (slice(len(type)), )
        out = self.__getitem__(ind)
        if reduce == "mean":
            out = out.mean()
        elif reduce == "sum":
            out = out.sum()
        return out
        
    def get_met(self):
        Sum = self.sumMet.copy()
        Num = [self.num if mode == "mean" else 1 for mode in self.format]
        return Sum / Num

    def add(self, met: np.ndarray):
        self.sumMet += met
        self.num += 1
    
    def __add__(self, other):
        if isinstance(other, np.ndarray):
            self.sumMet += other
            self.num += 1
        elif isinstance(self, ElemLayerMet):
            self.sumMet += other.sumMet
            self.num += other.num
        return self

class metStat():
    def __init__(self, value = None, mode:str = "mean", dtype = torch.float64, device = "cpu"):
        """To help you automatically find the mean or sum, which allows us to calculate something easily and consuming less space. 
        Args:
            mode (str): should be 'mean' or 'sum'
        """
        self.n = 0
        self._dtype = dtype
        self._mode = mode
        self._device = device
        self._value = torch.tensor(0, dtype= dtype, device= device)
        self._last = self._value.item()
        if value is not None:
            self.add(value)
    
    def add(self, other):
        if isinstance(other, Iterable) or isinstance(other ,metStat):
            self.extend(other)
        else:
            self.append(other)
        
    def append(self, x):
        if isinstance(x, torch.Tensor):
            x = x.item()
        self._last = x
        self.n += 1
        if self._mode == "mean":
            self._value = self._value * ((self.n - 1) / self.n) + x * (1 / self.n)
        elif self._mode == "sum":
            self._value = self._value + x
        elif self._mode == "max":
            self._value = max(self._value, x)
        elif self._mode == "min":
            self._value = min(self._value, x)
        self._value = self._value.type(self._dtype)
    
    def extend(self, xs):
        if isinstance(xs, metStat):
            value = xs.value.to(self.device, non_blocking=True)
            self._last = value.item()
            n = len(xs)
            if self._mode == "mean":
                self._value = self._value * (self.n / (n + self.n)) + value * (n / (n + self.n))
            elif self._mode == "sum":
                self._value = self._value + value
            elif self._mode == "max":
                self._value = max(self._value, value)
            elif self._mode == "min":
                self._value = min(self._value, value)
            self.n = self.n + n
            
        else:
            if isinstance(xs, torch.Tensor) and xs.dim() == 0:
                self.append(xs)
                
            elif isinstance(xs, Iterable):
                for value in xs:
                    self.append(value)
            else:
                raise TypeError(f"{type(xs)} is not an iterable")
          
    def __repr__(self):
        return f"{self._value.item()}"
    
    def __call__(self):
        return self._value.item()
    
    def __str__(self):
        return str(self._value.item())
    
    def __len__(self):
        return self.n
    
    def __format__(self, code):
        return self._value.item().__format__(code)
    
    @property
    def device(self):
        return self._device
    
    @property
    def value(self):
        return self._value
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def last(self):
        return self._last