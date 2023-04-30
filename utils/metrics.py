import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from datasets.poscar import poscar
from typing import Tuple, Dict
from collections.abc import Iterable
import math
from einops import rearrange, repeat

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

    def select(x, threshold = 0.7, sort = True):
        mask = (x[..., 0] > threshold)
        out = x[mask]
        if sort:
            out = out[out[..., 0].argsort(descending=True)]
        return out
    
    def select(self, pd):
        """Select all the prediction points
        :param pd: the prediction of the model, shape: (Z, X, Y, C)
        :return: pd_conf: the confidence of the prediction, shape: (N, )
        :return: pd_pos: the position index of the prediction, shape: (N, 3)
        """
        pd_clss = torch.argmax(pd, dim = -1)
        out = {}
        for i, e in enumerate(self.elements):
            if e is None:
                continue
            mask = pd_clss == i
            pd_conf = pd[mask]
            pd_pos = mask.nonzero()
            if self.SORT:
                order = torch.argsort(pd_conf, descending = True)
                pd_conf = pd_conf[order]
                pd_pos = pd_pos[order]
            out[e] = (pd_conf, pd_pos)
        return out
        
    def nms(self, pd_pos: Dict[str, torch.Tensor]):
        for e in pd_pos.keys():
            pos = pd_pos[e]
            if pos.nelement() == 0:
                continue
            cutoff = self.radius[e] * 2
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
        return f"{self._value.item():.4f}, mode: {self._mode}, len: {self.n}"
    
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


class analyse(nn.Module):
    def __init__(self, 
                 out_size = (4, 32, 32),                # output size of the model box ( Z * X * Y )
                 real_size = (3, 25, 25),               # real size of the box
                 scale = 1.4,                           # allow the radius become larger
                 threshold = 0.7,                       # the cutoff threshold
                 sort: bool = True,                     # sort the points according to the confidence
                 nms: bool = True,                      # use nms or not
                 split = (0, 3.0),                      # the boundary of the layer
                 d: OrderedDict = ...                   # radius of the atoms
                 ):
        super(analyse, self).__init__()
        if d is Ellipsis:
            d = OrderedDict(O = 0.740, H = 0.528)
        
        self.D = d
        self.S = split
        self._nms = nms
        self._scale = scale
        self._toSort = sort

        expand = (1, ) + tuple(j/i for i,j in zip(out_size, real_size))
        self.register_buffer("scale_factor", torch.tensor(expand))
        IND = torch.ones(1, *out_size).nonzero().T
        self.register_buffer("IND", IND)
        
        self.threshold = threshold
        
        # "O" : (0.0, 1.0) : TP, FP, FN,... 2d dict
        self.init_met()
        
        self.init()
        
    def init(self):
        self._P = OrderedDict((key, []) for key in self.D.keys())
        self._T = OrderedDict((key, []) for key in self.D.keys())
        self._TP = OrderedDict((key, []) for key in self.D.keys())
        self._FP = OrderedDict((key, []) for key in self.D.keys())
        self._FN = OrderedDict((key, []) for key in self.D.keys())

    def init_met(self):
        self._met = {e : {(low,up):{f"{key}": metStat(mode = m, dtype= dtp) 
                     for key,m, dtp in [("ACC", "mean", torch.float32), ("SUC", "mean", torch.float32), ("TP", "sum", torch.int64), ("FP", "sum", torch.int64), ("FN", "sum", torch.int64)] 
                     } for low,up in zip(self.S[:-1], self.S[1:])} for e in self.D.keys()}
    
    def forward(self, predictions, targets):
        if predictions.dim() == 4:
            predictions = predictions[None,...]
            targets = targets[None,...]
        B, C, Z, X, Y = predictions.shape
        targets = targets[:B,:C,:Z,:X,:Y]
        
        preds = self.organize(predictions, ind = self.IND, scale = self.scale_factor)
        targs = self.organize(targets, ind = self.IND, scale = self.scale_factor)
        
        for B , (Pred, Targ) in enumerate(zip(preds, targs)):
            for e, pred, targ in zip(self.D.keys() ,Pred, Targ):
                P = self.select(pred, threshold= self.threshold, sort = self._toSort)[:,1:]
                T = self.select(targ, threshold= 0.5, sort = False)[:,1:]
                if self._nms:
                    P, _ = self.nms(P, self.D[e] * self._scale)
                
                TP, FP, FN = self.match(P, T, 0.5)
                self.record(e, P, T, TP, FP, FN)
    
    def record(self, elem, P, T, TP, FP, FN):
        self._P[elem].append(P)
        self._T[elem].append(T)
        self._TP[elem].append(TP)
        self._FP[elem].append(FP)
        self._FN[elem].append(FN)
    
    def compute(self):
        period = tuple(zip(self.S[: -1], self.S[1: ]))
        for e in self.D.keys():
            for TP, FP, FN in zip(self._TP[e], self._FP[e], self._FN[e]):
                TP, FP, FN = map(lambda x: self.split(x, self.S), [TP, FP, FN])
                for layer, tp, fp, fn in zip(period, TP, FP, FN):
                    acc = tp.shape[0]/(tp.shape[0] + fp.shape[0] + fn.shape[0])
                    suc = acc == 1
                    for idt_name, idt in zip(["TP", "FP", "FN", "ACC", "SUC"],[tp.shape[0], fp.shape[0], fn.shape[0], acc, suc]):
                        self._met[e][layer][idt_name].add(idt)
        self.init()
        return self._met
        
    @staticmethod
    def organize(x, ind = ... , scale: torch.Tensor = torch.tensor([1.0, 1.0, 1.0, 1.0])):
        """ the input should be 'B C Z X Y' and output is 'B (Z X Y) C'"""
        out = rearrange(x, "B (E C) Z X Y -> B E C (Z X Y)", C = 4) + ind
        out = torch.einsum("B E C R, C -> B E R C" , out, scale)
        return out
    
    @staticmethod
    def select(x, threshold = 0.7, sort = True):
        mask = (x[..., 0] > threshold)
        out = x[mask]
        if sort:
            out = out[out[..., 0].argsort(descending=True)]
        return out
    
    @staticmethod
    def nms(points, distance):
        """do the nms for a table of points, and e means the elem type

        Args:
            points (tensor): the shape is like N * 3
            r (float): a float
            
            This algorithm is based on tensor operation
            1. create a distance matrix
            2. check if distance < threshold
            3. get the upper triangle of the matrix (remove the diagonal because it is always 0)
            4. sum for dimension 0 (sum for each column) means whether the points need to be restrained
            5. the second step is to cancel restraint points that contribute to the restraint of other points
            6. End 
            Example:
            >>>        0 1 1
            >>> DIS =  0 0 1
            >>>        0 0 0
            than
            >>> restrain_tensor = [0, 1, 2]
            but the point 1 is restrained by point 0, so we need to cancel the restraint of point 1, thus
            >>> restrain_one = [0, 1, 1]
            >>> restrain_one @ DIS_MAT = [0, 0, 1]
            >>> final = [0, 1, 1]

        Returns:
            tuple: Selected, Not selected
        """
        if points.nelement() == 0:
            return points, points
        
        DIS_MAT = torch.cdist(points, points) < distance
        DIS_MAT = (torch.triu(DIS_MAT, diagonal= 1)).float()
        restrain_tensor = DIS_MAT.sum(0)
        restrain_tensor -= ((restrain_tensor != 0).float() @ DIS_MAT)
        SELECT = restrain_tensor == 0
        
        return points[SELECT], points[SELECT.logical_not()]
    
    @staticmethod
    def match(pred, targ, distance):
        """match two group of points if there distance are lower that a given threshold

        Args:
            pred (tensor): N * 3
            targ (tensor): M * 3
            distance (_type_): float

        Returns:
            (matched from pred, non-matched from pred, non-matched from targ) : R * 3, (N - R) * 3, (M - R) * 3
        """
        if pred.nelement() == 0:
            return pred, pred, targ
        DIS_MAT = torch.cdist(pred, targ)
        SELECT_P = (DIS_MAT  < distance).sum(1) != 0
        SELECT_T = torch.full((targ.shape[0], ), False)
        mask = DIS_MAT[SELECT_P]
        if mask.nelement() != 0:
            mask = mask.argmin(1)
            SELECT_T[mask] = True
        return pred[SELECT_P], pred[SELECT_P.logical_not()], targ[SELECT_T.logical_not()]
        
    @classmethod
    def split(self, points, split):
        """_summary_

        Args:
            points (tensor): Z, X, Y
            split (None | tuple, optional): _description_. Defaults to None.
        """
        if points.nelement() == 0:
            return tuple(points for _ in range(len(split) - 1))
        
        SELECT = torch.logical_and(points[...,0] > split[0], points[...,0] < split[1])
        
        if len(split) > 2:
            return points[SELECT], *self.split(points[SELECT.logical_not()], split = split[1:])
        
        else:
            return (points[SELECT],)