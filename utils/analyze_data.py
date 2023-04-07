import os
import torch
import math
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance
from collections import OrderedDict
from einops import rearrange
from .tools import metStat

class FIDQ3D(nn.Module):
    """Calculate the FID for quasi 3D systems, Input shape is like (B, X, Y, Z, 8)

    Args:
        nn (_type_): _description_
    """
    def __init__(self, feature = 64):
        super(FIDQ3D, self).__init__()
        self.FID = FrechetInceptionDistance(feature=feature, normalize= True)
        
    def forward(self, predictions, targets):
        if predictions.shape != targets.shape:
            B, C, Z, X, Y = predictions.shape
            targets = targets[:B, :C, :Z, :X, :Y]
        # B * 8 * Z * X * Y
        predictions = predictions.detach().requires_grad_(False)
        predictions = rearrange(predictions, "B (C E) Z X Y -> B E (C X) (Z Y)", C = 4)
        targets = rearrange(targets, "B (C E) Z X Y -> B E (C X) (Z Y)", C = 4)
        if C // 4 == 1:
            predictions = predictions[:,(0,0,0),...]
            targets = targets[:,(0,0,0),...]
        elif C // 4 == 2:
            predictions = predictions[:, (0,0,1),...]
            targets = targets[:,(0,0,1),...]

        self.FID.update(predictions, real = False)
        self.FID.update(targets, real = True)
        return self.FID.compute()

class analyse(nn.Module):
    def __init__(self, 
                 out_size = (4, 32, 32),                # output size of the model box ( Z * X * Y )
                 real_size = (3, 25, 25),               # real size of the box
                 scale = 1.4,                           # allow the radius become larger
                 threshold = 0.5,                       # the cutoff threshold
                 sort: bool = True,                     # sort the points according to the confidence
                 nms: bool = True,                      # use nms or not
                 split = (0, 3.0),                      # the boundary of the layer
                 d: OrderedDict = ...                   # radius of the atoms
                 ):
        super(analyse, self).__init__()
        if d is Ellipsis:
            d = OrderedDict(O = 0.740, H = 0.528)
        for key in d:
            d[key] = d[key] * scale
            
        self.D = d
        self.S = split
        self._nms = nms
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
        self._met = {e : {(low,up):{f"{key}": metStat(mode = m) 
                     for key,m in [("ACC", "mean"), ("SUC", "mean"), ("TP", "sum"), ("FP", "sum"), ("FN", "sum")] 
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
                P = self.select(pred, threshold= self.threshold, sort = self._toSort)[:,:3]
                T = self.select(targ, threshold= self.threshold, sort = self._toSort)[:,:3]
                
                if self._nms:
                    P, _ = self.nms(P, self.D[e])
                    T, _ = self.nms(T, self.D[e])
                
                TP, FP, FN = self.match(P, T, self.D[e])
                
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
                    acc = len(tp)/(len(tp)+len(fp)+len(fn))
                    suc = acc == 1
                    for idt_name, idt in zip(["TP", "FP", "FN", "ACC", "SUC"],[len(tp), len(fp), len(fn), suc, acc]):
                        self._met[e][layer][idt_name].add(idt)
        self.init()
        return self._met
    
    @staticmethod
    def organize(x, ind = ... , scale: torch.Tensor = torch.tensor([1.0, 1.0, 1.0, 1.0])):
        """ the input should be 'B C Z X Y' and output is 'B (Z X Y) C'"""
        out = rearrange(x, "B (C E) Z X Y -> B E C (Z X Y)", C = 4) + ind
        out = torch.einsum("B E C R, C -> B E R C" , out, scale)
        return out
    
    @staticmethod
    def select(x, threshold = 0.5, sort = True):
        out = x[(x[..., 0] > threshold), :]
        if sort:
            out = out[out[..., 0].argsort(descending=True),:]
        return out
    
    @staticmethod
    def nms(points, distance):
        """do the nms for a table of points, and e means the elem type

        Args:
            points (tensor): the shape is like N * 3
            r (float): a float

        Returns:
            tuple: Selected, Not selected
        """
        SELECT = torch.full((points.shape[0],), True)
        DIS_MAT = torch.cdist(points, points) < distance
        DIS_MAT = torch.triu(DIS_MAT, diagonal= 1).T
        try:
            for b,a in DIS_MAT.nonzero():
                if SELECT[a]:
                    SELECT[b] = False
        except ValueError:
            pass
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
        SELECT_P = torch.full((pred.shape[0],), False)
        SELECT_T = torch.full((targ.shape[0],), False)
        DIS_MAT = torch.cdist(pred, targ) < distance
        try:
            for a,b in DIS_MAT.nonzero():
                if not SELECT_P[a] and not SELECT_T[b]:
                    SELECT_P[a], SELECT_T[b] = True, True
        except ValueError:
            pass
        return pred[SELECT_P], pred[SELECT_P.logical_not()], targ[SELECT_T.logical_not()]
        
    @classmethod
    def split(self, points, split):
        """_summary_

        Args:
            points (tensor): Z, X, Y
            split (None | tuple, optional): _description_. Defaults to None.
        """
            
        SELECT = torch.logical_and(points[...,0] > split[0], points[...,0] < split[1])
        
        if len(split) > 2:
            return points[SELECT], *self.split(points[SELECT.logical_not()], split = split[1:])
        
        else:
            return (points[SELECT],)