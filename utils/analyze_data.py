import os
import torch
import math
from torch import nn
from torchvision.utils import make_grid
from torchmetrics.image.fid import FrechetInceptionDistance
from collections import OrderedDict
from .tools import metStat

class FIDQ3D(nn.Module):
    """Calculate the FID for quasi 3D systems, Input shape is like (B, X, Y, Z, 8)

    Args:
        nn (_type_): _description_
    """
    def __init__(self, feature = 64):
        super(FIDQ3D, self).__init__()
        self.register_buffer("FID", FrechetInceptionDistance(feature=feature))
        
    def forward(self, predictions, targets):
        # B * X * Y * Z * 8
        batch, X, Y, Z, LST = predictions.shape
        x = predictions[..., range(3,LST, 4)] > 0
        x = torch.permute(x, (0, 4, 3, 1,  2))  # X, Y, Z, C -> C, Z, H, W
        x = x.reshape((batch * 2 * Z, 1, X, Y))
        x = (x * 255).to(dtype = torch.uint8)
        self.FID.update(x, real=False)
        
        batch, X, Y, Z, LST = targets.shape
        x = targets[..., range(3, LST, 4)] > 0
        x = torch.permute(x, (0, 4, 3, 1, 2))  # X, Y, Z, C -> C, Z, H, W
        x = x.reshape((batch * 2 * Z, 1, X, Y))
        x = (x * 255).to(dtype = torch.uint8)
        self.FID.update(x, real=True)
        
        self.FID.compute()

class Analyzer2(nn.Module):
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
        super(Analyzer2, self).__init__()
        if d is Ellipsis:
            d = OrderedDict(O = 0.740, H = 0.528)
        for key in d:
            d[key] = d[key] * scale
            
        self.D = d
        self.S = split
        self.N = nms
        self.sort = sort

        expand = tuple(j/i for i,j in zip(out_size, real_size))
        self.register_buffer("expand", torch.tensor(expand))
        IND_MAT = torch.ones(out_size)
        IND_MAT = torch.nonzero(IND_MAT).view(IND_MAT.shape + (3,))
        self.register_buffer("IND", IND_MAT)
        
        self.threshold = threshold
        
        self.init()
        
    def init(self):
        self.P = OrderedDict((key, []) for key in self.D.keys())
        self.T = OrderedDict((key, []) for key in self.D.keys())
        self.TP = OrderedDict((key, []) for key in self.D.keys())
        self.FP = OrderedDict((key, []) for key in self.D.keys())
        self.FN = OrderedDict((key, []) for key in self.D.keys())

    def forward(self, predictions, targets):
        assert predictions.shape == targets.shape, f"prediction shape {predictions.shape} doesn't match {targets.shape}"
        
        try:
            batch, X, Y, Z, LST = predictions.shape
        except ValueError:
            batch, X, Y, Z, LST = 1, *predictions.shape
            
        preds = torch.reshape(predictions, (batch, X, Y,Z, LST//4, 4))  # to (b, X, Y, Z, 2, 4)
        preds = torch.permute(preds, (0, 4, 3, 1, 2, 5))                # to (b, 2, Z, X, Y, 4)
        preds = preds[...,(2,0,1,3)]                                    # to (dz, dx, dy, c)
        preds[...,:3] = (preds[...,:3] + self.IND) * self.expand        # make offset to actual position
        preds = preds.reshape(batch, LST//4,-1, 4)                      # to (b, 2, Z * X * Y, 4)
        
        targs = torch.reshape(targets, (batch, X, Y,Z, LST//4, 4))
        targs = torch.permute(targs, (0, 4, 3, 1, 2, 5))
        targs = targs[...,(2,0,1,3)]                                    # to (dz, dx, dy, c)
        targs[...,:3] = (targs[...,:3] + self.IND) * self.expand
        targs = targs.reshape(batch, LST//4,-1, 4)
        self.init()
        for b , (Pred, Targ) in enumerate(zip(preds, targs)):
            for e, pred, targ in zip(self.D.keys() ,Pred, Targ):
                inds = pred[...,3] > self.threshold
                P = pred[inds]
                if self.sort:
                    inds = torch.argsort(P[...,3], descending= True)
                    P = P[inds]
                P = P[...,:3]
                
                inds = targ[...,3] > self.threshold
                T = targ[inds]
                if self.sort:
                    inds = torch.argsort(T[...,3], descending= True)
                    T = T[inds]
                T = T[...,:3]
                
                if self.N:
                    P, _ = self.nms(P, self.D[e])
                    T, _ = self.nms(T, self.D[e])
                
                TP, FP, FN = self.match(P, T, self.D[e])
                self.P[e].append(P)
                self.T[e].append(T)
                self.TP[e].append(TP)
                self.FP[e].append(FP)
                self.FN[e].append(FN)
        
                
    @classmethod
    def match(self, pred, targ, distance):
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
    def nms(self, points, distance):
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
            return points[SELECT]
                
class Analyzer(nn.Module):
    def __init__(self, cfg):
        super(Analyzer, self).__init__()
        # mark the config
        self.cfg = cfg

        # some initialization
        self.elem = cfg.data.elem_name
        self.elem_num = len(self.elem)
        self.ele_diameter = [0.740 * 1.4, 0.528 * 1.4]

        self.out_size = cfg.model.out_size[::-1]
        self.output_shape = self.out_size + (self.elem_num, 4)
        self.split = cfg.setting.split

        self.register_buffer('real_size', torch.tensor(cfg.data.real_size[::-1]))

        self.register_buffer(
            'lattice_expand', self.real_size/torch.tensor(self.out_size))

        self.do_nms = cfg.model.nms

        self.P_threshold = cfg.model.threshold
        self.T_threshold = 0.5

        # Construct the pit-tensor
        # Used to Caculate the absulute position of offset, this tensor fulfilled that t[x,y,z] = [x,y,z], pit refers to position index tensor
        self.register_buffer('pit', torch.ones(
            self.out_size).nonzero().view(self.out_size + (3,)))
        self.register_buffer('empty_tensor', torch.tensor([]))

    def forward(self, predictions, targets):
        assert predictions.shape == targets.shape, f"prediction shape {predictions.shape} doesn't match {targets.shape}"
        device = self.pit.device
        """_summary_

        Args:
            prediction (_type_): _description_
            target (_type_): _description_

            return (tuple): TP,FP,FN
        """
        # Store the previous information, the list contains

        total_P_nms = []
        total_T = []
        total_TP_index_nms = []
        total_P_pos_nms = []
        total_T_pos = []
        total_P_pos = []
        total_TP_index = []
        total_P = []

        # ------------------------------------
        # pre-process
        # ------------------------------------
        # Reshape to batch,X,Y,Z,4,ele
        predictions = predictions.view((-1,) + self.output_shape)
        targets = targets.view((-1,) + self.output_shape)
        batch_size = predictions.size(0)
        # Change to ele,X,Y,Z,4
        predictions = predictions.permute(0, 4, 1, 2, 3, 5)
        targets = targets.permute(0, 4, 1, 2, 3, 5)

        # ------------------------------------
        pit = self.pit
        lattice_expand = self.lattice_expand
        # ------------------------------------

        for batch in range(batch_size):

            T, P, P_nms, T_pos, P_pos, P_pos_nms, TP_index, TP_index_nms = [],[],[],[],[],[],[],[]

            for ele, diameter in enumerate(self.ele_diameter):
                prediction = predictions[batch, ele]
                target = targets[batch, ele]

                mask_t = target[..., 3] > self.T_threshold
                mask_p = prediction[..., 3] > self.P_threshold

                T_position = (target[..., :3] + pit)[mask_t] * lattice_expand
                P_position = (prediction[..., :3] + pit)[mask_p] * lattice_expand
                
                prediction_nms, P_nmspos = self.nms(prediction, diameter)

                # Matching the nearest
                TP_dist_nms = torch.cdist(T_position, P_nmspos)
                TP_T_index_nms = (TP_dist_nms < diameter).sum(1).nonzero()

                if TP_T_index_nms.nelement() != 0:
                    TP_T_index_nms = TP_T_index_nms.squeeze(1)
                    TP_P_index_nms = TP_dist_nms[TP_T_index_nms].min(
                        1).indices
                else:
                    TP_P_index_nms = self.empty_tensor

                TP_dist = torch.cdist(T_position, P_position)
                TP_T_index = (TP_dist < diameter).sum(1).nonzero()

                if TP_T_index.nelement() != 0:
                    TP_T_index = TP_T_index.squeeze(1)
                    TP_P_index = TP_dist[TP_T_index].min(1).indices
                else:
                    TP_P_index = self.empty_tensor

                P.append(prediction[mask_p])
                P_nms.append(prediction_nms)
                T.append(target[mask_t])
                TP_index.append([TP_T_index, TP_P_index])
                TP_index_nms.append([TP_T_index_nms, TP_P_index_nms])
                P_pos.append(P_position)
                P_pos_nms.append(P_nmspos)
                T_pos.append(T_position)

            total_P.append(P)
            total_P_nms.append(P_nms)
            total_T.append(T)
            total_TP_index.append(TP_index)
            total_TP_index_nms.append(TP_index_nms)
            total_P_pos.append(P_pos)
            total_P_pos_nms.append(P_pos_nms)
            total_T_pos.append(T_pos)

        return {"P_nms": total_P_nms,
                "P": total_P,
                "T": total_T,
                "TP_index": total_TP_index,
                "TP_index_nms": total_TP_index_nms,
                "P_pos": total_P_pos,
                "P_pos_nms": total_P_pos_nms,
                "T_pos": total_T_pos, }

    def nms(self, prediction, diameter) -> torch.Tensor:
        
        # ------------------------------------
        pit = self.pit
        lattice_expand = self.lattice_expand
        # ------------------------------------
        
        mask_p = prediction[..., 3] > self.P_threshold
        P_position = (prediction[..., :3] + pit)[mask_p] * lattice_expand

        index = torch.argsort(prediction[...,3][mask_p])

        sorted_prediction = P_position[index]

        dist_matrix = torch.cdist(sorted_prediction, sorted_prediction)

        dist_matrix = torch.triu(dist_matrix < diameter, diagonal=1).float()

        restrain_tensor = dist_matrix.sum(0)
        restrain_one = (restrain_tensor != 0).unsqueeze(0).float()
        correct = restrain_one.mm(dist_matrix)
        restrain_tensor = restrain_tensor - correct
        selection = restrain_tensor[0] == 0

        return prediction[mask_p][index[selection]], sorted_prediction[selection] # prediction, position
        # which show that whether the points should be restrained.
        # improve great performance ~ 100 times need 0.058s

    def count(self, info):
        device = self.pit.device
        TP_index, P_pos, T_pos = info["TP_index_nms"], info["P_pos_nms"], info["T_pos"]
        batch_size = len(TP_index)
        dic = {}
        for batch in range(batch_size):
            for i, ele in enumerate(self.elem):

                TP_num = TP_index[batch][i][0].size(0)

                T_z, P_z = T_pos[batch][i][..., 2], P_pos[batch][i][..., 2]

                if TP_index[batch][i][0].nelement() != 0:
                    TP_T_z = T_z[TP_index[batch][i][0]]
                else:
                    TP_T_z = self.empty_tensor
                if TP_index[batch][i][1].nelement() != 0:
                    TP_P_z = P_z[TP_index[batch][i][1]]
                else:
                    TP_P_z = self.empty_tensor

                split_past = 0
                for split in self.split[1:]:
                    TP_num = torch.logical_and(
                        TP_P_z >= split_past, TP_P_z < split).sum().float()
                    FP_num = torch.logical_and(
                        P_z >= split_past, P_z < split).sum().float() - TP_num
                    TP_num = torch.logical_and(
                        TP_T_z >= split_past, TP_T_z < split).sum().float()
                    FN_num = torch.logical_and(
                        T_z >= split_past, T_z < split).sum().float() - TP_num
                    if (TP_num + FP_num + FN_num) == 0:
                        acc = torch.ones(1, device=device).squeeze()
                        suc = acc
                    else:
                        acc = TP_num / (TP_num + FP_num + FN_num)
                        suc = (FP_num == 0 and FN_num == 0).float()
                    key = f"{ele}-{split_past:3.1f}-{split:3.1f}"

                    if f"{key}-TP" in dic:
                        dic[f"{key}-TP"].add(TP_num)
                        dic[f"{key}-FP"].add(FP_num)
                        dic[f"{key}-FN"].add(FN_num)
                        dic[f"{key}-ACC"].add(acc)
                        dic[f"{key}-SUC"].add(suc)
                    else:
                        dic[f"{key}-TP"] = metStat(TP_num, mode= "sum", dtype=torch.long)
                        dic[f"{key}-FP"] = metStat(FP_num, mode= "sum", dtype=torch.long)
                        dic[f"{key}-FN"] = metStat(FN_num, mode= "sum", dtype=torch.long)
                        dic[f"{key}-ACC"] = metStat(acc)
                        dic[f"{key}-SUC"] = metStat(suc)

                    split_past = split
        return dic