import torch
import torch.nn as nn
from einops import repeat, rearrange
from torch.autograd import grad
from torch.nn import functional as F
from typing import Tuple

class BinaryFocalLoss(nn.Module):
    r"""Focal loss for binary-classification
        input shape should be ( * , n) & target shape should be ( * , n)
        :math:`L = - α ( w+ * y * (1 - x) ** γ * log (x) + w- * (1 - y) * x ** γ * log (1 - x))`
    """
    def __init__(self, alpha = 0.25, gamma = 2, activation: None | str = None, reduction = "mean", pos_weight: torch.Tensor = ..., eps = 1e-8):
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        # activation function
        if activation is None:
            self.act = nn.Identity()
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "softmax":
            self.act = nn.Softmax(dim = -1)
        else:
            raise "Invalid activation function"
        
        # position weight for each class
        if not isinstance(pos_weight, torch.Tensor):
            if pos_weight is ...:
                pos_weight = torch.ones(2)
            else:
                pos_weight = torch.tensor(pos_weight)
        self.register_buffer("pos_weight", 2 * pos_weight / pos_weight.sum())
        
        # reduction method
        if reduction == "none":
            self.reduction = nn.Identity()
        elif reduction == "sum":
            self.reduction = lambda x: x.sum()
        elif reduction == "mean":
            self.reduction = lambda x: x.mean()
        else:
            raise ValueError(f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'")
    
    def forward(self, x, y):
        x = self.act(x)
        ce_loss = F.binary_cross_entropy(x, y, reduction="none")
        pt = x * y + (1 - x) * (1 - y)
        loss = ce_loss * ((1 - pt) ** self.gamma)
        if self.alpha > 0:
            alpha_t = self.alpha * y + (1 - self.alpha) * (1 - y)
            loss = alpha_t * loss
        loss = self.reduction(loss)
        return loss

class MultiCLSFocalLoss(nn.Module):
    r"""Focal loss for multi-classification
    
        :math:`L = - α * (1 - x) ** γ * log (x)`
    """
    def __init__(self, alpha = 1.0, gamma = 2, activation: None | str = "softmax", reduction = "mean", cls_num = 3, pos_weight: torch.Tensor = ...):
        
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        # activation function
        if activation is None:
            self.act = nn.Identity()
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "softmax":
            self.act = nn.Softmax(dim = -1)
        else:
            raise "Invalid activation function"
        
        # position weight for each class
        if not isinstance(pos_weight, torch.Tensor):
            if pos_weight is ...:
                pos_weight = torch.ones(cls_num)
            else:
                pos_weight = torch.tensor(pos_weight)
        self.register_buffer("pos_weight", pos_weight / pos_weight.sum())
        
        # reduction method
        if reduction == "none":
            self.reduction = nn.Identity()
        elif reduction == "sum":
            self.reduction = lambda x: x.sum()
        elif reduction == "mean":
            self.reduction = lambda x: x.mean()
        else:
            raise ValueError(f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'")
    
    def forward(self, pd: torch.Tensor, gt: torch.Tensor, channels_first = False):
        """
        :param pd: prediction, shape: (B, Z, X, Y, C)
        :param gt: ground truth, shape: (B, Z, X, Y, 1)
        """
        if channels_first:
            pd = pd.permute(0, 2, 3, 4, 1)
            
        pd = self.act(pd)
        pd = torch.gather(pd, -1, gt)
        loss = - self.pos_weight[gt] * self.alpha * (1 - pd) ** self.gamma * torch.log(pd)
        loss = self.reduction(loss)
        return loss

# ============================================
# Local Loss implementation
# paper: https://pubs.aip.org/aip/jcp/article/134/7/074106/954787/Atom-centered-symmetry-functions-for-constructing

class ACSFLoss(nn.Module):
    def __init__(self, use_func = ("g1", "g2", "g3"), elem_w = (0.2, 0.8), pos_w = (0.2, 0.8), threshold = 0.7):
        super().__init__()
        self.THRES = threshold
        #TODO
    
    def forward(self, pd_box, gt_box):
        # pd_box: (B, Z, X, Y, E, 4)
        # gt_box: (B, Z, X, Y, E, 4)
        # torch.bicount -> torch.split -> torch.bicount -> torch.split
        for b, (pd, gt) in enumerate(zip(pd_box, gt_box)):
            pd_mask, gt_mask = pd[...,0] > self.THRES, gt[...,0] > 0.5
            
            pd = pd[pd[...,0] > self.THRES] # (N, 4)
            gt = gt[gt[...,0] > 0.5] # (N, 4)
            

class modelLoss(nn.Module):
    def __init__(self, 
                 pos_w = (5.0, 5.0),
                 w_conf: float = 1.0, 
                 w_xy: float = 0.5, 
                 w_z: float = 0.5, 
                 w_local: float = 0.1):
        super(modelLoss, self).__init__()
        self.w_conf = w_conf
        self.w_xy = w_xy
        self.w_z = w_z
        self.w_local = w_local
        self.use_local = False
        
        self.Lc = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))
        self.Lxy = nn.MSELoss()
        self.Lz = nn.MSELoss()

    def forward(self, predictions, targets):
        B, Z, X, Y, E, C = predictions.shape
        predictions = predictions.view(-1, E, C)
        targets = targets.view(-1, E, C)
        
        mask = targets[..., 0] > 0.5
        LOSS = self.w_conf * self.Lc(predictions[..., 0], targets[..., 0])
        LOSS = LOSS + self.w_z * self.Lz(predictions[..., 1][mask], targets[..., 1][mask])
        LOSS = LOSS + self.w_xy * self.Lxy(predictions[..., 2:][mask], targets[..., 2:][mask])

        if self.use_local:
            LOSS = LOSS + self.w_local * self.Llocal(predictions, targets)
            
        return LOSS
    
class modelLoss_bnc(nn.Module):
    def __init__(self, w_xy = 0.5, w_z = 0.5, w_c = 1.0, w_cls = (7.0, 5.0, 1.0), order = ("O", "H")):
        super(modelLoss_bnc, self).__init__()
        self.w_xy = w_xy
        self.w_z = w_z
        self.w_c = w_c
        self.order = order
        
        self.Lc = nn.CrossEntropyLoss(weight=torch.tensor(w_cls))
        self.Lxy = nn.MSELoss()
        self.Lz = nn.MSELoss()
    
    def forward(self, pd: Tuple[torch.Tensor], tg: Tuple[torch.Tensor]):
        pd_pos, pd_cls = pd 
        tg_pos, tg_cls = tg
        lw = self.w_c * self.Lc(pd_cls, tg_cls)
        # B Z X Y
        mask = tg_cls != len(self.order)
        pd_pos, tg_pos = pd_pos.permute(0, 2, 3, 4, 1), tg_pos.permute(0, 2, 3, 4, 1)
        lz= self.w_z * self.Lz(pd_pos[...,0][mask], tg_pos[...,0][mask])
        lxy = self.w_xy * self.Lxy(pd_pos[...,1:][mask], tg_pos[...,1:][mask])
        print(lz, lxy, lw)
        return lw + lxy + lz