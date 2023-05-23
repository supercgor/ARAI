import torch
import torch.nn as nn
from einops import repeat, rearrange
from torch.autograd import grad
from torch.nn import functional as F

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

class wassersteinLoss(nn.Module):
    """Using WGan-gp idea, this is calculating the gan loss of the images: L = P(T) - P(G)"""
    def __init__(self, alpha = 1,real_label=1.0, fake_label=0.0):
        super(wassersteinLoss, self).__init__()
        
        self.alpha = alpha
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))

    def forward(self, pred, real):
        return self.alpha * (pred - real).mean()

def grad_penalty(net, real_fea, fake_fea):
    B, C, Z, X, Y = real_fea.shape
    alpha = torch.rand(B, device= real_fea.device)
    mixed = torch.einsum("B, B C Z X Y -> B C Z X Y", alpha, real_fea) + torch.einsum("B, B C Z X Y -> B C Z X Y", 1 - alpha, fake_fea)
    mixed.requires_grad_(True)
    pred = net(mixed) # B,
    gradients = grad(inputs=mixed, outputs = pred, create_graph=True, retain_graph = True, grad_outputs=torch.ones_like(pred))[0]
    gradients = gradients.view(B, -1)
    gradients = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return ((gradients - 1) ** 2).mean()

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
    def __init__(self, threshold: float = 0.5, pos_w = (25., 15.),w_conf: float = 1.0, w_xy: float = 1.0, w_z: float = 1.0, w_local: float = 0.1):
        super(modelLoss, self).__init__()
        self.threshold = threshold
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
        
        mask = targets[..., 0] > self.threshold
        LOSS = self.w_conf * self.Lc(predictions[..., 0], targets[..., 0])
        LOSS = LOSS + self.w_z * self.Lz(predictions[..., 1][mask], targets[..., 1][mask])
        LOSS = LOSS + self.w_xy * self.Lxy(predictions[..., 2:][mask], targets[..., 2:][mask])

        if self.use_local:
            LOSS = LOSS + self.w_local * self.Llocal(predictions, targets)
        return LOSS

    # def loss_local(self,epoch, info):
    #     # -----------------------------------------------
    #     device = self.decay_paras.device
    #     ele_name = self.ele_name
    #     if epoch < self.local_epoch:
    #         return torch.tensor([0.0], device= device)
    #     # -----------------------------------------------
    
    #     loss_local = torch.tensor([0.0], requires_grad=True, device = device)
        
    #     P_list, T_list, TP_index_list, P_pos_list, T_pos_list = info["P"], info["T"], info["TP_index"], info["P_pos"], info["T_pos"]
    #     for P,T,TP_index,P_pos,T_pos in zip(P_list, T_list, TP_index_list, P_pos_list, T_pos_list):
    #         one_sample_loss = torch.tensor([0.0], requires_grad=True, device=device)
    #         # -----------------------------------------------
    #         # Including O-O, H-H, O-H, H-O
    #         for i, ele in enumerate(ele_name):
    #             if TP_index[i][0].nelement() == 0:
    #                 continue
    #             for j,o_ele in enumerate(ele_name):
    #                 if o_ele == "O":
    #                     continue
    #                 TP_T_pos = T_pos[i][TP_index[i][0]]
    #                 TP_P_pos = P_pos[i][TP_index[i][1]]
                    
    #                 d_T = torch.cdist(TP_T_pos,T_pos[j]).int()
    #                 d_P = torch.cdist(TP_P_pos,P_pos[j]).int()

    #                 if i == j:
    #                     q1 = torch.arange(0,d_T.shape[0])
    #                     q2 = TP_index[j][0]
    #                     d_T[q1,q2] = 10
    #                     q1 = torch.arange(0,d_P.shape[0])
    #                     q2 = TP_index[j][1]
    #                     d_P[q1,q2] = 10
            
    #                 for r in range(4):
    #                     # To count for the near atoms
    #                     mask = (d_P == r).float()
    #                     np1 = mask.sum(1)
    #                     np2 = (d_T == r).float().sum(1) 
    #                     NP = (np1 - np2).abs().unsqueeze(0)
    #                     one_sample_loss = one_sample_loss + self.decay_paras[r] * NP.mm(mask).mv(P[j][...,3].sigmoid()) * 2
    #                 # Each prediction should be divided by the TP number and times FP/T
    #         loss_local = loss_local + one_sample_loss/ max(1,TP_index[i][0].nelement())
    
    #     loss_local = loss_local / len(T_list)
        
    #     if loss_local[0] == 0:
    #         return torch.tensor([0.0], requires_grad=True, device = device)
    #     else:
    #         return loss_local / loss_local.item() * 0.1 * self.loss
