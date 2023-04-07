import torch
import torch.nn as nn
from einops import repeat, rearrange
from utils.analyze_data import *
from torch.autograd import grad

class focalLoss(nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2, sigmoid = False, reduction = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid() if sigmoid else nn.Identity()
        self.bce = nn.BCELoss(reduction = "none")
        if reduction == "none":
            self.reduction = nn.Identity()
        elif reduction == "sum":
            self.reduction = lambda x: x.sum()
        elif reduction == "mean":
            self.reduction = lambda x: x.mean()
        else:
            raise ValueError(f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'")
    
    def forward(self, p, t):
        pt = p * t + (1 - p) * (1 - t)
        loss = self.bce(p, t)
        loss = loss * ((1 - pt) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * t + (1 - self.alpha) * (1 - t)
            loss = alpha_t * loss
            
        return self.reduction(loss)

class wassersteinLoss(nn.Module):
    """Using WGan-gp idea, this is calculating the gan loss of the images: L = P(T) - P(G)"""
    def __init__(self, alpha = 1,real_label=1.0, fake_label=0.0):
        super(wassersteinLoss, self).__init__()
        
        self.alpha = alpha
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))

    def forward(self, pred, real):
        return self.alpha * ( pred - real )

class grad_penalty(nn.Module):
    def __init__(self, alpha = 10, mix = False, net = None):
        super(grad_penalty, self).__init__()
        self.mix = mix
        self.net = net
        self.alpha = alpha
        
    def forward(self, inputs, other = ...):
        B, C, Z, X, Y = inputs.shape
        gp_w = torch.ones((B,), device = inputs.device)
        if self.mix:
            alpha = torch.rand(self.batch_size, device= inputs.device)
            inputs = inputs.detach().requires_grad_(True)
            alpha = repeat(torch.rand((B,), device = inputs.device), "B -> B C Z X Y", C = C, Z = Z, X = X, Y = Y)
            inputs = inputs * alpha + (1 - alpha) * other
            output = self.net(inputs)
        else:
            output = other
        
        gradients = grad(inputs=inputs, outputs=output, grad_outputs = gp_w, create_graph=True, retain_graph = ~self.mix)[0]
            
        gradients = gradients.view(B, -1)
        gradients = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return ((gradients - 1) ** 2).mean() * self.alpha
    
class basicLoss(nn.Module):
    def __init__(self,
                 loss_c = focalLoss(), loss_cw = 0.5,
                 loss_xy = nn.MSELoss(), loss_xyw = 0.25,
                 loss_z = nn.MSELoss(), loss_zw = 0.25,
                 threshold = 0.7,
                 ):
        super().__init__()
        self.threshold = threshold
        self.loss_c = loss_c
        self.loss_cw = loss_cw
        self.loss_xy = loss_xy
        self.loss_xyw = loss_xyw
        self.loss_z = loss_z
        self.loss_zw = loss_zw
    
    def forward(self, y, target):
        B, C, Z, X, Y = y.shape
        target = target[:B, :C, :Z, :X, :Y]
        lossc = self.loss_c(y[:,::4,...], target[:,::4,...]) * self.loss_cw
        y = rearrange(y, "B (C E) Z X Y -> B E Z X Y C", C = 4)
        target = rearrange(target, "B (C E) Z X Y -> B E Z X Y C", C = 4)
        mask = (y[...,0] > self.threshold)
        if mask.sum() == 0:
            lossxy = 0
            lossz = 0
        else:
            lossxy = self.loss_xy(y[...,1:3][mask], target[...,1:3][mask]) * self.loss_xyw
            lossz = self.loss_z(y[...,3][mask], target[...,3][mask]) * self.loss_zw
        return lossc + lossxy + lossz

class localLoss(nn.Module):
    def __init__(self, w = 0.1):
        super().__init__()
        #TODO
        

class Criterion(nn.Module):
    def __init__(self, cfg, local_epoch = 9999):
        super(Criterion, self).__init__()
        self.cfg = cfg
        self.ele_name = cfg.data.elem_name
        self.local_epoch = local_epoch
        # Used to Caculate the absulute position of offset, this tensor fulfilled that t[x,y,z] = [x,y,z], pit refers to position index tensor
        reduction = cfg.criterion.reduction
        # 用於解決正負樣本不平衡 e.g. 正:負 100:400, 那麼可以將pos_weight設定成 torch.tensor(4)
        pos_weight = torch.tensor(cfg.criterion.pos_weight, dtype=torch.float32)
        
        self.weight_confidence = cfg.criterion.weight_confidence
        self.weight_offset_xy = cfg.criterion.weight_offset_xy
        self.weight_offset_z = cfg.criterion.weight_offset_z

        self.register_buffer('decay_paras', torch.tensor(cfg.criterion.decay))
        self.loss_confidence = nn.BCEWithLogitsLoss(reduction=reduction,
                                                             pos_weight=pos_weight)
        self.loss_offset_xy = nn.MSELoss(reduction=reduction)
        self.loss_offset_z = nn.MSELoss(reduction=reduction)


    def forward(self, predictions, targets):
        assert predictions.shape == targets.shape, f"prediction shape {predictions.shape} doesn't match {targets.shape}"
        device = self.decay_paras.device
        prediction = predictions.view((-1, len(self.ele_name), 4))

        target = targets.view((-1, len(self.ele_name), 4))
        
        loss = torch.tensor([0.0], requires_grad=True, device=device)

        mask = (target[..., 3] > 0.5)
        loss_confidence = self.loss_confidence(
            prediction[..., 3], target[..., 3]) * self.weight_confidence
        loss_offset_xy = self.loss_offset_xy(
            prediction[..., :2][mask], target[..., :2][mask]) * self.weight_offset_xy
        loss_offset_z = self.loss_offset_z(
            prediction[..., 2][mask], target[..., 2][mask]) * self.weight_offset_z

        loss = loss + loss_confidence + loss_offset_xy + loss_offset_z

        self.loss = loss.item()
        
        return loss

    def loss_local(self,epoch, info):
        # -----------------------------------------------
        device = self.decay_paras.device
        ele_name = self.ele_name
        if epoch < self.local_epoch:
            return torch.tensor([0.0], device= device)
        # -----------------------------------------------
    
        loss_local = torch.tensor([0.0], requires_grad=True, device = device)
        
        P_list, T_list, TP_index_list, P_pos_list, T_pos_list = info["P"], info["T"], info["TP_index"], info["P_pos"], info["T_pos"]
        for P,T,TP_index,P_pos,T_pos in zip(P_list, T_list, TP_index_list, P_pos_list, T_pos_list):
            one_sample_loss = torch.tensor([0.0], requires_grad=True, device=device)
            # -----------------------------------------------
            # Including O-O, H-H, O-H, H-O
            for i, ele in enumerate(ele_name):
                if TP_index[i][0].nelement() == 0:
                    continue
                for j,o_ele in enumerate(ele_name):
                    if o_ele == "O":
                        continue
                    TP_T_pos = T_pos[i][TP_index[i][0]]
                    TP_P_pos = P_pos[i][TP_index[i][1]]
                    
                    d_T = torch.cdist(TP_T_pos,T_pos[j]).int()
                    d_P = torch.cdist(TP_P_pos,P_pos[j]).int()

                    if i == j:
                        q1 = torch.arange(0,d_T.shape[0])
                        q2 = TP_index[j][0]
                        d_T[q1,q2] = 10
                        q1 = torch.arange(0,d_P.shape[0])
                        q2 = TP_index[j][1]
                        d_P[q1,q2] = 10
            
                    for r in range(4):
                        # To count for the near atoms
                        mask = (d_P == r).float()
                        np1 = mask.sum(1)
                        np2 = (d_T == r).float().sum(1) 
                        NP = (np1 - np2).abs().unsqueeze(0)
                        one_sample_loss = one_sample_loss + self.decay_paras[r] * NP.mm(mask).mv(P[j][...,3].sigmoid()) * 2
                    # Each prediction should be divided by the TP number and times FP/T
            loss_local = loss_local + one_sample_loss/ max(1,TP_index[i][0].nelement())
    
        loss_local = loss_local / len(T_list)
        
        if loss_local[0] == 0:
            return torch.tensor([0.0], requires_grad=True, device = device)
        else:
            return loss_local / loss_local.item() * 0.1 * self.loss
        
if __name__ == '__main__':
    pass
