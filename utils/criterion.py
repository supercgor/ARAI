import torch
import torch.nn as nn
from utils.analyze_data import *


class Criterion(nn.Module):
    def __init__(self, cfg, local_epoch = 9999):
        super(Criterion, self).__init__()
        self.cfg = cfg
        self.ele_name = cfg.DATA.ELE_NAME
        self.local_epoch = local_epoch
        # Used to Caculate the absulute position of offset, this tensor fulfilled that t[x,y,z] = [x,y,z], pit refers to position index tensor
        reduction = cfg.TRAIN.CRITERION.REDUCTION
        pos_weight = torch.tensor(cfg.TRAIN.CRITERION.POS_WEIGHT, dtype=torch.float32)
        
        self.weight_confidence = cfg.TRAIN.CRITERION.WEIGHT_CONFIDENCE
        self.weight_offset_xy = cfg.TRAIN.CRITERION.WEIGHT_OFFSET_XY
        self.weight_offset_z = cfg.TRAIN.CRITERION.WEIGHT_OFFSET_Z

        self.register_buffer('decay_paras', torch.tensor(cfg.TRAIN.CRITERION.DECAY))
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
            return loss_local / loss_local.item() * 0.2 * self.loss
        
        
    
if __name__ == '__main__':
    pass
