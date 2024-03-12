import torch
from torch import nn
import torch.nn.functional as F

class BoxClsLoss2(nn.Module):
    def __init__(self, cls_weight, xy_weight, z_weight, rot_weight, pos_weight):
        super().__init__()
        self.register_buffer("pos_weight", torch.as_tensor(pos_weight))
        self.cls_weight = cls_weight
        self.xy_weight = xy_weight
        self.z_weight = z_weight
        self.rot_weight = rot_weight
        
    def forward(self, preds, targs):
        # preds: B, X, Y, Z, 10 ; targs: B, X, Y, Z, 10
        predc, predxy, predz, predrot = torch.split(preds, [1, 2, 1, 6], dim = -1)
        targc, targxy, targz, targrot = torch.split(targs, [1, 2, 1, 6], dim = -1)
        mask = targc[...,0] > 0.5
        lossc = F.binary_cross_entropy_with_logits(predc, targc, pos_weight=self.pos_weight) * self.cls_weight
        lossxy = F.binary_cross_entropy(predxy[mask], targxy[mask]) * self.xy_weight
        lossz = F.binary_cross_entropy(predz[mask], targz[mask]) * self.z_weight
        lossrot = F.mse_loss(predrot[mask], targrot[mask]) * self.rot_weight
        return lossc + lossxy + lossz + lossrot