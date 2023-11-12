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


class conditionVAELoss(nn.Module):
    def __init__(self, wc = 0.4, wpos_weight = 10.0, wpos = 0.3, wr = 0.3, wvae = 1.0):
        super().__init__()
        self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([wpos_weight]))
        self.pos_loss = nn.MSELoss()
        self.rot_loss = nn.MSELoss()
        self.wc = wc
        self.wpos = wpos
        self.wr = wr
        self.wvae = wvae
    
    # https://github.com/AntixK/PyTorch-VAE/blob/master/configs/vae.yaml
    def vaeloss(self, mu, logvar):
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim = 1).mean()
    
    def forward(self, pred, targ, mu, logvar):
        pd_conf, pd_pos, pd_rt = torch.split(pred, [1, 3, 6], dim = -1)
        tg_conf, tg_pos, tg_rt = torch.split(targ, [1, 3, 6], dim = -1)
        loss_wc = self.wc * self.cls_loss(pd_conf, tg_conf)
        mask = tg_conf[...,0] > 0.5
        loss_pos = self.wpos * self.pos_loss(pd_pos[mask], tg_pos[mask])
        loss_r = self.wr * self.rot_loss(pd_rt[mask], tg_rt[mask])
        loss_vae = self.wvae * self.vaeloss(mu, logvar)
        return loss_wc, loss_pos, loss_r, loss_vae
        