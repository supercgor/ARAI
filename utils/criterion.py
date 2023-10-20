import torch
from torch import nn, Tensor
from einops import rearrange

class BoxClsLoss(nn.Module):
    def __init__(self, wxy: float = 0.5, wz: float = 0.5, wcls: float = 1.0, ion_weight: tuple[float] = (0.1, 1.0, 2.0)):
        super().__init__()
        self.wxy = wxy
        self.wz = wz
        self.wcls = wcls
        
        self.nll = nn.NLLLoss(torch.as_tensor(ion_weight))
        self.mse = nn.MSELoss(reduction="none")
        
        
    def forward(self, pred_clses: Tensor, pred_boxes: Tensor, targ_clses: Tensor, targ_boxes: Tensor) -> Tensor:
        """
        _summary_

        Args:
            pred_clses (Tensor): B 3 D H W
            pred_boxes (Tensor): B 3 D H W
            targ_clses (Tensor): B D H W 
            targ_boxes (Tensor): B D H W 3

        Returns:
            Tensor: _description_
        """
        pred_clses = pred_clses.log_softmax(dim = 1)
        pred_boxes = pred_boxes.permute(0, 2, 3, 4, 1).sigmoid()
        
        cls_loss = self.nll(pred_clses, targ_clses)
        mask = targ_clses > 0.5
        pos_loss = self.mse(pred_boxes[mask], targ_boxes[mask]).mean(dim = 0)
        return cls_loss * self.wcls + pos_loss[0] * self.wz + pos_loss[1:].mean() * self.wxy
    
class conditionVAELoss(nn.Module):
    def __init__(self, wc = 0.4, wpos_weight = 10.0, wpos = 0.3, wr = 0.3, wvae = 1.0):
        super().__init__()
        self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([wpos_weight]))
        self.pos_loss = nn.MSELoss()
        self.rot_loss = nn.L1Loss()
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
        