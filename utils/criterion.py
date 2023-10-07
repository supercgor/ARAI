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