import torch
from torch import Tensor
from ase import Atoms

from . import lib
from torchmetrics import Metric

class confusion_matrix(Metric):
    def __init__(self, 
                 real_size: tuple[float, ...] = (25.0, 25.0, 4.0), 
                 match_distance: float = 1.0, 
                 split: list[float] = [0.0, 4.0, 8.0]
                 ):
        super().__init__()
        self.register_buffer("_real_size", torch.as_tensor(real_size, dtype=torch.float))
        # TP, FP, FN, AP, AR, ACC, SUC
        self.add_state("matrix", default = torch.zeros((len(split) - 1, 7), dtype = torch.float), dist_reduce_fx = "sum")
        self.add_state("total", default = torch.tensor([0]), dist_reduce_fx = "sum")
        self.match_distance = match_distance
        self.split = split
        self.split[-1] += 1e-5
    
    def update(self, preds: list[Atoms], targs: list[Atoms]):
        if isinstance(preds, Atoms):
            preds = [preds]
        if isinstance(targs, Atoms):
            targs = [targs]
            
        for b, (pred, targ) in enumerate(zip(preds, targs)):
            pred = pred.get_positions()
            targ = targ.get_positions()
            
            pd_match_ids, tg_match_ids = lib.argmatch(pred, targ, self.match_distance)
                
            match_tg_pos = targ[tg_match_ids] # N 3
            
            for i, (low, high) in enumerate(zip(self.split[:-1], self.split[1:])):
                num_match = ((match_tg_pos[:, 2] >= low) & (match_tg_pos[:, 2] < high)).sum() # TP
                num_pd = ((pred[:, 2] >= low) & (pred[:, 2] < high)).sum() # P
                num_tg = ((targ[:, 2] >= low) & (targ[:, 2] < high)).sum() # T
                self.matrix[i, 0] += num_match #TP
                self.matrix[i, 1] += num_pd - num_match # FP
                self.matrix[i, 2] += num_tg - num_match # FN
                self.matrix[i, 3] += num_match / num_pd if num_pd > 0 else 0 # AP
                self.matrix[i, 4] += num_match / num_tg if num_tg > 0 else 0 # AR
                self.matrix[i, 5] += num_match / (num_pd + num_tg - num_match) if num_pd + num_tg - num_match > 0 else 0 # ACC
                self.matrix[i, 6] += (num_match == num_pd) & (num_match == num_tg) # SUC
                
            self.total += 1
                    
    def compute(self):
        out = self.matrix.clone()
        out[:, 3:] = out[:, 3:] / self.total
        return out