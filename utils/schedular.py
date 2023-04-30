from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from math import exp

class Scheduler(_LRScheduler):
    def __init__(self, 
                 optimizer: Optimizer,
                 warmup_steps: int = 1000,
                 decay_factor: int = 4000,
                 last_epoch: int=-1,
                 verbose: bool=False) -> None:

        self.max_lr = optimizer.param_groups[0]['lr']
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)
        self.k = 1 / decay_factor
        
        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> float:
        lr = calc_lr(self._step_count, self.max_lr, self.warmup_steps, self.k)
        return [lr] * self.num_param_groups


def calc_lr(step, max_lr, warmup_steps, k):
    if step <= warmup_steps:
        return max_lr * step / warmup_steps
    else:
        return max_lr * exp(k * (warmup_steps - step))