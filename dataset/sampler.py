import random
import numpy as np

def z_sampler(use: int, total: int, is_rand: bool = False) -> tuple[int]:
    if is_rand:
        sp = random.sample(range(total), k=(use % total))
        return [i for i in range(total) for _ in range(use // total + (i in sp))]
    else:
        return [i for i in range(total) for _ in range((use // total + ((use % total) > i)))]
