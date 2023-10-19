import torch
import time
from torch import multiprocessing as mp
class test():
    def __init__(self):
        self.a = 10.0
        
    def __call__(self, xs):
        p = mp.Pool(6)
        xs, ys = p.map_async(self._forward, xs)
        return xs, ys
    
    @staticmethod
    def _forward(x):
        return 10 * x, 10 * x

if __name__ == "__main__":
    test= test()
    a = [torch.ones(1, 5, device="cpu") for _ in range(10000)]
    pp = test(a)
    time.sleep(1)
    print(pp.get())