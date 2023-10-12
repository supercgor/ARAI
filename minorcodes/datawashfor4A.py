import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from utils import xyz
from multiprocessing import Pool

path = "/Volumes/HEIU/data/ice_8_4A/data 2"
save_path = "/Volumes/HEIU/data/ice_8_4A/hup"
os.makedirs(save_path, exist_ok=True)

def func(name):
    temp, name = name
    print(name)
    typ, pos, charges, ids = xyz.read(os.path.join(path, temp, name))
    if name.split("_")[-2] == "0":
        num = np.array([len(i) for i in pos])
        pos = np.concatenate(pos, axis = 0)
        pos[:, -1] -= pos[:, -1].min()
        pos = np.split(pos, np.cumsum(num)[:-1], axis = 0)
    xyz.write(os.path.join(save_path, temp, name), typ, pos, charges, ids)
    
if __name__ == '__main__': 
    for temp in os.listdir(path):
        if temp.startswith("."):
            continue
        os.makedirs(os.path.join(save_path, temp), exist_ok=True)
        names = [name for name in os.listdir(os.path.join(path, temp)) if name.endswith(".xyz") and not name.startswith(".")]
        with Pool(8) as p:
            p.map(func, zip([temp]*len(names), names))
            
            
