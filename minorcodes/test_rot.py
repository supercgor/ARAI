import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from utils import xyz
from utils.library import decodeWater, encodeWater

path = "/Users/supercgor/Documents/data/ice_8_4A/hup/t160/ice_basal_T160_1000000_0_0_0_8A.xyz"


types, molecules, _, _ = xyz.read(path)

import timeit

molecules = np.array(molecules)
print("torch", timeit.timeit(lambda: decodeWater(torch.as_tensor(molecules)), number=1))
print("numpy", timeit.timeit(lambda: decodeWater(molecules), number=1))
rot = decodeWater(np.array(molecules))
make = encodeWater(rot)
print(make)

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 取出矩陣的每一列的數據
ax.scatter(make[...,0], make[...,1], make[...,2], c='r', marker='o')
ax.scatter(make[...,3], make[...,4], make[...,5], c='b', marker='o')
ax.scatter(make[...,6], make[...,7], make[...,8], c='b', marker='o')
# ax.scatter(rot[...,6], rot[...,7], rot[...,8], c='r', marker='o')
# ax.scatter(rot[...,3], rot[...,4], rot[...,5], c='b', marker='o')
# ratio
ax.set_aspect('equal', adjustable='box')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


plt.show()