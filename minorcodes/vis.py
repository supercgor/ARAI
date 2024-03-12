import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm

from scipy.optimize import differential_evolution
from utils import xyz
from utils.lib import rotate, replicate, encodewater, group_as_water, vec2box
from utils.const import ice_search_bound, ice_unit_cell, ice_axis
from dataset.crystal import match_ice_crystal

target_dir = '../data/ice_16A_R/data-2'
tar_temp = os.listdir(target_dir)
tar_temp = [temp for temp in tar_temp if not temp.startswith(".")]

fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(1, 5, 1)
ax1.set_xlabel(r'$\Delta R_z$')
ax1.set_xlim(-60,+60)
ax2 = fig.add_subplot(1, 5, 2)
ax2.set_xlabel(r'O_x')
ax2.set_xlim(*ice_search_bound[3])
ax3 = fig.add_subplot(1, 5, 3)
ax3.set_xlabel(r'O_y')
ax3.set_xlim(*ice_search_bound[4])
ax4 = fig.add_subplot(1, 5, 4)
ax4.set_xlabel(r'O_z')
ax4.set_xlim(*ice_search_bound[5])
ax5 = fig.add_subplot(1, 5, 5)
ax5.set_xlabel(r'Objective Function')

for j, temp in enumerate(tar_temp):
    all_files_names = os.listdir(os.path.join(target_dir, temp))
    rot = []
    offsets = []
    best_results = []
    not_good = []
    for i, f in enumerate(tqdm.tqdm(all_files_names, desc=temp)):
        types, pos, charges, ids = xyz.read(os.path.join(target_dir, temp, f))
        
        ref = float(f.replace(".xyz", "").split("_")[-1])
        
        pos = np.reshape(pos, (-1, 3))
        pos = pos[::3]
        pos[:, 2] *= -1
        pos = pos[pos[:, 2] > 8]
        
        dic = match_ice_crystal(pos, origin = [12.5, 12.5, 12.0])
        # print("平移向量:", translation_vector)
        rot.append(dic['rotvec'][2] - ref % 60)
        offsets.append(dic['transvec'])
        best_results.append(dic['cost'])
        
    offsets = np.asarray(offsets)
    ax1.hist(rot, bins=40, label=temp)
    ax2.hist(offsets[:, 0], bins=40, label=temp)
    ax3.hist(offsets[:, 1], bins=40, label=temp)
    ax4.hist(offsets[:, 2], bins=40, label=temp)
    ax5.hist(best_results, bins=40, label=temp)
    
    # Z_matched = rotate(Z + translation_vector, rotation_matrix)
    # matched_offset = rotation_matrix.dot(offset)

    # ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_xlim(-12.5, 12.5)
    # ax.set_ylim(-12.5, 12.5)
    # ax.set_zlim(-4, 4)
    # ax.set_aspect('equal', adjustable='box')

    # ax.scatter(atoms[:,0], atoms[:,1], atoms[:,2], s=20, c='g', label='unit cell')
    # ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=20, c='r', label='real ice')
    # ax.scatter(Z_matched[:,0], Z_matched[:,1], Z_matched[:,2], s=20, c='b', label='matched')
    # ax.legend()
    # ax = fig.add_subplot(122, projection='3d')
    # ax.scatter(atoms[:,0], atoms[:,1], atoms[:,2], s=120, c='r')
    # ax.set_aspect('equal')
    
plt.legend()
plt.show()
