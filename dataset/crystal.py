import h5py, dgl, ase, torch
import numpy as np
import random

from torch.utils.data import Dataset
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution

from utils.lib import rotate, replicate, encodewater, group_as_water, vec2box
from utils.const import ice_search_bound, ice_unit_cell, ice_axis
from fileio.asehdf import load_by_name

class Layer2CrystalDataset(Dataset):
    def __init__(self, path, transform = None):
        self.transform = transform
        self.path = path
        with h5py.File(path, 'r') as f:
            self.keys = list(f.keys())
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        with h5py.File(self.path, 'r') as f:
            name = self.keys[index]
            atoms = load_by_name(f, name)
            cell = atoms.cell / 2 + [0, 0, 4]
            # randomly offset the z axis
            noisez = random.uniform(0, 2)
            atoms.positions[:,2] += noisez
            atoms = atoms[atoms.positions[:, 2] < atoms.cell[2,2]]
            
            # match ice crystal at z = [8.0, 16.0]
            dic = match_ice_crystal(atoms.positions[atoms.symbols == 'O'], origin = np.diag(cell))
            rot = float(name.split("_")[-1])
            off = np.append(rot, dic['transvec'])
            y = torch.tensor(off, dtype = torch.float)
            y[0] = torch.deg2rad(y[0] % 60 - 30)
            
            periodic = ice_search_bound[2:]
            periodic[0] = periodic[0][1] / 2
            periodic[1] = periodic[1][1]
            periodic[2] = periodic[2][1]
            periodic[3] = periodic[3][1]
            periodic = torch.tensor(periodic)
            y /= periodic
            
            # generate toplayer input: water encoded (N, 9)
            top_atoms = atoms[(atoms.positions[:, 2] < 4)]
            cell = np.diag(top_atoms.cell).copy()
            cell[2] = 4.0
            
            top_atoms = group_as_water(torch.as_tensor(top_atoms.positions[top_atoms.symbols == 'O']), 
                                    torch.as_tensor(top_atoms.positions[top_atoms.symbols == 'H']))
            
            top_atoms = encodewater(top_atoms)
            g = dgl.knn_graph(top_atoms[:,:3], 6)
            
            top_atoms[:,:3] /= torch.as_tensor(cell)
            top_grid = vec2box(top_atoms[:,:3], top_atoms[:, 3:], (25, 25, 4))
            
            # generate middle layer input: water encoded (N, 9)
            middle_atoms = atoms[(atoms.positions[:, 2] < 8) & (atoms.positions[:, 2] >= 4)]
            cell = np.diag(middle_atoms.cell).copy()
            cell[2] = 4.0
            middle_atoms.positions[:,2] -= 4
            middle_atoms = group_as_water(torch.as_tensor(middle_atoms.positions[middle_atoms.symbols == 'O']), 
                                    torch.as_tensor(middle_atoms.positions[middle_atoms.symbols == 'H']))
            middle_atoms = encodewater(middle_atoms)
            middle_atoms[:,:3] /= torch.as_tensor(cell)
            
            middle_grid = vec2box(middle_atoms[:,:3], middle_atoms[:, 3:], (25, 25, 4))
            
            return top_atoms.float(), top_grid.float(), middle_grid.float(), g, y.float()


class RotateOnly(Dataset):
    def __init__(self, path, transform = None):
        self.transform = transform
        self.path = path
        with h5py.File(path, 'r') as f:
            self.keys = list(f.keys())
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        with h5py.File(self.path, 'r') as f:
            name = self.keys[index]
            atoms = load_by_name(f, name)
            cell = atoms.cell / 2 + [0, 0, 4]
            # randomly offset the z axis
            noisez = random.uniform(0, 2)
            atoms.positions[:,2] += noisez
            atoms = atoms[atoms.positions[:, 2] < atoms.cell[2,2]]
            
            # match ice crystal at z = [8.0, 16.0]
            rot = (float(name.split("_")[-1]) % 60 - 30) / 30
            y = torch.tensor([rot], dtype = torch.float)
            
            # generate toplayer input: water encoded (N, 9)
            top_atoms = atoms[(atoms.positions[:, 2] < 4)]
            cell = np.diag(top_atoms.cell).copy()
            cell[2] = 8.0
            
            top_atoms = group_as_water(torch.as_tensor(top_atoms.positions[top_atoms.symbols == 'O']), 
                                    torch.as_tensor(top_atoms.positions[top_atoms.symbols == 'H']))
            
            top_atoms = encodewater(top_atoms)
            g = dgl.knn_graph(top_atoms[:,:3], 6)
            
            top_atoms[:,:3] /= torch.as_tensor(cell)
            
            return top_atoms.float(), None, None, g, y

def match_ice_crystal(bulk, repeat = (3, 3, 1), origin: str | np.ndarray = 'mean'):
    if isinstance(origin, str):
        if origin == 'mean':
            origin = np.mean(bulk, axis = 0)
    cell_pos = replicate(ice_unit_cell, repeat, ice_axis) # mean 0
    return _optimize_rotation_and_translation_de(cell_pos, bulk-origin)

def _optimize_rotation_and_translation_de(Z, R_points):
    xmin = np.array([-12.5, -12.5, -4])
    xmax = np.array([ 12.5,  12.5,  4])
    def _objective_function(params, Z, R_points):
        # 將參數分解為旋轉向量和平移向量
        rot_vec, trans_vec = params[:3], params[3:]

        # 使用旋轉向量和平移向量對 Z 進行變換
        transformed_Z = rotate(Z + trans_vec, rot_vec)
        
        mask = np.all(transformed_Z > xmin, axis=1) & np.all(transformed_Z < xmax, axis=1)
        
        # 計算變換後的點集 Z 與點集 R 之間的最近點距離
        distances = cdist(transformed_Z, R_points)
        min_distances = np.min(distances, axis=1)
        min_distances[~mask] = 1
        # 返回距離之和作為代價
        return np.mean(min_distances)
    
    # 使用差分进化算法进行全局优化
    i = 0
    try_tol = 3
    now_tol = 0
    best = np.inf
    best_result = None
    while best > 0.9 and i < 20:
        i += 1
        result = differential_evolution(_objective_function, ice_search_bound, args=(Z, R_points), vectorized=False)
        if result.fun < best:
            best = result.fun
            best_result = result
            now_tol = 0
        else:
            now_tol += 1
            if now_tol > try_tol:
                break
            
    best_params = best_result.x

    return {'rotvec': np.rad2deg(best_params[:3]), 'transvec': best_params[3:], 'cost': best_result.fun}