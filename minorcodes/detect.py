import os, sys, tqdm, h5py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from fileio.asehdf import load_by_name
from ase import io
from ase.visualize import view
from torch.utils.data import Dataset, DataLoader
import multiprocessing  
import logging
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Align point clouds')
    parser.add_argument('--start', type=int, help='Start index', default=0)
    parser.add_argument('--end', type=int, help='End index', default=-1)
    args = parser.parse_args()
    return args

def rotate_z(theta, points):
    """在Z軸上旋轉點雲"""
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return np.dot(points, rotation_matrix)

def objective_function(params, source_points, target_tree):
    """目標函數，計算平移和旋轉後的點雲與目標點雲之間的距離和"""
    dx, dy, dz, theta = params
    dx, dy = map_hex_points(np.array([[dx, dy]]), 7.35357 / 2)[0]
    translated_points = source_points + np.array([dx, dy, dz])
    rotated_points = rotate_z(theta, translated_points)
    distances, _ = target_tree.query(rotated_points)
    return np.sum(distances)

def align_point_clouds(input_atoms, target_tree):
    """使用差分進化算法配準點雲"""
    
    # 定義參數範圍：平移範圍和旋轉角度
    a = 7.82284 / 3
    z = 7.35357 # 3.676785#7.35357 / 2
    bounds = [(-0.5*a, a), (-0.5 * np.sqrt(3) * a, + 0.5 * np.sqrt(3) * a), (0, z/4), (-np.pi/3, np.pi/3)]
    
    # 使用差分進化算法尋找最優參數
    result = differential_evolution(
        objective_function,
        bounds,
        args=(input_atoms.positions, target_tree),
        strategy='best1bin',
        maxiter=1000,
        popsize=75,
        tol=0.01,
        mutation=(0.5, 1.0),
        recombination=0.9,        
    )
    
    optimal_params = result.x
    optimal_params[0:2] = map_hex_points(optimal_params[None,0:2], a)[0]
    optimal_params[:2] /= a
    # print(f"Optimal Parameters: {optimal_params}")
    # print(f"Minimum Distance Sum: {result.fun}")
    
    # 使用找到的最優參數對源點雲進行變換
    return optimal_params, result.fun

def map_hex_points(P, a):
    """將其六邊形的空間平均不等性去除，所有點會映射回中心的六邊形"""
    # 转换到六边形网格的坐标系
    q = (2./3 * P[:, 0]) / a
    r = (-1./3 * P[:, 0] + np.sqrt(3)/3 * P[:, 1]) / a

    # 转换到立方体坐标
    x_cube = q
    z_cube = r
    y_cube = -x_cube - z_cube

    # 四舍五入找到最近的网格点
    rx = np.round(x_cube)
    ry = np.round(y_cube)
    rz = np.round(z_cube)

    # 调整坐标以补偿四舍五入的误差
    x_diff = np.abs(rx - x_cube)
    y_diff = np.abs(ry - y_cube)
    z_diff = np.abs(rz - z_cube)
    
    rx = np.where((x_diff > y_diff) & (x_diff > z_diff), -ry-rz, rx)
    # ry = np.where((x_diff <= y_diff) & (y_diff > z_diff), -rx-rz, ry)
    rz = np.where((x_diff <= z_diff) & (y_diff <= z_diff), -rx-ry, rz)

    # 将立方体坐标转换回原始坐标系
    hex_x = (3./2 * rx) * a
    hex_y = (np.sqrt(3)/2 * rx + np.sqrt(3) * rz) * a
    return P - np.column_stack((hex_x, hex_y))

class MyDataset(Dataset):
    def __init__(self, path, tree):
        self.path = path
        self.tree = tree
        self.ishdf = path.endswith('.hdf5')
        if self.ishdf:
            with h5py.File(self.path, 'r') as f:
                self.keys = list(f.keys())
        else:
            self.keys = [os.path.join(p, f) for p, _, files in os.walk(path) for f in files]
            self.keys = list(filter(lambda x: x.endswith('xyz') or x.endswith('poscar'), self.keys))
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        k = self.keys[idx]
        if self.ishdf:
            with h5py.File(self.path, 'r') as f:
                atoms = load_by_name(f, k)
                atoms = atoms[atoms.symbols == 'O']
                atoms = atoms[atoms.positions[:, 2] > 4]
                atoms.positions[:, :2] -= atoms.cell[(0,1), (0, 1)] / 2
                atoms.positions[:, 2] -= 4.0
        else:
            atoms = io.read(k)
            atoms = atoms[atoms.symbols == 'O']
            atoms.positions[:, 2] = atoms.cell[2, 2] - atoms.positions[:, 2]
            atoms = atoms[atoms.positions[:, 2] > 4]
            atoms.positions[:, :2] -= atoms.cell[(0,1), (0, 1)] / 2
            atoms.positions[:, 2] -= 4.0            
        
        return idx, k, atoms

def find_best(atoms, atree, s_tag=70.0):
    best_param = None
    best_score = np.inf
    for i in range(5):
        param, score = align_point_clouds(atoms, atree)
        if score < best_score:
            best_param = param
            best_score = score
        if i >= 3 and score < s_tag:
            break
    dx, dy, dz, theta = best_param
    atoms.positions += np.array([dx, dy, dz])
    atoms.positions = rotate_z(theta, atoms.positions)
    return atoms, best_param, best_score

def process_item(atom, atree):
    atoms, param, score = find_best(atom, atree)
    return param, score

if __name__ == '__main__':
    args = parse_args()
    ats = io.read('minorcodes/1h_basea_cleaned.poscar')
    ats = ats[ats.symbols == 'O']
    ats.symbols = 'N'
    ats.positions[:, :2] -= ats.cell[(0,1), (0, 1)] / 2
    atree = cKDTree(ats.positions)
    # dts = MyDataset('outputs/2024-03-04/15-33-25/', atree)
    dts = MyDataset('outputs/2024-03-04/15-34-13/', atree)
    print(len(dts))
    np.set_printoptions(precision=3, suppress=True)
    all_scores = {}
    all_params = {}
    s_tag = 65.0
    j = 0
    results = []
    print(f"Start: {args.start}, End: {args.end}")
    if args.end == -1:
        args.end = len(dts)
    for i in range(args.start, args.end):
        if i >= len(dts):
            break
        id, key, atom = dts[i]
        param, score = process_item(atom, atree)
        all_params[key] = param
        all_scores[key] = score
        correct = int(key.split("_")[7]) % 60 - 30
        pred = int(param[-1] * 180 / np.pi) % 60
        print(f"{key} | {score:.3f} | {param} | {correct} | {pred} | delta {(correct - pred + 30) % 60 - 30}")
        results.append((correct - pred + 30) % 60 - 30)
        # view(ats + atom, block=True)
        j+=1
        if j % 100 == 0:
            print(f"{args.start} - {args.end} | {j + args.start}")

    print(results)
    plt.hist(results, bins=60)
    plt.show()
    np.savez(f'align_result_{args.start}_{args.end}.npz', **all_params)