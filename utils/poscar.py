import numpy as np
import torch
import os
from pathlib import Path

from utils.basic import read_POSCAR, read_file
from utils.NMS import nms

class poscar():
    def __init__(self):
        self.info = {}
        self.info['ele'] = ("H","O")
        self.info['scale'] = 1.0
        self.info['lattice'] = torch.tensor([25,25,3])
        self.poscar = ""
        
    def generate_poscar(self,P_pos):
        output = ""
        output += f"{' '.join(self.info['ele'])}\n"
        output += f"{self.info['scale']:3.1f}" + "\n"
        output += f"\t{self.info['lattice'][0].item():.8f} {0:.8f} {0:.8f}\n"
        output += f"\t{0:.8f} {self.info['lattice'][1].item():.8f} {0:.8f}\n"
        output += f"\t{0:.8f} {0:.8f} {self.info['lattice'][2].item():.8f}\n"
        output += f"\t{' '.join(self.info['ele'])}\n"
        output += f"\t{' '.join([int(ele.size(0)) for ele in P_pos])}\n"
        output += f"Selective dynamics\n"
        output += f"Direct\n"
        for i,ele in enumerate(self.info['ele']):
            P_ele = P_pos[i].tolist()
            for atom in P_ele:
                output += f" {atom[0]:.8f} {atom[1]:.8f} {atom[2]:.8f} T T T"
        self.poscar = output
        return
        
    def save(self,pre_path):
        with open(pre_path,'w') as f:
            f.write(self.poscar)
        return
    
def generate_target(info, positions, ele_name, N):
    size = (32, 32, N)
    targets = np.zeros(size + (4 * len(ele_name),))
    for i, ele in enumerate(ele_name):
        target = np.zeros(size + (4,))
        position = positions[ele]
        position = position.dot(np.diag(size))
        for j in range(info['ele_num'][ele]):
            pos = position[j]
            coordinate = np.int_(pos)
            offset = pos - coordinate
            idx_x, idx_y, idx_z = coordinate
            offset_x, offset_y, offset_z = offset
            if idx_z >= N:
                idx_z = N - 1
                offset_z = 1 - 1e-4
            if target[idx_x, idx_y, idx_z, 3] == 0.0:  # overlap
                target[idx_x, idx_y, idx_z] = [offset_x, offset_y, offset_z, 1.0]
            else:
                raise Exception
        targets[..., 4 * i: 4 * (i + 1)] = target
    return targets


def target2positions(target, info, ele_name, threshold=0, NMS=False):
    positions = {}
    for i, ele in enumerate(ele_name):
        target_ele = target[..., 4 * i: 4 * (i + 1)]
        offset = target_ele[..., :3]
        confidence = target_ele[..., 3]
        grid = np.indices(offset.shape[:3]).transpose((1, 2, 3, 0)).astype(offset.dtype)
        position = (grid + offset) / np.asarray(confidence.shape).astype(offset.dtype)
        position = np.concatenate((position, np.expand_dims(confidence, axis=3)), axis=3)
        position = position[confidence > threshold]
        if NMS:
            position = nms(position, ele, info)
        positions[ele] = position
    return positions


# TODO:有问题
def target2positions_O(target, info, ele_name, threshold=0, NMS=False):
    bond_range = (0.8, 1.2)
    positions = {}
    for i, ele in enumerate(ele_name):
        if ele != 'O':
            continue
        target_ele = target[..., 4 * i: 4 * (i + 1)]
        offset = target_ele[..., :3]
        confidence = target_ele[..., 3]
        grid = np.indices(offset.shape[:3]).transpose((1, 2, 3, 0)).astype(offset.dtype)
        position = (grid + offset) / np.asarray(confidence.shape).astype(offset.dtype)
        position = np.concatenate((position, np.expand_dims(confidence, axis=3)), axis=3)
        position = position[confidence > threshold]
        if NMS:
            position = nms(position, ele, info)
        positions[ele] = position
    for i, ele in enumerate(ele_name):
        if ele != 'H':
            continue
        target_ele = target[..., 4 * i: 4 * (i + 1)]
        offset = target_ele[..., :3]
        confidence = target_ele[..., 3]
        grid = np.indices(offset.shape[:3]).transpose((1, 2, 3, 0)).astype(offset.dtype)
        position = (grid + offset) / np.asarray(confidence.shape).astype(offset.dtype)
        position = np.concatenate((position, np.expand_dims(confidence, axis=3)), axis=3)
        position = position[confidence > -3]
        if NMS:
            position = nms(position, ele, info)
        H_list = []
        for pos_O in positions['O']:
            distance = np.sqrt(np.sum(np.square((pos_O[None, :3] - position[:, :3]).dot(info['lattice'])), axis=1))
            mask = (distance > 0.8) & (distance < 1.2)
            position_H = position[mask]
            position_H = sorted(position_H, key=lambda x: x[-1], reverse=True)
            for pos_H in position_H[:2]:
                H_list.append(pos_H)
        positions[ele] = np.array(H_list)
    return positions


def calc_count(targets: dict, positions: dict, info, split):
    ele2r = {'H': 0.528, 'O': 0.74}
    ele2count = {}
    for ele in positions:
        target = targets[ele].dot(info['lattice'])
        position = positions[ele][..., :3].dot(info['lattice'])
        distance_array = np.sqrt(np.sum(np.square(np.expand_dims(target, axis=1) - np.expand_dims(position, axis=0)), axis=2))
        TP = [0 for _ in range(len(split))]
        distance = []
        while distance_array.shape[0] > 0 and distance_array.shape[1] > 0:
            index = np.unravel_index(np.argmin(distance_array), distance_array.shape)
            if distance_array[index] > ele2r[ele]:
                break
            distance.append(distance_array[index])
            # 移除配对的两个原子
            distance_array = np.delete(distance_array, index[0], axis=0)
            distance_array = np.delete(distance_array, index[1], axis=1)
            # 判断配对的原子在上层还是下层
            x, y, z = target[index[0]]
            target = np.delete(target, index[0], axis=0)
            position = np.delete(position, index[1], axis=0)
            for i in range(len(split)):
                low = sum(split) - sum(split[:i + 1])
                high = sum(split) - sum(split[:i])
                if low <= z <= high:
                    TP[i] += 1
                    break
            else:
                raise 'Split Error'
        FP = []
        FN = []
        for i in range(len(split)):
            low = sum(split) - sum(split[:i + 1])
            high = sum(split) - sum(split[:i])
            FP.append(np.sum((low <= position[:, 2]) & (position[:, 2] <= high)))
            FN.append(np.sum((low <= target[:, 2]) & (target[:, 2] <= high)))
        ele2count[ele] = ((TP, FP, FN), np.mean(distance) if distance else 0.0)
    return ele2count


def positions2poscar(positions, info, path_prediction):
    with open(path_prediction, 'w') as file:
        file.write(str(info['comment']))
        file.write(str(info['scale']) + '\n')
        lattice = info["lattice"]
        for i in range(3):
            file.write(f'  \t{lattice[i, 0]:.8f} {lattice[i, 1]:.8f} {lattice[i, 2]:.8f}\n')
        line1 = '\t'
        line2 = '\t'
        for ele in positions.keys():
            position = positions[ele]
            line1 += str(ele) + ' '
            line2 += str(len(position)) + ' '
            try:
                position_array = np.concatenate((position_array, position), axis=0)
            except UnboundLocalError:
                position_array = position
        line1 += '\n'
        line2 += '\n'
        file.write(line1)
        file.write(line2)
        file.write("Selective dynamics\nDirect\n")
        for line in position_array:
            file.write(f' {line[0]:.8f} {line[1]:.8f} {line[2]:.8f} T T T\n')


def pre_tar2xyz(prediction, target, save_path, info, filename):
    ele2r = {'H': 0.528, 'O': 0.74}
    ele2color_cor = {'H': '1 1 1', 'O': '1 0.051 0.051'}  # R G B
    ele2color_wor = {'H': '0 1 0', 'O': '0 0 1'}
    save_path = Path(save_path) / filename
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / f'pre.xyz', 'w') as f_pre:
        with open(save_path / f'tar.xyz', 'w') as f_tar:
            n_pre = 0
            n_tar = 0
            for ele in prediction:
                n_pre += len(prediction[ele])
                n_tar += len(target[ele])
            f_pre.write(f'{n_pre}\n')
            f_pre.write('Lattice="25.0 0.0 0.0 0.0 25.0 0.0 0.0 0.0 7.0"\n')
            f_tar.write(f'{n_tar}\n')
            f_tar.write('Lattice="25.0 0.0 0.0 0.0 25.0 0.0 0.0 0.0 7.0"\n')

            for ele in prediction:
                pos_pre = prediction[ele][..., :3].dot(info['lattice'])
                pos_tar = target[ele].dot(info['lattice'])
                distance_array = np.sqrt(np.sum(np.square(np.expand_dims(pos_tar, axis=1) - np.expand_dims(pos_pre, axis=0)), axis=2))
                while distance_array.shape[0] > 0 and distance_array.shape[1] > 0:
                    index = np.unravel_index(np.argmin(distance_array), distance_array.shape)
                    if distance_array[index] > ele2r[ele]:
                        break
                    # 移除配对的两个原子
                    distance_array = np.delete(distance_array, index[0], axis=0)
                    distance_array = np.delete(distance_array, index[1], axis=1)
                    x_tar, y_tar, z_tar = pos_tar[index[0]]
                    x_pre, y_pre, z_pre = pos_pre[index[1]]
                    f_tar.write(f'{ele} {x_tar:.8f} {y_tar:.8f} {z_tar:.8f} {ele2color_cor[ele]}\n')
                    f_pre.write(f'{ele} {x_pre:.8f} {y_pre:.8f} {z_pre:.8f} {ele2color_cor[ele]}\n')
                    pos_tar = np.delete(pos_tar, index[0], axis=0)
                    pos_pre = np.delete(pos_pre, index[1], axis=0)
                for x_tar, y_tar, z_tar in pos_tar:
                    f_tar.write(f'{ele} {x_tar:.8f} {y_tar:.8f} {z_tar:.8f} {ele2color_wor[ele]}\n')
                for x_pre, y_pre, z_pre in pos_pre:
                    f_pre.write(f'{ele} {x_pre:.8f} {y_pre:.8f} {z_pre:.8f} {ele2color_wor[ele]}\n')


def show_poscar(path_ovito,a, path_poscar):
    os.chdir(path_ovito)
    os.system(f'./ovito {a} {path_poscar}')


def main():
    path_ovito = r'D:\Software\OVITO Basic'
    path = r'D:\data\afm_ml_2d_t220\test'

    root_path = r'D:\data\afm_ml_2d_t220'
    filenames = read_file(os.path.join(root_path, 'FileList'))


if __name__ == '__main__':
    main()
