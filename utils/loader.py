import os
import json
import torch
import cv2
import time
import logging
import logging.handlers
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from collections import OrderedDict

def Loader(cfg, make_dir=True):
    # 這一部份會決定是否先產生一個新的文件夾
    load_name = cfg.model.checkpoint
    if load_name != "":
        if os.path.exists(f"{cfg.path.check_root}/{load_name}"):
            f = open(f"{cfg.path.check_root}/{load_name}/info.json")
        else:
            raise NameError(f"No model in {cfg.path.check_root} is called {load_name}")
    else:
        f = open(f"{cfg.path.check_root}/default_info.json")
    
    info_dict = json.load(f)
    f.close()
    
    if make_dir:
        i = 0
        while True:
            name = f"{cfg.model.net}-{time.strftime('%m%d-%H')}-{i}"
            if info_dict['best'] != "" and os.path.exists(f"{cfg.path.check_root}/{name}"):
                i += 1
            else:
                if not os.path.exists(f"{cfg.path.check_root}/{name}"):
                    os.mkdir(f"{cfg.path.check_root}/{name}")
                break
    else:
        name = load_name

    cfg.merge_from_dict(info_dict)

    with open(f"{cfg.path.check_root}/{name}/info.json", "w") as f:
        json.dump(info_dict, f, indent = 4)
    
    load_dir = f"{cfg.path.check_root}/{load_name}"
    work_dir = f"{cfg.path.check_root}/{name}"
        
    return load_dir, work_dir

class sampler():
    def __init__(self, name, path="/home/supercgor/gitfile/data"):
        self.abs_path = f"{path}/{name}"
        if not os.path.exists(self.abs_path):
            raise FileNotFoundError(f"Not such dataset in {self.abs_path}")
        self.datalist = os.listdir(f"{self.abs_path}/afm")

    def __getitem__(self, index):
        img_path = f"{self.abs_path}/afm/{self.datalist[index]}"
        pl = poscarLoader(f"{self.abs_path}/label")
        info, positions = pl.load(f"{self.datalist[index]}.poscar")
        images = []
        for path in sorted(os.listdir(img_path), key=lambda x: int(x.split(".")[0])):
            img = cv2.imread(f"{img_path}/{path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)

        return {"name": self.datalist[index], "info": info, "image": images, "position": positions}

    def get(self, name):
        index = self.datalist.index(name)
        return self.__getitem__(index)

    def get_npy(self, index):
        loc = f"{self.abs_path}/npy/{self.datalist[index]}.npy"
        pred = np.load(loc)
        return pred

    def __len__(self):
        return len(self.datalist)

    def __next__(self):
        for i in range(self.__len__):
            return self.__getitem__(i)


class poscarLoader():
    def __init__(self, path, model_name="", lattice=(25, 25, 3), out_size=(32, 32, 4), elem=("O", "H"), cutoff=OrderedDict(O=2.2, H=0.8)):
        self.path = path
        self.model_name = model_name
        self.lattice = np.asarray(lattice)
        self.out_size = np.asarray(out_size)
        self.elem = elem
        self.cutoff = cutoff
        self.zoom = [i/j for i, j in zip(lattice, out_size)]

    def load(self, name, NMS=True):
        """Load the poscar file or npy file. For npy file the Tensor should have the shape of ( B, X, Y, Z, 8).

        Args:
            name (str): file name

        Returns:
            info: dict with keys: 'scale': 1.0, 'lattice': diag_matrix, 'elem_num': 2, 'ele_name': ('O', 'H'), 'comment'
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"No such directory: {self.path}")

        if name.split(".")[1] == "npy":
            return {'scale': 1.0,
                    'lattice': np.diag(self.lattice),
                    'ele_num': len(self.elem),
                    'ele_name': self.elem,
                    'comment': ""}, self._load_npy(name, NMS=NMS)

        with open(f"{self.path}/{name}") as fr:
            comment = fr.readline().split("\x00")[0]
            line = fr.readline()
            scale_length = float(_clean(line)[0])
            lattice = []
            for _ in range(3):
                lattice.append(_clean(fr.readline()).astype(float))
            lattice = np.array(lattice)
            ele_name = _clean(fr.readline())
            counts = _clean(fr.readline()).astype(int)
            ele_num = dict(zip(ele_name, counts))
            fr.readline()
            fr.readline()
            positions = {}
            for ele in ele_name:
                position = []
                for _ in range(ele_num[ele]):
                    line = _clean(fr.readline())
                    position.append(line[:3].astype(float))
                positions[ele] = np.asarray(position)
        info = {'scale': scale_length,
                'lattice': lattice,
                'ele_num': ele_num,
                'ele_name': tuple(ele_name),
                'comment': comment}
        return info, positions

    def _load_npy(self, name, NMS=True, conf=0.7):
        pred = np.load(f"{self.path}/{name}")  # ( X, Y, Z, 8 )
        return self.npy2pos(pred, NMS=NMS, conf=conf)

    def npy2pos(self, pred, NMS=True, conf=0.7):
        pred_box = pred.shape[:3]
        ind = np.indices(pred_box)
        ind = np.transpose(ind, (1, 2, 3, 0))
        pred = pred.cpu().numpy()
        pred = pred.reshape((*pred_box, 2, 4))
        pred = np.transpose(pred, (3, 0, 1, 2, 4))
        pred[..., :3] = (pred[..., :3] + ind) * self.zoom
        out = {}
        for elem, submat in zip(self.cutoff, pred):
            select = submat[..., 3] > conf
            offset = submat[select]
            offset = offset[np.argsort(offset[..., 3])][::-1]
            if NMS:
                offset = self.nms(offset, self.cutoff[elem])
            out[elem] = offset[..., :3]
        return out

    @staticmethod
    def nms(pos, cutoff):
        reduced_index = np.full(pos.shape[0], True)
        dis_mat = cdist(pos[..., :3], pos[..., :3]) < cutoff
        dis_mat = np.triu(dis_mat, k=1)
        trues = dis_mat.nonzero()
        for a, b in zip(*trues):
            if reduced_index[a]:
                reduced_index[b] = False
        return pos[reduced_index]

    def save(self, name, pos):
        output = ""
        output += f"{' '.join(self.elem)}\n"
        output += f"{1:3.1f}" + "\n"
        output += f"\t{self.lattice[0]:.8f} {0:.8f} {0:.8f}\n"
        output += f"\t{0:.8f} {self.lattice[1]:.8f} {0:.8f}\n"
        output += f"\t{0:.8f} {0:.8f} {self.lattice[2]:.8f}\n"
        output += f"\t{' '.join([str(ele) for ele in pos])}\n"
        output += f"\t{' '.join([str(pos[ele].shape[0]) for ele in pos])}\n"
        output += f"Selective dynamics\n"
        output += f"Direct\n"
        for ele in pos:
            p = pos[ele]
            for a in p:
                output += f" {a[0]/self.lattice[0]:.8f} {a[1]/self.lattice[1]:.8f} {a[2]/self.lattice[2]:.8f} T T T\n"

        path = f"{self.path}/result/{self.model_name}"
        if not os.path.exists(path):
            os.mkdir(path)

        with open(f"{path}/{name}", 'w') as f:
            f.write(output)
        return

    def save4npy(self, name, pred, NMS=True, conf=0.7):
        return self.save(name, self.npy2pos(pred, NMS=NMS, conf=conf))
    
    @classmethod
    def pos2poscar(self, 
                   path,                    # the path to save poscar
                   points_dict,             # the orderdict of the elems : {"O": N *　(Z,X,Y)}
                   real_size = (3, 25, 25)  # the real size of the box
                   ):
        output = ""
        output += f"{' '.join(points_dict.keys())}\n"
        output += f"{1:3.1f}" + "\n"
        output += f"\t{real_size[1]:.8f} {0:.8f} {0:.8f}\n"
        output += f"\t{0:.8f} {real_size[2]:.8f} {0:.8f}\n"
        output += f"\t{0:.8f} {0:.8f} {real_size[0]:.8f}\n"
        output += f"\t{' '.join(points_dict.keys())}\n"
        output += f"\t{' '.join(str(len(i)) for i in points_dict.values())}\n"
        output += f"Selective dynamics\n"
        output += f"Direct\n"
        for ele in points_dict:
            p = points_dict[ele]
            for a in p:
                output += f" {a[1]/real_size[1]:.8f} {a[2]/real_size[2]:.8f} {a[0]/real_size[0]:.8f} T T T\n"

        with open(path, 'w') as f:
            f.write(output)
            
        return
    
    @classmethod
    def poscar2pos(self, path):
        with open(path, 'r') as f:
            f.readline()
            scale = float(f.readline())
            x = float(f.readline().split(" ")[0])
            y = float(f.readline().split(" ")[1])
            z = float(f.readline().split(" ")[2])
            real_size = (z, x, y)
            elem = OrderedDict((i,int(j)) for i,j in zip(f.readline()[1:-1].split(" "),f.readline().split(" ")))
            f.readline()
            f.readline()
            pos = OrderedDict((e, []) for e in elem.keys())
            for e in elem:
                for i in range(elem[e]):
                    X,Y,Z = map(float,f.readline().split(" ")[1:4])
                    pos[e].append([Z * z, X * x, Y * y])
                pos[e] = torch.tensor(pos[e])
        return scale, real_size, elem, pos
    
    @classmethod
    def pos2box(self, points_dict, real_size, out_size):
        expand = torch.tensor(tuple(i/j for i,j in zip(out_size, real_size)))
        OUT = torch.zeros(len(points_dict),*out_size, 4)
        for i, e in enumerate(points_dict):
            POS = points_dict[e] * expand
            IND = POS.int()
            offset = POS - IND
            ONE = torch.ones((offset.shape[0],1))
            offset = torch.cat((offset, ONE), dim = 1)
            IND = IND
            OUT[i,IND[...,0],IND[...,1],IND[...,2]] = offset
        return OUT