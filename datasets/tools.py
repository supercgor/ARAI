import random
import math
import os
import numpy as np
import torch
import cv2

def read_file(file_name):
    data = []
    with open(file_name) as fr:
        for line in fr:
            fn = line.strip().replace('\t', ' ')
            fn2 = fn.split(" ")
            if fn2[0] != '':
                data.append(fn2[0])
    return data

class indexGen():
    def __init__(self,
                 use_len: int | None = 10,
                 out_len: int = 16,
                 split_border: tuple = (),
                 split_ratio= ...,
                 rand=False,
                 ):

        self.split_border = split_border
        self.random = rand
        if use_len is None:
            use_len = out_len
        if split_ratio is ...:
            self.use_num = (use_len,)
            self.out_num = (out_len,)
        else:
            self.use_num = self.weighted_split(use_len, split_ratio)
            self.out_num = self.weighted_split(out_len, split_ratio)

    def get(self,max_index = 20):
        split = [0, *self.split_border, max_index]

        use_indices = []
        for i in range(len(split) - 1):
            lower = split[i]
            upper = split[i+1]
            use = self.use_num[i]
            out = self.out_num[i]
            indices = sorted(random.sample(
                range(lower, upper), k=min((upper - lower), use)))
            use_indices.extend(indices[i] for i in self.gives_indices(
                min((upper - lower), use), out, rand=self.random))
        return use_indices

    @staticmethod
    def weighted_split(nums, ratio, mode=math.ceil):
        if len(ratio) <= 1:
            return nums,
        else:
            out = mode(nums * ratio[0]/sum(ratio))
            return out, *indexGen.weighted_split(nums - out, ratio=ratio[1:], mode=mode)

    @staticmethod
    def gives_indices(upper, select_num, offset=0, rand=False):
        if rand:
            sp = random.sample(range(upper), k=(select_num % upper))
            return [i + offset for i in range(upper) for _ in range(select_num // upper + (i in sp))]
        else:
            return [i + offset for i in range(upper) for _ in range((select_num // upper + ((select_num % upper) > i)))]
        
def read_pic(path: str,
             indices: list,
             img_size: tuple = (128, 128)
             ) -> torch.Tensor:

    IMG = []
    last = None

    for i in indices:
        filename = f'{i}.png'
        img_path = os.path.join(path, filename)
        if i == last:
            pass
        elif os.path.exists(img_path):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = np.flip(img.T, axis=1)
            if img.shape != img_size and img_size is not None:
                img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
            img = img[None,...]
            last = i
        out = img.copy()
        IMG.append(out)
    IMG = np.stack(IMG) / 255
    IMG = torch.from_numpy(IMG).float()
    return IMG