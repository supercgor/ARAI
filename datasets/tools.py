import random
import math
import os
import numpy as np
import torch
import cv2
from typing import Tuple


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
    @classmethod
    def get(cls, use_len: int = 10,
            out_len:int = 16,
            max_index: int = 20,
            split_border: None | Tuple[int] = None,
            split_ratio: Tuple[int] = (0.4, 0.3, 0.3),
            rand: bool = False):
        split = split_border or []
        split = [0, *split, max_index]
        ratio = split_ratio[:len(split) - 1]
        use_num = cls.weighted_split(use_len, ratio)
        out_num = cls.weighted_split(out_len, ratio)
        use_indices = []
        for lower, upper, use, out in zip(split[:-1], split[1:], use_num, out_num):
            try:
                indices = sorted(random.sample(range(lower, upper), k=min((upper - lower), use)))
                use_indices.extend(indices[i] for i in cls.gives_indices(min((upper - lower), use), out, rand=rand))
            except:
                print(lower, upper, use, out, split, ratio, out_num)
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
        assert upper > 0, f"Input is not valid: select_num = {select_num}, upper = {upper}"
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
            img = img[None, ...]
            last = i
        out = img.copy()
        IMG.append(out)
    IMG = np.stack(IMG) / 255
    IMG = torch.from_numpy(IMG).float()
    return IMG
