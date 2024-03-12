import random
import torch
import dgl
import numpy as np

def z_sampler(use: int, total: int, is_rand: bool = False) -> tuple[int]:
    if is_rand:
        sp = random.sample(range(total), k=(use % total))
        return [i for i in range(total) for _ in range(use // total + (i in sp))]
    else:
        return [i for i in range(total) for _ in range((use // total + ((use % total) > i)))]

def layerz_sampler(use, total, is_rand, layer = [0, 5, 12]):
    out = []
    while layer[-1] > total:
        layer.pop()
    num_layer = len(layer)
    layer = [*layer, total]
    for i, (low, high) in enumerate(zip(layer[:-1], layer[1:])):
        sam = z_sampler((use // num_layer + ((use % num_layer) > i)), high - low, is_rand)
        out.extend([j + low for j in sam])
    return out

def collate_fn(batch):
    batch = list(zip(*batch))
    out = []
    out.append(torch.cat(batch[0]))
    out += [None, None]
    out.append(dgl.batch(batch[3]))
    out.append(torch.stack(batch[4]))
    return out