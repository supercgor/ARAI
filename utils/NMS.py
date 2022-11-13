import numpy as np
import random


def nms(position, ele, info):
    ele2r = {'H': 0.528, 'O': 0.74}
    r = ele2r[ele]
    if len(position) == 0:
        return position
    position = np.asarray(sorted(position, key=lambda x: x[-1], reverse=True))
    mask = np.full(len(position), True)
    mask[1000:] = False  # 最多1000个原子，截断
    random.shuffle(mask)
    for i in range(len(position)):
        if mask[i]:
            for j in range(i + 1, len(position)):
                if mask[j]:
                    distance = np.sqrt(np.sum(np.square((position[i][:3] - position[j][:3]).dot(info['lattice']))))
                    if distance < r:
                        mask[j] = False
    position = position[mask]
    return position
