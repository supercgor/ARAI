import numpy as np
import torch
from typing import Any

from . import const

def load(name:str) -> dict[str, Any]:
    with open(name, 'r') as f:
        comment = f.readline()
        scaling_factor = float(f.readline().strip())
        lattice = np.array([list(map(float, f.readline().strip().split(" "))) for _ in range(3)])
        ion: list[str] = f.readline().strip().split(" ")
        ion_num: list[int] = list(map(int, f.readline().strip().split(" ")))
        line = f.readline().strip()
        pos_mode = "direct"
        pos = []
        pos_dyn = []
        if line.startswith(("S","s")):
            selective_dynamics = True
            line = f.readline().strip()
        else:
            selective_dynamics = False
            
        if line.startswith(("D","d","C","c")):
            if line.startswith(("C", "c")):
                pos_mode = "cartesian"
            line = f.readline().strip()
            
        lines = [line,]
        for _ in range(sum(ion_num) - 1):
            lines.append(f.readline().strip())
        for line in lines:
            st = line.split(" ")
            pos.append(list(map(float, st[:3])))
            if selective_dynamics:
                pos_dyn.append(list(map(lambda x: x == "T", st[3:])))
        pos = np.array(pos)
        if selective_dynamics:
            pos_dyn = np.array(pos_dyn)
        else:
            pos_dyn = None
        
        ion, ion_num, pos = rearrange(ion, ion_num, pos)
    return {
        "comment": comment,
        "scaling_factor": scaling_factor,
        "lattice": lattice,
        "ion": ion,
        "ion_num": ion_num,
        "selective_dynamics": selective_dynamics,
        "pos_mode": pos_mode,
        "pos": pos,
        "pos_dyn": pos_dyn,
    }

def rearrange(ion: list[str], ion_num: list[int], pos: np.ndarray[np.float_], ion_order: str = ...,
              ) -> tuple[list[str], list[int], np.ndarray[np.float_]]:
    """
    Rearrage the order of ions.

    Args:
        ion (list[str]): _description_
        ion_num (list[int]): _description_
        pos (np.ndarray[np.float_]): _description_

    Returns:
        tuple[list[str], list[int], np.ndarray[np.float_]]: _description_
    """
    pos = np.split(pos, np.cumsum(ion_num)[:-1])
    if ion_order is ...:
        ion_order = const.ion_order
    ordions = ion_order.split()
    for i in ordions:
        if i in ion:
            ind = ion.index(i)
            ion.append(ion.pop(ind))
            ion_num.append(ion_num.pop(ind))
            pos.append(pos.pop(ind))
    return ion, ion_num, np.concatenate(pos)

def pos2boxncls(pos: np.ndarray[np.float_], ion_num: list[int], size: tuple[int] = (50, 50, 5), eps: float = 1E-6, ZXYformat: bool = True) -> np.ndarray[np.float_]:
    # pos: (n, 3), ion_num: (na, nb, nc, ...), size: XYZ
    pos = (pos * size).clip(0, np.asarray(size) - eps)
    ind = np.floor(pos).astype(int)
    offset = pos - ind
    ion_cls = np.asarray([[i] for i, num in enumerate(ion_num, 1) for _ in range(num)])
    box_cls, box_off = np.zeros((*size,1)), np.zeros((*size, 3))
    box_cls[tuple(ind.T)] = ion_cls
    box_off[tuple(ind.T)] = offset
    if ZXYformat:
        box_cls = np.transpose(box_cls, (2, 0, 1, 3))[..., 0]
        box_off = np.transpose(box_off, (2, 0, 1, 3))[..., (2, 0 ,1)]
    return box_cls, box_off

def boxncls2pos_np(box_cls: np.ndarray[np.float_], box_off: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    """
    _summary_

    Args:
        box_cls (np.ndarray[np.float_]): Z X Y
        box_off (np.ndarray[np.float_]): Z X Y 3

    Returns:
        np.ndarray[np.float_]: (N, N 3)
    """
    Z, X, Y = box_cls.shape
    tupmask = np.nonzero(box_cls)
    mask = np.asarray(tupmask).T
    box_cls = box_cls[tupmask]
    box_off = (box_off[tupmask] + mask) / (Z, X, Y)
    return box_cls, box_off

@torch.jit.script
def boxncls2pos_torch(box_cls: torch.Tensor, box_off: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    _summary_

    Args:
        box_cls (torch.tensor): Z X Y
        box_off (torch.tensor): Z X Y 3

    Returns:
        torch.tensor: (N, N 3)
    """
    ZXY = box_cls.shape
    mask = torch.nonzero(box_cls)
    tupmask = mask.T
    box_cls = box_cls[tupmask[0], tupmask[1], tupmask[2]]
    box_off = (box_off[tupmask[0], tupmask[1], tupmask[2]] + mask) / torch.as_tensor(ZXY, dtype = box_off.dtype, device = box_off.device)
    
    return box_cls, box_off, tupmask
    
def targ2pred(box_cls: torch.Tensor, box_off: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    box_cls = torch.nn.functional.one_hot(box_cls).permute(3, 0, 1, 2)
    box_off = box_off.permute(3, 0, 1, 2)
    box_off = torch.log(box_off / (1 - box_off))
    return box_cls, box_off

# def boxncls2pos(OFFSET: torch.Tensor, CLS: torch.Tensor, real_size = (3, 25, 25), order = ("O", "H"), nms = True, sort = True) -> dict[str, torch.Tensor]:
#     """
#     OFFSET: (3, Z, X, Y)
#     CLS: (Z, X, Y) or (n, Z, X, Y)
#     """
#     if CLS.dim() == 3: # n
#         conf, CLS =  torch.ones_like(CLS), CLS# (Z, X, Y)
#         sort = False
#     else:
#         conf, CLS = CLS.max(dim = 0).values, CLS.argmax(dim = 0) # (Z, X, Y)
#     Z, X, Y = CLS.shape

#     scale = torch.tensor([i/j for i,j in zip(real_size, (Z, X, Y))], device = CLS.device)
#     pos = OrderedDict()
#     for i, e in enumerate(order):
#         IND = CLS == i
#         POS = (OFFSET.permute(1,2,3,0)[IND] + IND.nonzero()) * scale
#         CONF = conf[IND]
        
#         if sort:
#             sort_ind = torch.argsort(CONF, descending=True)
#             POS = POS[sort_ind]
        
#         pos[e] = POS
#     if nms:
#         pos = cls.nms(pos)
#     return pos

# @classmethod
# def box2pos(cls, 
#             box, 
#             real_size = (3, 25, 25), 
#             order = ("O", "H"), 
#             threshold = 0.5, 
#             nms = True,
#             sort = True,
#             format = "DHWEC") -> dict[str, torch.Tensor]:
    
#     threshold = 1 / (1 + math.exp(-threshold))
    
#     newformat = "E D H W C"
#     format, newformat = " ".join(format), " ".join(newformat)
    
#     box = rearrange(box, f"{format} -> {newformat}", C = 4)
#     E, D, H, W, C = box.shape
#     scale = torch.tensor([i/j for i,j in zip(real_size, (D, H, W))], device= box.device)
#     pos = OrderedDict()
#     for i, e in enumerate(order):
#         pd_cls = box[i,...,0]
#         mask = pd_cls > threshold
#         pd_cls = pd_cls[mask]
#         pos[e] = torch.nonzero(mask) + box[i,...,1:][mask]
#         pos[e] = pos[e] * scale
#         if sort:
#             sort_ind = torch.argsort(pd_cls, descending=True)
#             pos[e] = pos[e][sort_ind]
#     if nms:
#         pos = cls.nms(pos)
#     return pos

# @classmethod
# def nms(cls, 
#         pd_pos: dict[str, torch.Tensor], 
#         radius: dict[str, float] = {"O":0.8, "H":0.6}, 
#         recusion = 1):
#     for e in pd_pos.keys():
#         pos = pd_pos[e]
#         if pos.nelement() == 0:
#             continue
#         if e == "O":
#             cutoff = radius[e] * 1.8
#         else:
#             cutoff = radius[e] * 1.8
#         DIS = torch.cdist(pos, pos)
#         DIS = DIS < cutoff
#         DIS = (torch.triu(DIS, diagonal= 1)).float()
#         restrain_tensor = DIS.sum(0)
#         restrain_tensor -= ((restrain_tensor != 0).float() @ DIS)
#         SELECT = restrain_tensor == 0
#         pd_pos[e] = pos[SELECT]
        
#     if recusion <= 1:
#         return pd_pos
#     else:
#         return cls.nms(pd_pos, radius, recusion - 1)

# @classmethod
# def plotAtom(cls, 
#                 bg: torch.Tensor | np.ndarray, 
#                 x: dict[str, torch.Tensor] | torch.Tensor, 
#                 order = ("none", "O", "H"), 
#                 color = {"O": (255, 0, 0), "H": (255, 255, 255)}, 
#                 radius = {"O": 0.7, "H": 0.4},
#                 scale = (1.0, 1.0, 1.0)
#             ) -> np.ndarray:
#     """bg should be HWC format, points_dict should be in CZXY format"""
#     if isinstance(bg, torch.Tensor):
#         bg = bg.cpu().numpy()
#     if isinstance(x, torch.Tensor):
#         points_dict = cls.czxy2pos(x, order)
#     else:
#         points_dict = x
#     bg = bg.copy()
#     zoom = (0, *bg.shape[:2]) / np.asarray(scale)
#     points_dict = {k: v * zoom for k, v in points_dict.items()}
#     for elem, pos in points_dict.items():
#         if isinstance(pos, torch.Tensor):
#             pos = pos.numpy()
#         pos = pos.astype(int)[...,:0:-1] # y x format
#         c = color[elem]
#         r = int(radius[elem] * zoom[1])
#         for p in pos:
#             bg = cv2.circle(bg, p, r, c, -1)
#             bg = cv2.circle(bg, p, r, (255, 255, 255), 1)
#     return bg

# @classmethod
# def pos2poscar(cls, 
#                 path,                    # the path to save poscar
#                 points_dict,             # the orderdict of the elems : {"O": N *　(Z,X,Y)}
#                 real_size = (3, 25, 25)  # the real size of the box
#                 ):
#     output = ""
#     output += f"{' '.join(points_dict.keys())}\n"
#     output += f"{1:3.1f}" + "\n"
#     output += f"\t{real_size[1]:.8f} {0:.8f} {0:.8f}\n"
#     output += f"\t{0:.8f} {real_size[2]:.8f} {0:.8f}\n"
#     output += f"\t{0:.8f} {0:.8f} {real_size[0]:.8f}\n"
#     output += f"\t{' '.join(points_dict.keys())}\n"
#     output += f"\t{' '.join(str(len(i)) for i in points_dict.values())}\n"
#     output += f"Selective dynamics\n"
#     output += f"Direct\n"
#     for ele in points_dict:
#         p = points_dict[ele]
#         for a in p:
#             output += f" {a[1]/real_size[1]:.8f} {a[2]/real_size[2]:.8f} {a[0]/real_size[0]:.8f} T T T\n"

#     with open(path, 'w') as f:
#         f.write(output)
        
#     return
