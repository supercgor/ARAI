import numpy as np
import torch
from typing import Any, overload

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

def rearrange(ion, ion_num, pos, ion_order = ..., pos_dyn = None):
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
    if pos_dyn is not None:
        pos_dyn = np.split(pos_dyn, np.cumsum(ion_num)[:-1])
    if ion_order is ...:
        ion_order = const.ion_order
    ordions = ion_order.split()
    for i in ordions:
        if i in ion:
            ind = ion.index(i)
            ion.append(ion.pop(ind))
            ion_num.append(ion_num.pop(ind))
            pos.append(pos.pop(ind))
            if pos_dyn is not None:
                pos_dyn.append(pos_dyn.pop(ind))
    if pos_dyn is not None:
        return ion, ion_num, np.concatenate(pos), np.concatenate(pos_dyn)
    else:
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
        box_cls (torch.tensor): Z X Y | B Z X Y
        box_off (torch.tensor): Z X Y 3 | B Z X Y 3 

    Returns:
        tuple[torch.tensor]: (N, ), (N, 3), (3, Z * X * Y) or list[(N,)], list[(N, 3)], list[(3, Z * X * Y)]
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

def save(path: str, 
         lattice: np.ndarray | list | tuple, 
         ion: list[str], 
         ion_num: list[int], 
         pos: np.ndarray, 
         pos_mode: str = "Direct", 
         selective_dynamics: bool = True,
         pos_dyn: np.ndarray | None = None, 
         scaling_factor: float = 1.0, 
         comment: str = "Generated by 'poscar.save'",
         ZXYformat: bool = True,):
    if ".poscar" not in path[-7:]:
        path += ".poscar"
    if selective_dynamics:
        if pos_dyn is None and selective_dynamics:
            pos_dyn = np.ones_like(pos, dtype = bool)
        assert isinstance(pos_dyn, np.ndarray), "pos_dyn should be a numpy array"
    if isinstance(lattice, (tuple, list)) or len(lattice.shape) == 1:
        lattice = np.diag(lattice)
    if pos_dyn is not None:
        ion, ion_num, pos, pos_dyn = rearrange(ion, ion_num, pos, pos_dyn = pos_dyn)
    else:
        ion, ion_num, pos = rearrange(ion, ion_num, pos)
    if ZXYformat:
        pos = pos[..., (1, 2, 0)]
        if pos_dyn is not None:
            pos_dyn = pos_dyn[..., (1, 2, 0)]
        lattice = lattice[(1, 2, 0), :][..., (1, 2, 0)]
    out = f"{comment}\n{scaling_factor}\n"
    for row in lattice:
        out += f"\t{row[0]:11.8f} {row[1]:11.8f} {row[2]:11.8f}\n"
    out += f"\t{' '.join(ion)}\n"
    out += f"\t{' '.join(map(lambda x: str(int(x)), ion_num))}\n"
    if selective_dynamics:
        out += f"Selective dynamics\n"
    if pos_mode.lower().startswith("d"):
        out += f"Direct\n"
    elif pos_mode.lower().startswith("c"):
        out += f"Cartesian\n"
    else:
        raise ValueError(f"pos_mode should be 'Direct' or 'Cartesian', not {pos_mode}")
    if selective_dynamics:
        for p, dy in zip(pos, pos_dyn):
            out += f" {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {'T' if dy[0] else 'F'} {'T' if dy[1] else 'F'} {'T' if dy[2] else 'F'}\n"
    else:
        for p in pos:
            out += f" {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n"
    with open(path, 'w') as f:
        f.write(out)
    return