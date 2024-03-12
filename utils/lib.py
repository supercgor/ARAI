from ase import Atoms
import numpy as np
from typing import overload
from numpy import ndarray
from torch import Tensor
import numba as nb
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
import torch
from torchvision.utils import make_grid

@nb.jit(nopython=True)
def getWaterRotate(tag: ndarray, inv_ref: ndarray) -> ndarray:
    O, H1, H2 = tag
    H1, H2 = H1 - O, H2 - O
    H1 = H1
    H2 = H2
    H3 = np.cross(H1, H2)
    if H3[2] < 0:
        tag[0] = H2
        tag[1] = H1
        tag[2] = -H3
    else:
        tag[0] = H1
        tag[1] = H2
        tag[2] = H3
    return inv_ref @ tag

@overload
def box2vec(box_cls: ndarray, box_off: ndarray, *args: ndarray, threshold: float = 0.5) -> tuple[ndarray, ...]: ...
@overload
def box2vec(box_cls: Tensor, box_off: Tensor, *args: Tensor, threshold: float = 0.5) -> tuple[Tensor, ...]: ...

def box2vec(box_cls, box_off, *args, threshold = 0.5):
    """
    _summary_

    Args:
        box_cls (Tensor): X Y Z
        box_off (Tensor): X Y Z (ox, oy, oz)
        args (tuple[ Tensor]): X Y Z *
    Returns:
        tuple[Tensor]: (N, ), (N, 3), N
    """
    if isinstance(box_cls, Tensor):
        return _box2vec_th(box_cls, box_off, *args, threshold = threshold)
    elif isinstance(box_cls, np.ndarray):
        return _box2vec_np(box_cls, box_off, *args, threshold = threshold)

def _box2vec_np(box_cls, box_off, *args, threshold = 0.5):
    box_size = box_cls.shape
    mask = np.nonzero(box_cls > threshold)
    box_cls = box_cls[mask]
    box_off = box_off[mask] + np.stack(mask, axis = -1)
    box_off = box_off / box_size
    args = [arg[mask] for arg in args]
    return box_cls, box_off, *args

def _box2vec_th(box_cls, box_off, *args, threshold = 0.5):
    box_size = box_cls.shape
    mask = torch.nonzero(box_cls > threshold, as_tuple=True)
    box_cls = box_cls[mask]
    box_off = box_off[mask] + torch.stack(mask, dim = -1)
    box_off = box_off / torch.as_tensor(box_size, dtype=box_cls.dtype, device=box_cls.device)
    args = [arg[mask] for arg in args]
    return box_cls, box_off, *args

@overload
def masknms(pos: ndarray, cutoff: float) -> ndarray: ...
@overload
def masknms(pos: Tensor, cutoff: float) -> Tensor: ...

def masknms(pos, cutoff):
    """
    _summary_

    Args:
        pos (Tensor): N 3

    Returns:
        Tensor: N 3
    """
    if isinstance(pos, Tensor):
        return _masknms_th(pos, cutoff)
    else:
        return _masknms_np(pos, cutoff)

def _masknms_np(pos: ndarray, cutoff: float) -> ndarray:
    dis = cdist(pos, pos) < cutoff
    dis = np.triu(dis, 1).astype(float)
    args = np.ones(pos.shape[0], dtype = bool)
    while True:
        N = pos.shape[0]
        restrain_tensor = dis.sum(0)
        restrain_tensor -= (restrain_tensor != 0).astype(float) @ dis
        SELECT = restrain_tensor == 0
        dis = dis[SELECT][:, SELECT]
        pos = pos[SELECT]
        args[args.nonzero()] = SELECT
        if N == pos.shape[0]:
            break
    return args

def _masknms_th(pos: Tensor, cutoff: float) -> Tensor:
    dis = torch.cdist(pos, pos) < cutoff
    dis = torch.triu(dis, 1).float()
    args = torch.ones(pos.shape[0], dtype = torch.bool, device = pos.device)
    while True:
        N = pos.shape[0]
        restrain_tensor = dis.sum(0)
        restrain_tensor -= ((restrain_tensor != 0).float() @ dis)
        SELECT = restrain_tensor == 0
        dis = dis[SELECT][:, SELECT]
        pos = pos[SELECT]
        args[args.nonzero(as_tuple=True)] = SELECT
        if N == pos.shape[0]:
            break
    return args

@overload
def argmatch(pred: ndarray, targ: ndarray, cutoff: float) -> tuple[ndarray, ...]: ...
@overload
def argmatch(pred: Tensor, targ: Tensor, cutoff: float) -> tuple[Tensor, ...]: ...

def argmatch(pred, targ, cutoff):
    # This function is only true when one prediction does not match two targets and one target can match more than two predictions
    # return pred_ind, targ_ind
    if isinstance(pred, Tensor):
        return _argmatch_th(pred, targ, cutoff)
    else:
        return _argmatch_np(pred, targ, cutoff)

def _argmatch_np(pred: ndarray, targ: ndarray, cutoff: float) -> tuple[ndarray, ...]:
    dis = cdist(targ, pred)
    dis = np.stack((dis < cutoff).nonzero(), axis = -1)
    dis = dis[:, (1, 0)]
    _, idx, counts = np.unique(dis[:,1], return_inverse=True, return_counts=True)
    idx = np.argsort(idx)
    counts = counts.cumsum()
    if counts.shape[0] != 0:
        counts = np.concatenate(([0], counts[:-1]))
    idx = idx[counts]
    dis = dis[idx]
    return dis[:,0], dis[:,1]

def _argmatch_th(pred: Tensor, targ: Tensor, cutoff: float) -> tuple[Tensor, ...]:
    dis = torch.cdist(targ, pred)
    dis = (dis < cutoff).nonzero()
    dis = dis[:, (1, 0)]
    _, idx, counts = torch.unique(dis[:,1], sorted=True, return_inverse=True, return_counts=True)
    idx = torch.argsort(idx, stable=True)
    counts = counts.cumsum(0)
    if counts.shape[0] != 0:
        counts = torch.cat([torch.as_tensor([0], dtype=counts.dtype, device=counts.device), counts[:-1]])
    idx = idx[counts]
    dis = dis[idx]
    return dis[:,0], dis[:,1]

def group_as_water(O_position, H_position):
    """
    Group the oxygen and hydrogen to water molecule

    Args:
        pos_o (ndarray | Tensor): shape (N, 3)
        pos_h (ndarray | Tensor): shape (2N, 3)

    Returns:
        ndarray | Tensor: (N, 9)
    """
    if isinstance(O_position, Tensor):
        dis = torch.cdist(O_position, H_position)
        dis = torch.topk(dis, 2, dim = 1, largest = False).indices
        return torch.cat([O_position,H_position[dis].view(-1, 6)], dim = -1)
    else:
        dis = cdist(O_position, H_position)
        dis = np.argpartition(dis, 1, axis = 1)[:, :2]
        return np.concatenate([O_position, H_position[dis].reshape(-1, 6)], axis = -1)

@overload
def box2orgvec(box: ndarray, threshold: float, cutoff: float, real_size, sort: bool, nms: bool) -> tuple[ndarray, ...]: ...
@overload
def box2orgvec(box: Tensor, threshold: float, cutoff: float, real_size, sort: bool, nms: bool) -> tuple[Tensor, ...]: ...

def box2orgvec(box, threshold = 0.5, cutoff = 2.0, real_size = (25, 25, 12), sort = True, nms = True):
    """
    Convert the prediction/target to the original vector, including the confidence sequence, position sequence, and rotation matrix sequence

    Args:
        box (Tensor): X Y Z 10
        threshold (float): confidence threshold
        cutoff (float): nms cutoff distance
        real_size (Tensor): real size of the box
        sort (bool): to sort the box by confidence
        nms (bool): to nms the box

    Returns:
        tuple[Tensor]: `conf (N,)`, `pos (N, 3)`, `R (N, 3, 3)`
    """
    if isinstance(box, Tensor):
        return _box2orgvec_th(box, threshold, cutoff, real_size, sort, nms)
    elif isinstance(box, np.ndarray):
        return _box2orgvec_np(box, threshold, cutoff, real_size, sort, nms)

def _box2orgvec_np(box, threshold, cutoff, real_size, sort, nms) -> tuple[ndarray, ...]:
    if box.shape[-1] == 4:
        pd_conf, pd_pos = box2vec(box[..., 0:1], box[..., 1:4], threshold = threshold)
        pd_pos = pd_pos * real_size
        if sort:
            pd_conf_order = pd_conf.argsort()[::-1]
            pd_pos = pd_pos[pd_conf_order]
        if nms:
            pd_nms_mask = masknms(pd_pos, cutoff)
            pd_conf = pd_conf[pd_nms_mask]
            pd_pos = pd_pos[pd_nms_mask]
        return pd_conf, pd_pos
    
    elif box.shape[-1] == 10:
        pd_conf, pd_pos, pd_rotx, pd_roty = box2vec(box[..., 0], box[..., 1:4], box[..., 4:7], box[..., 7:10], threshold = threshold)
        pd_rotz = np.cross(pd_rotx, pd_roty)
        pd_R = np.stack([pd_rotx, pd_roty, pd_rotz], axis = -2)
        if sort:
            pd_conf_order = pd_conf.argsort()[::-1]
            pd_pos = pd_pos[pd_conf_order]
            pd_R = pd_R[pd_conf_order]
        if nms:
            pd_nms_mask = masknms(pd_pos, cutoff)
            pd_conf = pd_conf[pd_nms_mask]
            pd_pos = pd_pos[pd_nms_mask]
            pd_R = pd_R[pd_nms_mask]
        return pd_conf, pd_pos, pd_R
    
    else:
        raise ValueError(f"Require the last dimension of the box to be 4 or 10, but got {box.shape[-1]}")
    
def _box2orgvec_th(box, threshold, cutoff, real_size, sort, nms):
    if box.shape[-1] == 4:
        pd_conf, pd_pos = box2vec(box[...,0], box[...,1:4], threshold = threshold)
        pd_pos = pd_pos * torch.as_tensor(real_size, dtype=pd_pos.dtype, device=pd_pos.device)
        if sort:
            pd_conf_order = pd_conf.argsort(descending = True)
            pd_pos = pd_pos[pd_conf_order]
        
        if nms:
            pd_nms_mask = masknms(pd_pos, cutoff)
            pd_conf = pd_conf[pd_nms_mask]
            pd_pos = pd_pos[pd_nms_mask]
            
        return pd_conf, pd_pos
        
    elif box.shape[-1] == 10:
        pd_conf, pd_pos, pd_rotx, pd_roty = box2vec(box[...,0], box[...,1:4], box[...,4:7], box[...,7:10], threshold = threshold)
        pd_pos = pd_pos * torch.as_tensor(real_size, dtype=pd_pos.dtype, device=pd_pos.device)
        pd_rotz = torch.cross(pd_rotx, pd_roty, dim = -1)
        pd_R = torch.stack([pd_rotx, pd_roty, pd_rotz], dim = -2)
    
        if sort:
            pd_conf_order = pd_conf.argsort(descending = True)
            pd_pos = pd_pos[pd_conf_order]
            pd_R = pd_R[pd_conf_order]
        
        if nms:
            pd_nms_mask = masknms(pd_pos, cutoff)
            pd_conf = pd_conf[pd_nms_mask]
            pd_pos = pd_pos[pd_nms_mask]
            pd_R = pd_R[pd_nms_mask]
        
        return pd_conf, pd_pos, pd_R
    
    else:
        raise ValueError(f"Require the last dimension of the box to be 4 or 10, but got {box.shape[-1]}")

@overload
def vec2box(unit_pos: ndarray, vec: ndarray, box_size: tuple[int, int, int]) -> ndarray: ...
@overload
def vec2box(unit_pos: Tensor, vec: Tensor, box_size: tuple[int, int, int]) -> Tensor: ...

def vec2box(unit_pos, vec, box_size):
    if isinstance(unit_pos, Tensor):
        box = torch.zeros((*box_size, 4 + vec.shape[1]), dtype = unit_pos.dtype, device = unit_pos.device)
        box_size = torch.as_tensor(box_size, dtype = unit_pos.dtype, device = unit_pos.device)
        pd_ind = torch.floor(unit_pos.clamp(0, 1-1E-7) * box_size[None]).long()
        
        all_same = ((pd_ind[None] - pd_ind[:, None]) == 0).all(dim=-1)
        all_same.fill_diagonal_(False)
        if all_same.any():
            print("(Warning) There are same positions in the unit_pos")
            print(pd_ind[all_same.nonzero(as_tuple=True)[0]])
            
        pd_off = unit_pos * box_size - pd_ind
        feature = torch.cat([torch.ones(unit_pos.shape[0], 1, dtype=torch.float, device=unit_pos.device), pd_off, vec], dim = -1)
        box[pd_ind[:,0], pd_ind[:,1], pd_ind[:,2]] = feature
    else:
        box = np.zeros((*box_size, 4 + vec.shape[1]))
        pd_ind = np.floor(unit_pos * box_size).astype(int)
        pd_off = unit_pos * box_size - pd_ind
        feature = np.concatenate([np.ones((unit_pos.shape[0], 1)), pd_off, vec], axis = -1)
        box[pd_ind[:,0], pd_ind[:,1], pd_ind[:,2]] = feature
    return box

def box2atom(box, cell = [25.0, 25.0, 16.0], threshold = 0.5, cutoff= 2.0) -> list[Atoms] | Atoms:
    if box.dim() > 4:
        return list(box2atom(b, cell, threshold, cutoff) for b in box)
    else:
        confidence, positions = box2orgvec(box, threshold, cutoff, cell, sort=True, nms=True)
        if isinstance(positions, Tensor):
            confidence = confidence.detach().cpu().numpy()
            positions = positions.detach().cpu().numpy()
        atoms = Atoms("O" * positions.shape[0], positions, tags=confidence, cell=cell, pbc=False)
    return atoms
    
def makewater(pos: ndarray, rot: ndarray):
    # N 3, N 3 3 -> N 3 3
    if not isinstance(pos, ndarray):
        pos = pos.detach().cpu().numpy()
    if not isinstance(rot, ndarray):
        rot = rot.detach().cpu().numpy()
        
    water = np.array([
        [ 0.         , 0.         , 0.        ],
        [ 0.         , 0.         , 0.9572    ],
        [ 0.9266272  , 0.         ,-0.23998721],
    ])
    
    # print( np.einsum("ij,Njk->Nik", water, rot) )
    return np.einsum("ij,Njk->Nik", water, rot) + pos[:, None, :]

@torch.jit.script
def __encode_th(positions):
    positions = positions.reshape(-1, 9)
    o, u, v = positions[...,0:3], positions[...,3:6], positions[...,6:]
    u, v = u + v - 2 * o, u - v
    u = u / torch.norm(u, dim=-1, keepdim=True)
    v = v / torch.norm(v, dim=-1, keepdim=True)
    v = torch.where(v[..., 1].unsqueeze(-1) >= 0, v, -v)
    v = torch.where(v[..., 0].unsqueeze(-1) >= 0, v, -v)
    return torch.cat([o, u, v], dim=-1)
    
#@nb.njit(fastmath=True, cache=True)
def __encode_np(positions:ndarray):
    positions = positions.reshape(-1, 9)
    o, u, v = positions[...,0:3], positions[...,3:6], positions[...,6:]
    u, v = u + v - 2 * o, u - v
    u = u / np.expand_dims(((u**2).sum(axis=-1)**0.5), -1)
    v = v / np.expand_dims(((v**2).sum(axis=-1)**0.5), -1)
    v = np.where(v[..., 1][...,None] >= 0, v, -v)
    v = np.where(v[..., 0][...,None] >= 0, v, -v)
    return np.concatenate((o, u, v), axis=-1)

@torch.jit.script
def __decode_th(emb):
    o, u, v = emb[...,0:3], emb[...,3:6], emb[...,6:]
    h1 = (0.612562225 * u + 0.790422368 * v) * 0.9584 + o
    h2 = (0.612562225 * u - 0.790422368 * v) * 0.9584 + o
    return torch.cat([o, h1, h2], dim=-1)

@nb.njit(fastmath=True,parallel=True, cache=True)
def __decode_np(emb):
    o, u, v = emb[...,0:3], emb[...,3:6], emb[...,6:]
    h1 = (0.612562225 * u + 0.790422368 * v) * 0.9584 + o
    h2 = (0.612562225 * u - 0.790422368 * v) * 0.9584 + o
    return np.concatenate((o, h1, h2), axis=-1)
    
def encodewater(positions):
    if isinstance(positions, Tensor):
        return __encode_th(positions)
    else:
        return __encode_np(positions)

def decodewater(emb):
    if isinstance(emb, Tensor):
        return __decode_th(emb)
    else:
        return __decode_np(emb)
    

def rotate(points, rotation_vector: ndarray):
    """
    Rotate the points with rotation_vector.

    Args:
        points (_type_): shape (..., 3)
        rotation_vector (_type_): rotation along x, y, z axis. shape (3,)
    """
    if points.shape[-1] == 2:
        rotation_vector = np.array([0, 0, rotation_vector])
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
    
    if isinstance(points, Tensor):
        rotation_matrix = torch.as_tensor(rotation_matrix)
        
    if points.shape[-1] == 3:
        return points @ rotation_matrix.T
    
    elif points.shape[-1] == 2:
        return points @ rotation_matrix.T[:2, :2]
    
    else:
        raise NotImplementedError("Only 2D and 3D rotation is implemented.")

def logit(x, eps = 1E-7):
    if isinstance(x, (float, np.ndarray)):
        return - np.log(1 / (x + eps) - 1)
    else:
        return torch.logit(x, eps)


def replicate(points: ndarray, times: list[int], offset: ndarray) -> ndarray:
    """
    Replicate the points with times and offset.

    Args:
        points (ndarray): shape (N, 3)
        times (list[int]): [x times, y times, z times]
        offset (ndarray): shape (3, 3) 3 vectors

    Returns:
        _type_: _description_
    """
    if len(offset.shape) == 1:
        offset = np.diag(offset)
        
    for i, (t, o) in enumerate(zip(times, offset)):
        if t == 1:
            continue
        buf = []
        low = -(t//2)
        for j in range(low, low+t):
            res = points + j * o
            buf.append(res)
        points = np.concatenate(buf, axis=0)
    return points

def grid_to_water_molecule(grids, cell = [25.0, 25.0, 16.0], threshold = 0.5, cutoff= 2.0, flip_axis=[False, False, True]) -> Atoms:
    """
    Convert grids to atoms formats.

    Args:
        grids (Tensor | ndarray): shape: (X, Y, Z, C)

    Returns:
        atoms (ase.Atoms)
    """
    conf, pos, rotation = box2orgvec(grids, threshold=threshold, cutoff=cutoff, real_size=cell, sort=True, nms=True)
    rotation = rotation.view(-1, 9)[:, :6]
    
    atoms_pos = decodewater(np.concatenate([pos, rotation], axis = -1)).reshape(-1,3)
    atoms_pos = atoms_pos * np.where(flip_axis, -1, 1) + np.where(flip_axis, cell, 0)
    
    atoms_types = ["O","H","H"] * len(pos)
    atoms_conf = np.repeat(conf, 3)
    atoms = Atoms(atoms_types, atoms_pos, tags=atoms_conf, cell=cell, pbc=False)
    
    return atoms

class view(object):
    @staticmethod
    def space_to_image(tensor: Tensor):
        "C H W D -> (C D) 1 H W"
        tensor = tensor.permute(0, 3, 1, 2).flatten(0, 1).unsqueeze(1)
        image = make_grid(tensor, nrow = int(np.sqrt(tensor.shape[0])))
        return image
    