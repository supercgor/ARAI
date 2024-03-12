import numpy as np
from functools import partial

import torch
from torch import nn, Tensor

from ase import Atoms
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from scipy.spatial import cKDTree
from torchmetrics import Metric

from . import lib

from typing import Iterable

class MetricCollection(object):
    def __init__(self, *args: Metric, **kwargs: Metric):
        if len(args) > 0:
            self.keys = list(map(str, range(len(args))))
            self._metrics = dict(zip(self.keys, args))
        elif len(kwargs) > 0:
            self.keys = list(kwargs.keys())
            self._metrics = kwargs
        else:
            raise ValueError("At least one metric should be provided")
    
    def update(self, *args, **kwargs):
        if len(args) > 0:
            for k, v in zip(self.keys, args):
                if isinstance(v, (tuple, list)):
                    self._metrics[k].update(*v)
                else:
                    self._metrics[k].update(v)
        else:
            for k, v in kwargs.items():
                if isinstance(v, (tuple, list)):
                    self._metrics[k].update(*v)
                else:
                    self._metrics[k].update(v)
    
    def compute(self):
        return {k: v.compute() for k, v in self._metrics.items()}
    
    def reset(self):
        for v in self._metrics.values():
            v.reset()
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._metrics[self.keys[key]]
        else:
            return self._metrics[key]
            

# class Analyser(nn.Module):
#     """
#     Analyser module for the model prediction. This module is jit-compatible
#     """
#     def __init__(self, 
#                  real_size: tuple[float] = (3.0, 25.0, 25.0), 
#                  nms: bool = True, 
#                  cutoff: tuple[float] = (1.0, 0.75),
#                  sort: bool = True, 
#                  split: list[float] = [0.0, 3.0]):
#         super().__init__()
#         self.register_buffer("real_size", torch.as_tensor(real_size))
#         self._nms = nms
#         self.cutoff = cutoff
#         self._sort = sort
#         self.split = split
#         self.split[-1] += 1e-5
#         self.S = len(split) - 1
    
#     def forward(self, pred_clses: Tensor, pred_boxes: Tensor, targ_clses: Tensor, targ_boxes: Tensor) -> Tensor:
#         """
#         _summary_

#         Args:
#             pred_clses (Tensor): B C D H W
#             pred_boxes (Tensor): B 3 D H W
#             targ_clses (Tensor): B D H W 
#             targ_boxes (Tensor): B D H W 3

#         Returns: 
#             Tensor: B C-1 S (TP, FP, FN)
#         """
#         with torch.no_grad():
#             B, C, D, H, W = pred_clses.shape
#             pred_clses = pred_clses.permute(0, 2, 3, 4, 1)
#             pred_clses, pred_arges = pred_clses.max(dim = -1)
#             pred_boxes = pred_boxes.permute(0, 2, 3, 4, 1).sigmoid() # B D H W 3
#             confusion_matrix = torch.zeros((B, C - 1, self.S, 3), dtype = torch.long, device = pred_clses.device) # B C-1 (TP, FP, FN)
#             for b in range(B):
#                 pred_cls = pred_clses[b]
#                 pred_arg, pred_box, cls_mask = poscar.boxncls2pos_torch(pred_arges[b], pred_boxes[b])
#                 targ_cls, targ_box, _ = poscar.boxncls2pos_torch(targ_clses[b], targ_boxes[b])
#                 for c in range(1, C):
#                     mask = pred_arg == c
#                     pred_ions = pred_box[mask] * self.real_size
#                     if self._sort:
#                         mid = pred_cls[cls_mask[0], cls_mask[1], cls_mask[2]][mask]
#                         mid = mid.argsort(descending = True)
#                         pred_ions = pred_ions[mid]
#                     if self._nms:
#                         pred_ions = self.nms(pred_ions, self.cutoff[c-1])
#                     targ_ions = targ_box[targ_cls == c] * self.real_size
#                     confusion_matrix[b, c-1] = self.match(pred_ions, targ_ions, cutoff= self.cutoff[c-1], split= self.split)
#         return confusion_matrix

#     @torch.jit.export
#     def process_pred(self, pred_clses: Tensor, pred_boxes: Tensor, nms: bool | None = None, sort: bool | None = None) -> tuple[list[Tensor], list[list[int]]]:
#         _nms = self._nms if nms is None else nms
#         _sort = self._sort if sort is None else sort
#         with torch.no_grad():
#             B, C, D, H, W = pred_clses.shape
#             pred_clses = pred_clses.permute(0, 2, 3, 4, 1)
#             pred_clses, pred_arges = pred_clses.max(dim = -1)
#             pred_boxes = pred_boxes.permute(0, 2, 3, 4, 1).sigmoid() # B D H W 3
#             batch_pred_pos = []
#             batch_ion_num = []
#             for b in range(B):
#                 pred_pos = []
#                 ion_num = []
#                 pred_cls = pred_clses[b]
#                 pred_arg, pred_box, cls_mask = poscar.boxncls2pos_torch(pred_arges[b], pred_boxes[b])
#                 for c in range(1, C):
#                     mask = pred_arg == c
#                     pred_ions = pred_box[mask] * self.real_size
#                     if _sort:
#                         mid = pred_cls[cls_mask[0], cls_mask[1], cls_mask[2]][mask]
#                         mid = mid.argsort(descending = True)
#                         pred_ions = pred_ions[mid]
#                     if _nms:
#                         pred_ions = self.nms(pred_ions, self.cutoff[c-1])
#                     pred_pos.append(pred_ions)
#                     ion_num.append(pred_ions.shape[0])
#                 batch_pred_pos.append(torch.cat(pred_pos, dim = 0))
#                 batch_ion_num.append(ion_num)
#         return batch_pred_pos, batch_ion_num
        
    
#     @staticmethod
#     def nms(pos: Tensor, cutoff: float) -> Tensor:
#         """
#         _summary_

#         Args:
#             pos (Tensor): N 3

#         Returns:
#             Tensor: N 3
#         """
#         DIS = torch.cdist(pos, pos)
#         DIS = DIS < cutoff
#         DIS = (torch.triu(DIS, diagonal= 1)).float()
#         while True:
#             N = pos.shape[0]
#             restrain_tensor = DIS.sum(0)
#             restrain_tensor -= ((restrain_tensor != 0).float() @ DIS)
#             SELECT = restrain_tensor == 0
#             DIS = DIS[SELECT][:, SELECT]
#             pos = pos[SELECT]
#             if N == pos.shape[0]:
#                 break
#         return pos
        
#     @staticmethod
#     def match(pred: Tensor, targ: Tensor, cutoff: float, split: list[float]) -> Tensor:
#         """
#         _summary_

#         Args:
#             pred (Tensor): _description_
#             targ (Tensor): _description_
#             cutoff (float): _description_
#             split (tuple[float]): _description_

#         Returns:
#             Tuple[list[Tensor], list[Tensor], list[Tensor]]: _description_
#         """
#         OUTS = torch.empty((len(split) - 1, 3), dtype = torch.long, device = pred.device)
#         DIS = torch.cdist(pred, targ)
#         SELECT_T = (DIS  < cutoff).sum(0) != 0
#         for i in range(len(split) - 1):
#             LOW, HIGH = split[i], split[i+1]
#             layer_P, layer_T = (pred[:, 0] >= LOW) & (pred[:, 0] < HIGH), (targ[:, 0] >= LOW) & (targ[:, 0] < HIGH)
#             layer_TP = layer_T & SELECT_T
#             P, T, TP = layer_P.sum(), layer_T.sum(), layer_TP.sum()
#             FP, FN = P - TP, T - TP
#             OUTS[i,0], OUTS[i,1], OUTS[i,2] = TP, FP, FN
#         return OUTS

class parallelAnalyser(nn.Module):
    def __init__(self, real_size: tuple[float] = (25.0, 25.0, 4.0), nms: bool = True, sort: bool = True, threshold: float = 0.5, cutoff: float = 2.0, split: list[float] = [0.0, 4.0, 8.0]):
        super().__init__()
        self.register_buffer("_real_size", torch.as_tensor(real_size, dtype=torch.float))
        self._nms = nms
        self._sort = sort
        self._threshold = threshold
        self._cutoff = cutoff
        self._split = split
        self._split[-1] += 1e-5
    
    @torch.no_grad()
    def forward(self, preds: Tensor, targs: Tensor):
        # B D H W C -> B H W D C
        # preds = preds.permute(0, 2, 3, 1, 4)
        # targs = targs.permute(0, 2, 3, 1, 4)
        
        f = partial(self.eval_one, threshold = self._threshold, cutoff = self._cutoff, real_size = self._real_size, sort = self._sort, nms = self._nms, split = self._split)
        results = [f(pred, targ) for pred, targ in zip(preds.detach(), targs.detach())]
        return torch.stack(results, dim = 0)
             
    @staticmethod
    def eval_one(pred: Tensor, targ: Tensor, threshold: float, cutoff: float, real_size: Tensor, sort: bool, nms: bool, split: list[float]) -> tuple[Tensor]:
        _, pd_pos, pd_R = lib.box2orgvec(pred, lib.logit(threshold), cutoff, real_size, sort, nms)
        _, tg_pos, tg_R = lib.box2orgvec(targ, 0.5, cutoff, real_size, False, False)
        
        cm = torch.zeros((1, len(split) - 1, 4), dtype = torch.float, device = pred.device) # C S (TP, FP, FN)
            
        pd_match_ids, tg_match_ids = lib.argmatch(pd_pos, tg_pos, cutoff/2)
        
        pd_R = pd_R[pd_match_ids][:, (0, 1)] # N 3 3
        tg_R = tg_R[tg_match_ids][:, (0, 1)] # N 3 3
        
        match_tg_pos = tg_pos[tg_match_ids] # N 3
        
        ang = torch.einsum("bij,bij->bi", pd_R, tg_R) / pd_R.norm(dim=-1) / tg_R.norm(dim=-1)
        ang = torch.div(torch.arccos(ang.clamp(-1, 1)), torch.pi / 180.0)
        ang = torch.where(ang > 90, 180 - ang, ang)
        
        for i, (low, high) in enumerate(zip(split[:-1], split[1:])):
            match_tg_mask = (match_tg_pos[:, 2] >= low) & (match_tg_pos[:, 2] < high) # TP
            pd_mask = (pd_pos[:, 2] >= low) & (pd_pos[:, 2] < high) # P
            tg_mask = (tg_pos[:, 2] >= low) & (tg_pos[:, 2] < high) # T
            cm[0, i, 0] = match_tg_mask.sum() #TP
            cm[0, i, 1] = pd_mask.sum() - match_tg_mask.sum() # FP
            cm[0, i, 2] = tg_mask.sum() - match_tg_mask.sum() # FN
            cm[0, i, 3] = ang.mean().nan_to_num(90) # ANGxw
        return cm
                
class metStat():
    def __init__(self, value = None, reduction:str = "mean"):
        """To help you automatically find the mean or sum, which allows us to calculate something easily and consuming less space. 
        Args:
            reduction (str): should be 'mean', 'sum', 'max', 'min', 'none'
        """
        self._value = []
        if reduction == "mean":
            self._reduction = np.mean
        elif reduction == "sum":
            self._reduction = np.sum
        elif reduction == "max":
            self._reduction = np.max
        elif reduction == "min":
            self._reduction = np.min
        else:
            raise ValueError(f"reduction should be 'mean' or 'sum', but got {reduction}")

        if value is not None:
            self.add(value)
    
    def add(self, other):
        if isinstance(other, Iterable) or isinstance(other ,metStat):
            self.extend(other)
        else:
            self.append(other)
        
    def append(self, x: np.ndarray | torch.Tensor | float | int):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray) and x.size != 1:
            raise ValueError(f"Only support scalar input, but got {x}")
        self._value.append(x)
    
    def extend(self, xs: Iterable):
        if isinstance(xs, (metStat, list, tuple)):
            self._value.extend(xs)
        elif isinstance(xs, np.ndarray):
            xs = xs.view(-1)
            self._value.extend(xs)
        elif isinstance(xs, torch.Tensor):
            xs = xs.detach().cpu().view(-1).numpy()
            self._value.extend(xs)
        elif isinstance(xs, Iterable):
            self._value.extend(xs)
        else:
            raise TypeError(f"{type(xs)} is not an iterable")
    
    def reset(self):
        self._value = []
    
    def calc(self) -> float:
        return self._reduction(self._value)
    
    def __call__(self, other):
        self.add(other)
    
    def __repr__(self):
        return f"{self._reduction(self._value)}"
        
    def __str__(self):
        return str(self._reduction(self._value))
    
    def __len__(self):
        return len(self._value)
    
    def __format__(self, code):
        return self._reduction(self._value).__format__(code)

    @property
    def value(self):
        return self._value
    
class ConfusionCounter(object):
    def __init__(self,):
        self._cmc = []
    
    def add(self, cm: torch.Tensor) -> None:
        if cm.device != torch.device("cpu"):
            cm = cm.detach().cpu().numpy()
        else:
            cm = cm.numpy()
        if cm.ndim == 3:
            self._cmc.append(cm)
        elif cm.ndim >= 4:
            self._cmc.extend(cm)
        
    def calc(self):
        self._cmc = np.stack(self._cmc, axis = 0)
        return self
    
    def reset(self):
        self._cmc = []
    
    @property
    def TP(self):
        return self._cmc[..., 0].sum(axis = 0)
    
    @property
    def FP(self):
        return self._cmc[..., 1].sum(axis = 0)
    
    @property
    def FN(self):
        return self._cmc[..., 2].sum(axis = 0)
    
    @property
    def AR(self):
        T = self._cmc[..., 0] + self._cmc[..., 2]
        mask = (T == 0) & (self._cmc[..., 1] == 0)
        T.clip(1, out = T)
        T = np.where(mask, 1, self._cmc[..., 0] / T)
        return T.mean(axis = 0)
    
    @property
    def AP(self):
        P = self._cmc[..., 0] + self._cmc[..., 1]
        mask = (P == 0) & (self._cmc[..., 2] == 0)
        P.clip(1, out = P)
        P = np.where(mask, 1, self._cmc[..., 0] / P)
        return P.mean(axis = 0)
    
    @property
    def ACC(self):
        A = self._cmc[..., 0] + self._cmc[..., 1] + self._cmc[..., 2]
        mask = (A == 0)
        A.clip(1, out = A)
        A = np.where(mask, 1.0, self._cmc[..., 0] / A)
        return A.mean(axis = 0)
    
    @property
    def SUC(self):
        return ((self._cmc[..., 1] == 0) & (self._cmc[...,2] == 0)).astype(np.float_).mean(axis = 0)
    
class ConfusionRotate(ConfusionCounter):
    def __init__(self):
        super().__init__()
        
    @property
    def ROT(self):
        return self._cmc[..., 3].mean(axis = 0)
    
def ice_rule_counter(atoms: Atoms, scale_factor = 27.8) -> float:
    """
    Calculate the score of a structure based on ice rules:
    
    `S = \sum (\rho_i * |f(x) - f(Y)|_i), i \in [corner, face, edge, inner]`
    
    Args:
        atoms (Atoms): input structure

    Returns:
        score (float): the wasserstein distance, ~ < 0.04
    """
    
    stats_bonds = [[0.039, 0.401, 0.393, 0.149, 0.019, 0.   , 0.   ],
                   [0.008, 0.136, 0.427, 0.341, 0.087, 0.002, 0.   ],
                   [0.   , 0.025, 0.123, 0.529, 0.315, 0.007, 0.   ],
                   [0.   , 0.   , 0.002, 0.045, 0.931, 0.021, 0.001]]
    
    gr_ref = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 
              0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 
              0.000, 0.000, 0.135, 2.576, 6.314, 4.258, 1.486, 0.336, 0.078, 0.017, 0.003, 
              0.003, 0.009, 0.023, 0.056, 0.130, 0.295, 0.581, 1.059, 1.718, 2.299, 2.754, 
              2.522, 1.503, 0.677, 0.365, 0.282, 0.229, 0.179, 0.142, 0.092, 0.044]
    
    # Center the atoms
    atoms = atoms[atoms.symbols == 'O']
    atoms.center()
    # Get the cell and positions
    cell = atoms.cell.diagonal()
    pos = atoms.positions
    if np.all(cell == 0):
        cell_max = np.max(pos, axis = 0)
        cell_min = np.min(pos, axis = 0)
    else:
        cell_max = cell
        cell_min = np.zeros(3)
    
    up_atoms = atoms[atoms.positions[:, 2] > 12]
    down_atoms = atoms[atoms.positions[:, 2] < 12]
    distance = cdist(up_atoms.positions, down_atoms.positions, 'euclidean')
    distance = distance[distance < 3.2]
    # print(len(distance))
    connection = 1 - len(distance) / 30
    
    # split boarder atoms and inner atoms
    bond_min = 3.0
    bond_max = 3.2
    dist = cdist(pos, pos, 'euclidean') + np.eye(len(pos)) * 100
    
    # Split atoms
    inner_atoms_axis = np.logical_and(pos > cell_min + bond_min, pos < cell_max - bond_min)
    inner_atoms_axis = np.sum(inner_atoms_axis, axis = 1)
    
    # count the number of bonds
    bonds = []
    for i in range(4):
        fitted = inner_atoms_axis == i
        inner_dist = dist[fitted]
        # print((inner_dist < bond_max).sum(1))
        neighbors = np.sum(inner_dist < bond_max, axis = 1)
        num_bonds, _ = np.histogram(neighbors, bins = 7, range = (0, 7.0))
        bonds.append(num_bonds)
    
    score = []
    for i,j in zip(bonds, stats_bonds):
        rho = np.sum(i)
        if rho == 0:
            score.append(0.0)
        else:
            score.append(rho / len(pos) * wasserstein_distance(i / rho, j))
    score = np.sum(score) * scale_factor
    
    pos = down_atoms.positions - np.array([0, 0, 4])
    
    gr, _ = rdf(pos, dr=0.1, dims=cell_max - cell_min - [0, 0, 4])
    
    rmse = np.sqrt(((gr - gr_ref) ** 2).mean())
    
    return score + rmse * 0.5 + 0 * connection

# Adapted from https://github.com/by256/rdfpy
def rdf(particles, dr, rho=None, dims=None, rcutoff=0.9, eps=1e-15, progress=False):
    """
    Computes 2D or 3D radial distribution function g(r) of a set of particle 
    coordinates of shape (N, d). Particle must be placed in a 2D or 3D cuboidal 
    box of dimensions [width x height (x depth)].
    
    Parameters
    ----------
    particles : (N, d) np.array
        Set of particle from which to compute the radial distribution function 
        g(r). Must be of shape (N, 2) or (N, 3) for 2D and 3D coordinates 
        repsectively.
    dr : float
        Delta r. Determines the spacing between successive radii over which g(r)
        is computed.
    rho : float, optional
        Number density. If left as None, box dimensions will be inferred from 
        the particles and the number density will be calculated accordingly.
    rcutoff : float
        radii cutoff value between 0 and 1. The default value of 0.8 means the 
        independent variable (radius) over which the RDF is computed will range 
        from 0 to 0.8*r_max. This removes the noise that occurs at r values 
        close to r_max, due to fewer valid particles available to compute the 
        RDF from at these r values.
    eps : float, optional
        Epsilon value used to find particles less than or equal to a distance 
        in KDTree.
    parallel : bool, optional
        Option to enable or disable multiprocessing. Enabling affords 
        significant increases in speed.
    progress : bool, optional
        Set to False to disable progress readout.
        
    
    Returns
    -------
    g_r : (n_radii) np.array
        radial distribution function values g(r).
    radii : (n_radii) np.array
        radii over which g(r) is computed
    """

    if not isinstance(particles, np.ndarray):
        particles = np.array(particles)
    # assert particles array is correct shape
    shape_err_msg = 'particles should be an array of shape N x d, where N is \
                     the number of particles and d is the number of dimensions.'
    assert len(particles.shape) == 2, shape_err_msg
    # assert particle coords are 2 or 3 dimensional
    assert particles.shape[-1] in [2, 3], 'RDF can only be computed in 2 or 3 \
                                           dimensions.'
    
    mins = np.min(particles, axis=0)
    maxs = np.max(particles, axis=0)
    # translate particles such that the particle with min coords is at origin
    particles = particles - mins

    # dimensions of box
    if dims is None:
        dims = maxs - mins

    r_max = (np.min(dims) / 2)*rcutoff
    radii = np.arange(dr, r_max, dr)

    N, d = particles.shape
    if not rho:
        rho = N / np.prod(dims) # number density
    
    # create a KDTree for fast nearest-neighbor lookup of particles
    tree = cKDTree(particles)

    g_r = np.zeros(shape=(len(radii)))
    for r_idx, r in enumerate(radii):
        # find all particles that are at least r + dr away from the edges of the box
        valid_idxs = np.bitwise_and.reduce([(particles[:, i]-(r+dr) >= mins[i]) & (particles[:, i]+(r+dr) <= maxs[i]) for i in range(d)])
        valid_particles = particles[valid_idxs]
        
        # compute n_i(r) for valid particles.
        for particle in valid_particles:
            n = tree.query_ball_point(particle, r+dr-eps, return_length=True) - tree.query_ball_point(particle, r, return_length=True)
            g_r[r_idx] += n
        
        # normalize
        n_valid = len(valid_particles)
        shell_vol = (4/3)*np.pi*((r+dr)**3 - r**3) if d == 3 else np.pi*((r+dr)**2 - r**2)
        if n_valid != 0:
            g_r[r_idx] /= n_valid*shell_vol*rho

    return g_r, radii

def complex_relative_square_error(input: torch.Tensor, target: torch.Tensor, batched: bool = True, channel_first: bool = True) -> torch.Tensor:
    if channel_first:
        input = input.transpose(1, -1)
        target = target.transpose(1, -1)
    assert input.shape[-1] == 2, "The last dimension (real, complex) of input should be 2, but got {input.shape[-1]}"
    assert target.shape[-1] == 2, "The last dimension (real, complex) of target should be 2, but got {target.shape[-1]}"
    
    input = input.flatten(int(batched), -2)
    target = target.flatten(int(batched), -2)

    inp_norm = torch.norm(((input - target) ** 2).sum(-1), p = 2, dim = 1)
    tgt_norm = torch.norm((target - target.mean(-2, keepdim=True)).sum(-1), p = 2, dim = 1)
    
    return inp_norm / tgt_norm

def lfd(input, target):
    x_real, x_imag = input[:, 0], input[:, 1]
    y_real, y_imag = target[:, 0], target[:, 1]
    d_square = (x_real - y_real) ** 2 + (x_imag - y_imag) ** 2
    met = (d_square.mean(dim=[1,2,3]) + 1).log()
    return met