import numpy as np
import numba as nb
import torch
@nb.jit(nopython=True)
def getWaterRotate(tag: np.ndarray, inv_ref: np.ndarray) -> np.ndarray:
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

@nb.njit(fastmath=True,parallel=True, cache=True)
def cdist2(A,B):
    assert A.shape[1]==B.shape[1]
    C=np.empty((A.shape[0],B.shape[0]),A.dtype)
    
    init_val_arr=np.zeros(1,A.dtype)
    init_val=init_val_arr[0]
    
    for i in nb.prange(A.shape[0]):
        for j in range(B.shape[0]):
            acc=init_val
            for k in range(A.shape[1]):
                acc+=(A[i,k]-B[j,k])**2
            C[i,j]=np.sqrt(acc)
    return C

def box2vec(box_cls: torch.Tensor, box_off: torch.Tensor, *args, threshold: float = 0.5) -> tuple[torch.Tensor]:
    """
    _summary_

    Args:
        box_cls (torch.tensor): X Y Z
        box_off (torch.tensor): X Y Z (ox, oy, oz)
        args (tuple[ torch.tensor]): X Y Z *
    Returns:
        tuple[torch.tensor]: (N, ), (N, 3), N
    """
    box_size = box_cls.shape
    mask = torch.nonzero(box_cls > threshold)
    tupmask = mask.T
    box_cls = box_cls[tupmask[0], tupmask[1], tupmask[2]]
    box_off = (box_off[tupmask[0], tupmask[1], tupmask[2]] + mask) / torch.as_tensor(box_size, dtype = box_off.dtype, device = box_off.device)
    args = [arg[tupmask[0], tupmask[1], tupmask[2]] for arg in args]
    return box_cls, box_off, *args

def masknms(pos: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    _summary_

    Args:
        pos (Tensor): N 3

    Returns:
        Tensor: N 3
    """
    DIS = torch.cdist(pos, pos)
    DIS = DIS < cutoff
    DIS = (torch.triu(DIS, diagonal= 1)).float()
    args = torch.ones(pos.shape[0], dtype = torch.bool, device = pos.device)
    while True:
        N = pos.shape[0]
        restrain_tensor = DIS.sum(0)
        restrain_tensor -= ((restrain_tensor != 0).float() @ DIS)
        SELECT = restrain_tensor == 0
        DIS = DIS[SELECT][:, SELECT]
        pos = pos[SELECT]
        args[args.nonzero(as_tuple=True)] = SELECT
        if N == pos.shape[0]:
            break
    return args

def argmatch(pred: torch.Tensor, targ: torch.Tensor, cutoff: float) -> tuple[torch.Tensor]:
        # This function is only true when one prediction does not match two targets and one target can match more than two predictions
        # return pred_ind, targ_ind
        dis = torch.cdist(targ, pred)
        dis = (dis < cutoff).nonzero()
        dis = dis[:, (1, 0)]
        _, idx, counts = torch.unique(dis[...,1], sorted=True, return_inverse=True, return_counts=True)
        idx = torch.argsort(idx, stable=True)
        counts = counts.cumsum(0)
        if counts.shape[0] != 0:
            counts = torch.cat((torch.tensor([0], device = pred.device), counts[:-1]))
        idx = idx[counts]
        dis = dis[idx]
        
        return dis[...,0], dis[...,1]

def box2orgvec(box: torch.Tensor, threshold: float, cutoff: float, real_size: torch.Tensor, sort: bool, nms: bool) -> tuple[torch.Tensor]:
    """
    Convert the prediction/target to the original vector, including the confidence sequence, position sequence, and rotation matrix sequence

    Args:
        box (torch.Tensor): X Y Z 10
        threshold (float): confidence threshold
        cutoff (float): nms cutoff distance
        real_size (torch.Tensor): real size of the box
        sort (bool): to sort the box by confidence
        nms (bool): to nms the box

    Returns:
        tuple[torch.Tensor]: `conf (N,)`, `pos (N, 3)`, `R (N, 3, 3)`
    """
    with torch.no_grad():
        pd_conf, pd_pos, pd_rotx, pd_roty = torch.split(box, [1, 3, 3, 3], dim = -1)
        pd_conf.squeeze_(-1)
        pd_conf, pd_pos, pd_rotx, pd_roty = box2vec(pd_conf, pd_pos, pd_rotx, pd_roty, threshold = threshold)
        
        pd_rotz = torch.cross(pd_rotx, pd_roty, dim = -1)
        pd_R = torch.stack([pd_rotx, pd_roty, pd_rotz], dim = -2)
        
        if not isinstance(real_size, torch.Tensor):
            real_size = torch.as_tensor(real_size, dtype = pd_pos.dtype, device = pd_pos.device)
        pd_pos = pd_pos * real_size
        
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

def orgvec2box(pd_pos, pd_R, box_size) -> torch.Tensor:
    #TODO
    box = torch.zeros((*box_size, 10), dtype = pd_pos.dtype, device = pd_pos.device)
    box_size = torch.as_tensor(box_size, dtype = pd_pos.dtype, device = pd_pos.device)
    pd_ind = torch.floor(pd_pos * box_size).long()
    pd_off = pd_pos * box_size - pd_ind
    pd_R = pd_R.view(-1, 9)[:, :6]
    feature = torch.cat([torch.ones(pd_pos.shape[0], 1, dtype=torch.float, device=pd_pos.device), pd_off, pd_R], dim = -1)
    box[pd_ind[:,0], pd_ind[:,1], pd_ind[:,2]] = feature
    return box

def box2box(box, real_size = (25.0, 25.0, 4.0), threshold = 0.5, nms= True, sort=True, cutoff=0.5):
    xyz = torch.as_tensor(box.shape[:3], dtype = box.dtype, device = box.device)
    _, pos, rot = box2orgvec(box, threshold, cutoff, real_size, sort = sort, nms = nms)
    pos /= xyz
    box = orgvec2box(pos, rot, box.shape[:3])
    return box

def inverse_sigmoid(x: torch.Tensor | float) -> torch.Tensor | float:
    if isinstance(x, float):
        return np.log(x / (1 - x))
    else:
        return torch.log(x / (1 - x))

def makewater(pos: np.ndarray, rot: np.ndarray):
    # N 3, N 3 3 -> N 3 3
    if not isinstance(pos, np.ndarray):
        pos = pos.detach().cpu().numpy()
    if not isinstance(rot, np.ndarray):
        rot = rot.detach().cpu().numpy()
        
    water = np.array([
        [ 0.         , 0.         , 0.        ],
        [ 0.         , 0.         , 0.9572    ],
        [ 0.9266272  , 0.         ,-0.23998721],
    ])
    
    # print( np.einsum("ij,Njk->Nik", water, rot) )
    return np.einsum("ij,Njk->Nik", water, rot) + pos[:, None, :]
    
    