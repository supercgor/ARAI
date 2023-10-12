import numpy as np
import numba as nb
@nb.jit(nopython=True)
def getWaterRotate(tag: np.ndarray, inv_ref: np.ndarray) -> np.ndarray:
    O, H1, H2 = tag
    H1, H2 = H1 - O, H2 - O
    H1 = H1 / 0.9572
    H2 = H2 / 0.9572
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

@nb.njit(fastmath=True, cache=True)
def nms_mask(pos: np.ndarray, cutoff: float) -> np.ndarray:
    DIS = cdist2(pos, pos)
    DIS = DIS < cutoff
    DIS = np.triu(DIS, k= 1).astype(np.float_)
    args = np.ones(pos.shape[0], dtype = pos.dtype)
    while True:
        N = pos.shape[0]
        rt = DIS.sum(0)
        rt -= ((rt != 0).astype(np.float_) @ DIS)
        SELECT = rt == 0
        DIS = DIS[SELECT][:, SELECT]
        pos = pos[SELECT]
        args[args.nonzero()] = SELECT
        if N == pos.shape[0]:
            break
    return args



# def nms_mask(pos: np.ndarray, cutoff: float, iu) -> np.ndarray:
#     dis = cdist2(pos, pos)
#     dis[iu] = cutoff
#     total = (dis < cutoff).sum()
#     if total == 0:
#         return np.ones(pos.shape[0], np.bool_)
#     indd = np.unravel_index(np.argpartition(dis, total, axis=None), dis.shape)
#     out = _indset(indd, total, pos.shape[0])
#     return out

def argmatch(A: np.ndarray, B: np.ndarray, cutoff: float) -> tuple[np.ndarray]:
    dis = cdist2(A, B)
    total = (dis < cutoff).sum()
    indd = np.unravel_index(np.argpartition(dis, total, axis=None), dis.shape)
    out = _indset(indd, total)
    return out

@nb.njit(fastmath=True,parallel=True, cache=True)
def _indset(indd: tuple[np.ndarray], total: int, size) -> np.ndarray:
    out = np.ones(size, np.bool_)
    for i in nb.prange(total):
        left, right = indd[0][i], indd[1][i]
        if out[left] and out[right]:
            if left < right:
                out[right] = False
            else:
                out[left] = False
    return out

# @nb.njit(fastmath=True,parallel=True, cache=True)
# def _indset(indd: tuple[np.ndarray], total: int) -> np.ndarray:
#     out = np.ones(indd[0].shape[0], np.bool_)
#     for i in nb.prange(total):
#         left, right = indd[0][i], indd[1][i]
#         if out[left] and out[right]:
#             if left < right:
#                 out[right] = False
#             else:
#                 out[left] = False
#     return out
