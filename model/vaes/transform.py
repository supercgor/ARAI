import torch
import torch.nn.functional as F

def inp_transform(inp: torch.Tensor):
    # B X Y Z C -> B C Z X Y
    inp = inp.permute(0, 4, 3, 1, 2)
    return inp

def out_transform(inp: torch.Tensor):
    # B C Z X Y -> B X Y Z C
    inp = inp.permute(0, 3, 4, 2, 1)
    conf, pos, rotx, roty = torch.split(inp, [1, 3, 3, 3], dim = -1)
    pos = pos.sigmoid()
    c1 = F.normalize(rotx, dim=-1)
    c2 = roty - (c1 * roty).sum(-1, keepdim=True) * c1
    c2 = F.normalize(c2, dim=-1)
    return torch.cat([conf, pos, c1, c2], dim=-1)
    