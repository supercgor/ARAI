import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import torch
import einops
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image

def out2img(x: torch.Tensor, y: torch.Tensor, out_size = (3,128,128)):
    """Turn the prediction to image
    Only show the first image in the batch
    :param x: B x D x H x W x E x C"""
    x = x[0,...,0].sigmoid() # D x H x W x E
    y = y[0,...,0]
    x = x.permute(3,0,1,2)
    y = y.permute(3,0,1,2)
    x = F.interpolate(x[None, ...], size = out_size, mode = "trilinear")[0] # E x 3 x 128 x 128
    y = F.interpolate(y[None, ...], size = out_size, mode = "trilinear")[0]
    IMGS = []
    for i,j in zip(x,y): # i: 3 x 128 x 128
        img = make_grid(torch.stack([i,j,j], dim = 1), nrow = 3)
        IMGS.append(img)
    IMGS = make_grid(IMGS, nrow = 1)
    return IMGS

def label2img(x: torch.Tensor, out_size = (4,32,32), format = "BZXYEC"):
    if "B" in format:
        x = x.select(dim = format.index("B"), index = 0)
        format = format.replace("B", "")
    if "C" in format:
        x = x.select(dim = format.index("C"), index = 0)
        format = format.replace("C", "")
    format = " ".join(format)
    x = x.clip(0,1)
    x = einops.rearrange(x, f"{format} -> E Z X Y")
    x = F.interpolate(x[None,...], size = out_size, mode = "trilinear")
    x = x/ x.max()
    # combine first and second dims -> (1 x (E x Z) x X x Y)
    x = x.flatten(1,2)
    x = x.permute(1,0,2,3)
    return make_grid(x, nrow = out_size[0])
    
class ployView():
    def __init__(self):
        pass
    
    @torch.no_grad()
    def read(self,
            pd_bbox,                        # B x N x 3
            pd_clss,                        # B x N x 3
            gt_bbox,                        # B x M x 3
            gt_clss,                        # B x M x 1
            real_size = (3.0,25.0,25.0),
            match = None,
            cut_off = 0.02
            ):
        pd_match = self.clss_select(pd_clss) # B x N x 1
        real_size = torch.tensor(real_size, device = pd_bbox.device)
        unit_length = torch.prod(real_size) ** (1/3)
        
        pd_bbox = pd_bbox * real_size
        gt_bbox = gt_bbox * real_size
        
        if match is None:
            match, DIS_met = self.match(pd_bbox / unit_length, pd_match, gt_bbox / unit_length, gt_clss)
        else:
            match, DIS_met = match
        B, N, _ = pd_bbox.shape
        
        # type 0: P(O) 1: P(H) 2:P(None), 3:T(O) 4:T(H)

        self.atoms = []
        self.match = []

        """
        atoms[batch] = dataframe
        e.g.
        index  x   y   z     type
          0   1.5 2.5 3.5  "TP(O)"
          
        match[batch] = Dict[dataframe]
        e.g.
        "TP":
        index  x    y    z
          0    1    2    3
          1    2    5    6
          2  None  None None
        """
        
        pd_bbox, gt_bbox, pd_match, gt_clss, DIS_met = map(lambda x: x.detach().cpu().numpy(), (pd_bbox, gt_bbox, pd_match, gt_clss, DIS_met))
        for b, (pdbox, gtbox, pdcls, gtcls, dis_met, (pdind, gtind)) in enumerate(zip(pd_bbox, gt_bbox, pd_match, gt_clss, DIS_met, match)):
            dic = {"TP": [], "FP": [], "FN": [], "TN": []}
            pdbox = pdbox[pdind]
            gtbox = gtbox[gtind]
            pdcls = pdcls[pdind,0]
            gtcls = gtcls[gtind,0]
            disc = dis_met[pdind, gtind]
            # match
            BOND = np.full((N), "", dtype = object)
            mask = pdcls == gtcls
            BOND = np.where(mask & (pdcls != 2), "TP", BOND)
            BOND = np.where(mask & (pdcls == 2), "TN", BOND)
            BOND = np.where((pdcls == 2) & (gtcls != 2), "FN", BOND)
            BOND = np.where((pdcls != 2) & (gtcls == 2), "FP", BOND)
            BOND = np.where(disc > cut_off, "TN", BOND)
            # atoms
            PTYPE = np.full((N), "", dtype = object)
            PTYPE = np.where(pdcls == 2, "P(None)", PTYPE)
            PTYPE = np.where(pdcls == 1, "P(H)", PTYPE)
            PTYPE = np.where(pdcls == 0, "P(O)", PTYPE)
            TTYPE = np.full((N), "T(None)", dtype = object)
            TTYPE = np.where(gtcls == 2, "T(None)", TTYPE)
            TTYPE = np.where(gtcls == 1, "T(H)", TTYPE)
            TTYPE = np.where(gtcls == 0, "T(O)", TTYPE)
            TYPE = np.concatenate([TTYPE, PTYPE], axis = 0)
            ATOMS = np.concatenate([gtbox, pdbox], axis = 0)
            
            bonds = {}
            for bond in ["TP", "FP", "FN", "TN"]:
                box = pdbox[BOND == bond]
                box = np.concatenate([box, gtbox[BOND == bond], np.full(box.shape, None)], axis = 1)
                box = np.reshape(box, (-1, 3))
                # box = pd.DataFrame(box, columns = ["z", "x", "y"])
                bonds[bond] = box
                
            self.atoms.append(pd.DataFrame(np.concatenate([ATOMS, TYPE[:,None]], axis = 1), columns = ["z", "x", "y", "type"]))
            self.match.append(bonds)
    
    @classmethod
    def match(cls, pd_bbox: torch.Tensor, pd_clss: torch.Tensor, gt_bbox: torch.Tensor, gt_clss: torch.Tensor, none_dist = 10, cls_eff = 1, dis_eff = 1):
        B, N, _ = pd_bbox.shape
        DIS_met = torch.cdist(pd_bbox, gt_bbox, p = 2) ** 2
        mask = torch.einsum('B N I, B M I -> B N M', pd_clss != 2, gt_clss != 2).logical_not()
        DIS_met[mask] = none_dist                       # can change
        
        cls_cost = pd_clss.repeat(1, 1, N) != gt_clss.permute(0,2,1).repeat(1, N, 1)
        cls_cost = cls_cost.float()
        COST = cls_eff * cls_cost + dis_eff * DIS_met ** 2

        match_result = tuple(linear_sum_assignment(cost) for cost in COST)
        
        return match_result, DIS_met
    
    def show(self, b):
        atoms = self.atoms[b]
        match = self.match[b]
        # order the dataframe by type
        atoms = atoms.sort_values(by = "type")
        typeSize = {
            "P(O)": 3,
            "P(H)": 1,
            "P(None)": 0,
            "T(O)": 1.5,
            "T(H)": 0.5,
            "T(None)": 0
        }
        cds = ['rgba(176,196,222, 0.5)',
               #'rgba(0, 0, 0, 0.0)', 
               'rgba(70, 130, 180, 0.5)', 
               'rgba(255, 255, 255, 1)', 
               #'rgba(0, 0, 0, 0.0)', 
               'rgba(255, 13, 13, 1)']
        atoms['S'] = atoms['type'].map(typeSize)
        # exclude T(None), P(None) atom
        atoms = atoms[atoms.type != "T(None)"]
        atoms = atoms[atoms.type != "P(None)"]
        fig = px.scatter_3d(atoms,x = 'x', y = 'y', z = 'z', color = 'type', size = 'S',size_max=35, opacity=1,color_discrete_sequence=cds)
        
        for key, pos in match.items():
            if key == "TN":
                continue
            # print(df)
            fig.add_trace(
                go.Scatter3d(z = pos[...,0], x = pos[...,1], y = pos[...,2], mode='lines', name = key, legendgroup=key, showlegend=False)
            )
        
        fig.update_layout(
            scene = dict(
                    zaxis = dict(nticks=4, range=[0,3],),
                     xaxis = dict(nticks=4, range=[0,25],),
                     yaxis = dict(nticks=4, range=[0,25],),
                     aspectratio=dict(x=2, y=2, z=6/25)),
            width=1000,
            margin=dict(r=20, l=10, b=10, t=10),
            template='plotly_dark',
            # plot_bgcolor='rgba(0, 0, 0, 0.5)',
            # paper_bgcolor='rgba(0, 0, 0, 1)',
            )

        
        fig.show()
        

    @classmethod
    def clss_select(cls, clss):
        clss = clss.softmax(dim = -1)                   # (B, N, 3)
        return torch.argmax(clss, dim = -1, keepdim= True)          # (B, N)