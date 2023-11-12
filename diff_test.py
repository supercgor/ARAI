import torch
import hydra
from torch.utils.data import Dataset, DataLoader
import h5py
from model import unet_water
from model.diff import GaussianDiffusion
from utils import library, model_save, model_load, xyz
import tqdm
import numpy as np
import math
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

# %%
def input_transform(x):
    # B, H, W, D, C -> B, C, D, H, W
    return x.permute(0, 4, 3, 1, 2)

def out_transform(inp):
    # B C Z X Y -> B X Y Z C
    inp = inp.permute(0, 3, 4, 2, 1)
    return inp
    
def last_tranform(inp):
    conf, pos, rotx, roty = torch.split(inp, [1, 3, 3, 3], dim = -1)
    c1 = rotx / torch.norm(rotx, dim=-1, keepdim=True)    
    c2 = roty - (c1 * roty).sum(-1, keepdim=True) * c1
    c2 = c2 / torch.norm(c2, dim=-1, keepdim=True)
    return torch.cat([conf, pos, c1, c2], dim=-1)
    

class AFMDataset(Dataset):
    def __init__(self, path: str, label_size= [25, 25, 16], key_filter = None):
        self._path = path
        with h5py.File(path, 'r') as f:
            self._keys = list(f.keys())
            if key_filter is not None:
                self._keys = list(filter(key_filter, self._keys))
            self._len = len(self._keys)
        self._label_size = torch.tensor(label_size)
        
    def __getitem__(self, index: int):
        file_name = self._keys[index]
        out = [file_name]
        hfile = h5py.File(self._path, 'r')
        data = hfile[file_name]

        water = data['pos'][()]
        water = torch.as_tensor(water, dtype=torch.float)
        water = library.decodeWater(water)
        water[:,:3] /= torch.tensor([25.0, 25.0, 16.0])
        water[:,:3].clamp_max_(1 - 1E-7).clamp_min_(0)
        waterind = torch.floor(water[:,:3] * self._label_size).long()
        water[:,:3] = (water[:, :3] * self._label_size) - waterind
        water = torch.cat([torch.ones((water.shape[0],1)), water], axis=1)
        box = torch.zeros((*self._label_size, 10))
        box[tuple(waterind.T)] = water
        box = torch.as_tensor(box, dtype=torch.float)
        box = box + 0.01 * torch.randn_like(box)
        hfile.close()
        return file_name, box
        
    def __len__(self):
        return len(self._keys)

def timestep_embedding(timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps(int64): a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
# %%

# %%
@hydra.main(config_path="config", config_name="unetv3_local", version_base=None)
def main(cfg):
    if torch.cuda.is_available():
        dts = AFMDataset("../data/ice_16A_train.hdf5")
        device = torch.device("cuda:0")
        dtl = DataLoader(dts, batch_size=8, shuffle = True, num_workers= 6, drop_last=True)
        update_every = 40
        
    else:
        dts = AFMDataset("../data/ice_16A/ice_16A_train.hdf5")
        device = torch.device("cpu")
        dtl = DataLoader(dts, batch_size=2, shuffle = True, num_workers= 0, drop_last=True)
        
    net = unet_water(in_size = (16, 25, 25), 
                in_channels = 10, 
                out_size = (16, 25, 25), 
                out_channels = 10,
                out_conv_blocks=1,
                embedding_input=128, 
                embedding_channels=128, 
                channel_mult = [1, 2, 4, 8],
                attention_resolutions = [4,8],
                out_mult = 1,
                ).to(device, dtype=torch.float)
    update_every = 4
    itdtl = iter(dtl)
    
    work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    net.apply_transform(input_transform, out_transform)
    net.eval().requires_grad_(False)

    missed = model_load(net, "outputs/diff_9.pkl")
    if missed:
        print("Missed keys:", missed)
    print('Network parameters:', sum([p.numel() for p in net.parameters()]))

    diffusion = GaussianDiffusion(T=1000, schedule='linear')

    # %% uncondditional sampling
    with torch.no_grad():
        for ind in range(10):
            x = last_tranform(diffusion.inverse(net, shape=(25,25,16, 10)))[0] # B, X, Y, Z, C
            
            _, pos, rot = library.box2orgvec(x, 0.7, 2.0, [25.0, 25.0, 16.0], sort = False, nms = True)
            rot = rot.reshape(-1, 9)[:,:6]
            pos = np.concatenate([pos, rot], axis = -1) # N, 9
            pos = library.encodeWater(pos)
            xyz.write(f"{work_dir}/{ind}_pred.xyz", np.tile(np.array(["O", "H", "H"]),(pos.shape[0],1)), pos.reshape(-1, 3, 3))

if __name__ == '__main__':
    main()
    