import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from model import unet_water
from model.diff import GaussianDiffusion
from utils import lib, model_save
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

def output_transform(x):
    return x.permute(0, 3, 4, 2, 1)

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
        water = lib.encodewater(water)
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
# Train network
if torch.cuda.is_available():
    dts = AFMDataset("../data/ice_16A_train.hdf5")
    device = torch.device("cuda:0")
    dtl = DataLoader(dts, batch_size=8, shuffle = True, num_workers= 6, drop_last=True)
    update_every = 40
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
                   channel_mult = [1, 1, 2, 2],
                   attention_resolutions = [4,8],
                   out_mult = 1,
                   ).to(device, dtype=torch.float)
    update_every = 4
itdtl = iter(dtl)
print(torch.float)


net.apply_transform(input_transform, output_transform)
net.train()

print('Network parameters:', sum([p.numel() for p in net.parameters()]))

opt = torch.optim.Adam(net.parameters(), lr=1e-4)

diffusion = GaussianDiffusion(T=1000, schedule='linear')

epochs = 10
for e in range(epochs):
    print(f'Epoch [{e+1}/{epochs}]')
    
    losses = []
    grads = []
    batch_bar = tqdm.tqdm(dtl)
    for i, batch in enumerate(batch_bar):
        file_name, box = batch
        box = box.to(device)
        # Sample from the diffusion process
        t = np.random.randint(1, diffusion.T+1, box.shape[0]).astype(int)
        boxt, epsilon = diffusion.sample(box, t)
        t = torch.as_tensor(t, device = device, dtype = torch.float)
        t_emb = timestep_embedding(t, 128)
        # Pass through network
        out = net(boxt, t_emb)

        # Compute loss and backprop
        loss = F.mse_loss(out, epsilon)
        opt.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0, error_if_nonfinite=True)
        grads.append(grad.item())
        opt.step()
        
        losses.append(loss.item())
        if i % update_every == 0:
            batch_bar.set_postfix({'Loss': np.mean(losses), 'Grad': np.mean(grads)})
            losses = []
            
    batch_bar.set_postfix({'Loss': np.mean(losses)})
    losses = []

    # Visualize sample
    with torch.no_grad():
        net.eval()
        x = diffusion.inverse(net, shape=(25,25,16, 10)) # B, H, W, D, C
        image = make_grid(x[0,...,(0,)].permute(2, 3, 0, 1).cpu()) # D C H W
        save_image(image, f"diff_{e}.png")
        net.train()

    model_save(net, f"diff_{e}.pkl")