import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from fileio.asehdf import load_by_name
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from scipy.spatial import cKDTree

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
    
    start = time.time()

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

        if progress:
            print('Computing RDF     Radius {}/{}    Time elapsed: {:.3f} s'.format(r_idx+1, len(radii), time.time()-start), end='\r', flush=True)

    return g_r, radii


class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        with h5py.File(self.path, 'r') as f:
            self.keys = list(f.keys())
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        k = self.keys[idx]
        with h5py.File(self.path, 'r') as f:
            atoms = load_by_name(f, k)
            atoms = atoms[atoms.symbols == 'O']
            atoms = atoms[atoms.positions[:,2] > 4]
            atoms.positions[:,2] -= 4
        gr, _ = rdf(atoms.positions, dr=0.1, dims=[25.0, 25.0, 12.0])
        return gr

all_gr = []
path = ['../data/ice_16A_R_hup_low_T_train.hdf5', '../data/ice_16A_R_hup_low_T_test.hdf5']

for p in path:
    dts = MyDataset(p)
    dtl = DataLoader(dts, batch_size=18, num_workers=6, multiprocessing_context='fork')
    for gr in tqdm(dtl):
        all_gr.append(gr)
        
       
all_gr = torch.cat(all_gr, dim=0).numpy()
mean_gr = np.mean(all_gr, axis=0)
r = np.arange(len(mean_gr)) * 0.1
print(", ".join([f"{i:.3f}" for i in r]))
print(", ".join([f"{i:.3f}" for i in mean_gr]))
plt.plot(r, mean_gr)

plt.show()


"""
r
[0.000, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000, 
1.100, 1.200, 1.300, 1.400, 1.500, 1.600, 1.700, 1.800, 1.900, 2.000, 2.100, 
2.200, 2.300, 2.400, 2.500, 2.600, 2.700, 2.800, 2.900, 3.000, 3.100, 3.200, 
3.300, 3.400, 3.500, 3.600, 3.700, 3.800, 3.900, 4.000, 4.100, 4.200, 4.300, 
4.400, 4.500, 4.600, 4.700, 4.800, 4.900, 5.000, 5.100, 5.200, 5.300]
gr
[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 
0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 
0.000, 0.000, 0.135, 2.576, 6.314, 4.258, 1.486, 0.336, 0.078, 0.017, 0.003, 
0.003, 0.009, 0.023, 0.056, 0.130, 0.295, 0.581, 1.059, 1.718, 2.299, 2.754, 
2.522, 1.503, 0.677, 0.365, 0.282, 0.229, 0.179, 0.142, 0.092, 0.044]
"""