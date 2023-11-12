import numpy as np
import torch
import math

class GaussianDiffusion():
    '''Gaussian Diffusion process with linear beta scheduling'''
    def __init__(self, T, schedule):
        # Diffusion steps
        self.T = T
    
        # Noise schedule
        if schedule == 'linear':
            b0=1e-4
            bT=2e-2
            self.beta = np.linspace(b0, bT, T)
        elif schedule == 'cosine':
            self.alphabar = self.__cos_noise(np.arange(0, T+1, 1)) / self.__cos_noise(0) # Generate an extra alpha for bT
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)
            
        self.betabar = np.cumprod(self.beta)
        self.alpha = 1 - self.beta
        self.alphabar = np.cumprod(self.alpha)

    def __cos_noise(self, t):
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t/self.T + offset) / (1+offset)) ** 2
   
    def sample(self, x0, t):        
        # Select noise scales
        noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))        
        atbar = torch.as_tensor(self.alphabar[t-1], device = x0.device, dtype=torch.float).view(noise_dims)
        assert len(atbar.shape) == len(x0.shape), 'Shape mismatch'
        
        # Sample noise and add to x0
        epsilon = torch.randn_like(x0, device=x0.device, dtype = x0.dtype)
        xt = torch.sqrt(atbar) * x0 + torch.sqrt(1-atbar) * epsilon        
        return xt, epsilon
    
    def inverse(self, net, shape=(1,64,64), steps=None, x=None, start_t=None):
        device = next(net.parameters()).device
        # Specify starting conditions and number of steps to run for 
        if x is None:
            x = torch.randn((1,) + shape).to(device)
        if start_t is None:
            start_t = self.T
        if steps is None:
            steps = self.T

        for t in range(start_t, start_t-steps, -1):
            print(f"Inverse step: {t:4d}/{start_t-steps:4d}", end='\r')
            at = self.alpha[t-1]
            atbar = self.alphabar[t-1]
            
            if t > 1:
                z = torch.randn_like(x, device=device)
                atbar_prev = self.alphabar[t-2]
                beta_tilde = self.beta[t-1] * (1 - atbar_prev) / (1 - atbar) 
            else:
                z = torch.zeros_like(x, device=device)
                beta_tilde = 0

            with torch.no_grad():
                t = torch.tensor([t], device=device).view(1)
                t = self.__timestep_embedding(t, 128)
                pred = net(x, t)

            x = (1 / np.sqrt(at)) * (x - ((1-at) / np.sqrt(1-atbar)) * pred) + np.sqrt(beta_tilde) * z

        return x
    
    def inverse_inpainting_by_points(self, net, shape, points, steps=None, x=None, start_t=None):
        """
        _summary_

        Args:
            net (_type_): nn.Module
            shape (_type_): input shape, e.g. (25, 25, 16, 10)
            points (_type_): conditional points, shape: (N, 4) or (N, 10). 
            steps (_type_, optional): _description_. Defaults to None.
            x (_type_, optional): _description_. Defaults to None.
            start_t (_type_, optional): _description_. Defaults to None.
        """
    
    def __timestep_embedding(self, timesteps, dim, max_period=10000):
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