#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filename: diffusion.py
# modified: 2023/04/30

from typing import Tuple, Dict, Union, Optional, List
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math

# ===============================
# This module is reference from:
# github: https://github.com/AlexGraikos/diffusion_priors

class GaussianDiffuser(nn.Module):
    '''Gaussian Diffusion process with linear/cosine beta scheduling,
    Also able to due with 3D generation'''
    def __init__(self, T, schedule = "linear"):
        """Init"""
        super().__init__()
        self.register_buffer("_T", torch.tensor(T, dtype=torch.long))
        self._schedule = schedule
        if schedule == "linear":
            b0, bT = 1e-4, 2e-2
            self.register_buffer("_beta", torch.linspace(b0, bT, T))
        elif schedule == "cosine":
            self._alphabar = self.__cos_noise(torch.arange(0, T+1, 1)) / self.__cos_noise(0)
            self.register_buffer("_beta", torch.clamp(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999))
        self.register_buffer("_betabar", torch.cumprod(self._beta, dim=0))
        
        self.register_buffer("_alpha", 1 - self._beta)
        self.register_buffer("_alphabar", torch.cumprod(self.alpha, dim=0))

    def sample(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to imgs, return the noisy imgs (x_t) and noise (ϵ)
        :param x0: imgs (B, C, (D), H, W)
        :param t: time step (B, )
        
        :return x_t: noisy imgs (B, C, (D), H, W)
        :return ϵ: noise (B, C, (D), H, W)"""
        B, *S = x0.shape
        shape = (B, ) + (1, ) * len(S)
        atbar = self._alphabar[t - 1].view(shape)
        epsilon = torch.randn_like(x0)
        xt = atbar.sqrt() * x0 + (1 - atbar).sqrt() * epsilon
        return xt, epsilon

    def __cos_noise(self, t: torch.Tensor | int, eps = 8e-3) -> torch.Tensor:
        """Cosine noise:
        :math:`\\alpha(t) = \\cos^2(\\frac{\\pi}{2} \\frac{t}{T + \\epsilon})`"""
        return torch.cos(math.pi * 0.5 * (t/self.T + eps) / (1+ eps)) ** 2

    def inverse(self, net: nn.Module, shape = (2, 8, 32, 32), step = None, x = None, start_t = None, prior = None, gui = 2.0) -> torch.Tensor:
        if x is None:
            x = torch.randn((1, *shape), device = self._alphabar.device)
        if start_t is None:
            start_t = self.T - 1
        if step is None:
            steps = self.T
        
        for t in range(start_t, start_t - steps, -1):
            t = torch.tensor([t], device = self._alphabar.device)
            at = self.alpha[t-1]
            atbar = self.alphabar[t - 1]
            
            if t > 1:
                z = torch.randn_like(x)
                atbar_prev = self.alphabar[t - 2]
                beta_tilde = ((1 - atbar_prev) / (1 - atbar)) * self.beta[t - 1]
            else:
                z = torch.zeros_like(x)
                beta_tilde = 0
            
            with torch.no_grad():
                if prior is not None:
                    pred = net(x, t, ref = prior)
                else:
                    pred = net(x, t)
                #     xc = net(torch.cat([x, prior], dim = 1),t)
                #     xnc = net(torch.cat([x, torch.zeros_like(prior)], dim = 1),t)
                #     pred = (1 + gui) * xc - gui * xnc
                # else:
                #     pred = net(torch.cat([x, torch.zeros_like(prior)], dim = 1),t)
                    
            x = (1 / at.sqrt()) * (x - ((1 - at) / (1 - atbar).sqrt()) * pred) + math.sqrt(beta_tilde) * z
        
        return x

    @property
    def T(self):
        """Maximum diffusion steps"""
        return self._T
    
    @property
    def schedule(self):
        """Noise schedule"""
        return self._schedule
    
    @property
    def beta(self):
        return self._beta
    
    @property
    def betabar(self):
        return self._betabar
    
    @property
    def alpha(self):
        return self._alpha
    
    @property
    def alphabar(self):
        return self._alphabar