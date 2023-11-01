"""
@author: Yanzuo Lu
@email:  luyz5@mail2.sysu.edu.cn
"""

import diffusers
import torch
import torch.nn as nn


class VariationalAutoencoder(nn.Module):
    def __init__(self, pretrained_path, subfolder, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

        self.model = diffusers.AutoencoderKL.from_pretrained(
            pretrained_path, subfolder=subfolder, use_safetensors=True)
        self.model.requires_grad_(False)
        self.model.enable_slicing()

    @torch.no_grad()
    def encode(self, x):
        z = self.model.encode(x).latent_dist
        z = z.sample()
        z = self.scale_factor * z
        return z

    @torch.no_grad()
    def decode(self, z):
        z = 1. / self.scale_factor * z
        x = self.model.decode(z).sample
        return x