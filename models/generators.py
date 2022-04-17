import torch
from torch import nn as nn


class FCGenerator(nn.Module):
    def __init__(self, z_dim: int, img_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.gen = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, img_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gen(x)