import torch
from torch import nn as nn


class FCDiscriminator(nn.Module):
    def __init__(self, img_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.disc = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.disc(x).view(-1)