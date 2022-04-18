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


class DCGenerator(nn.Module):
    def __init__(self, out_channels: int, z_dim: int, img_dim: int):
        super().__init__()

        num_blocks = self._count_blocks(img_dim)
        assert num_blocks > 0

        self.proj_spatial_size = img_dim // (2 ** num_blocks)
        self.proj_num_channels = 128 * (2 ** (num_blocks - 1))

        self.proj = nn.Linear(z_dim, self.proj_spatial_size * self.proj_spatial_size * self.proj_num_channels)

        tconv = []
        in_channels = self.proj_num_channels
        for _ in range(num_blocks - 1):
            tconv.append(self._block(in_channels=in_channels, out_channels=in_channels // 2, bias=False))
            in_channels //= 2
        tconv.append(self._block(in_channels=in_channels, out_channels=out_channels, bias=True, batchnorm=False))
        self.tconv = nn.Sequential(*tconv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).view(-1, self.proj_num_channels, self.proj_spatial_size, self.proj_spatial_size)
        x = self.tconv(x)
        x = torch.tanh(x)
        return x

    @staticmethod
    def _block(
            in_channels: int, out_channels: int, bias: bool, batchnorm: bool = True,
            kernel_size=4, stride=2, padding=1
    ):
        layers = [
            nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
        ]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    @staticmethod
    def _count_blocks(img_dim):
        cnt = 0
        while img_dim > 4 and img_dim % 2 == 0:
            cnt += 1
            img_dim //= 2
        return cnt


if __name__ == '__main__':
    z_dim = 100
    x = torch.randn((32, z_dim))
    g = DCGenerator(out_channels=1, z_dim=z_dim, img_dim=32)
    y = g(x)
    print(y.shape)