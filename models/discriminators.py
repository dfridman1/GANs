import torch
from torch import nn as nn


class FCDiscriminator(nn.Module):
    def __init__(self, img_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.disc = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.disc(x).view(-1)


class DCDiscriminator(nn.Module):
    def __init__(self, in_channels: int, img_dim: int):
        super().__init__()

        assert in_channels in (1, 3)

        num_blocks = self._count_blocks(img_dim)
        assert num_blocks > 0

        conv = []
        out_channels = 128
        for i in range(num_blocks - 1):
            conv.append(
                self._block(
                    in_channels=in_channels, out_channels=out_channels,
                    bias=(i == 0), batchnorm=(i > 0)
                )
            )
            in_channels, out_channels = out_channels, out_channels * 2
        conv.append(
            self._block(
                in_channels=in_channels, out_channels=out_channels,
                bias=True, batchnorm=False
            )
        )

        final_spatial_size = img_dim // (2 ** num_blocks)
        if final_spatial_size > 1:
            conv.append(
                nn.Conv2d(
                    in_channels=out_channels, out_channels=1,
                    kernel_size=final_spatial_size, stride=1, padding=0, bias=True
                )
            )
        self.conv = nn.Sequential(*conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(-1, 1)
        x = torch.sigmoid(x)
        return x

    @staticmethod
    def _block(in_channels: int, out_channels: int, bias: bool, batchnorm: bool = True):
        layers = [
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=4, stride=2, padding=1, bias=bias
            )
        ]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    @staticmethod
    def _count_blocks(img_dim: int):
        cnt = 0
        while img_dim > 0 and img_dim % 2 == 0:
            cnt += 1
            img_dim //= 2
        return cnt


if __name__ == '__main__':
    img_dim = 66
    x = torch.randn((4, 1, img_dim, img_dim))
    d = DCDiscriminator(in_channels=1, img_dim=img_dim)
    y = d(x)
    print(y.shape)
    print(y)