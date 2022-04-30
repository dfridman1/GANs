import torch
from abc import abstractmethod
from torch import nn as nn

from dcgan.train_config import TrainConfig


class BaseDiscriminator(nn.Module):
    @classmethod
    @abstractmethod
    def from_train_config(cls, train_config: TrainConfig):
        raise NotImplemented


class DCDiscriminator(BaseDiscriminator):
    def __init__(self, in_channels: int, img_dim: int, conditional_dim: int = -1):
        super().__init__()

        assert in_channels in (1, 3)

        num_blocks = self._count_blocks(img_dim)
        assert num_blocks > 0

        if conditional_dim > 0:
            in_channels += conditional_dim

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

    @classmethod
    def from_train_config(cls, train_config: TrainConfig):
        return cls(in_channels=train_config.in_channels, img_dim=train_config.image_size,
                   conditional_dim=train_config.conditional_dim)


if __name__ == '__main__':
    img_dim = 66
    x = torch.randn((4, 1, img_dim, img_dim))
    d = DCDiscriminator(in_channels=1, img_dim=img_dim)
    y = d(x)
    print(y.shape)
    print(y)