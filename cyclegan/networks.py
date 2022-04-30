import torch
import torch.nn as nn

from pix2pix.models import PatchNetDisciminator


class Discriminator(PatchNetDisciminator):
    def __init__(self, in_channels: int):
        super().__init__(in_channels=in_channels)


class Generator(nn.Module):
    def __init__(self, in_channels: int, num_residuals: int = 6):
        super().__init__()

        assert in_channels in (1, 3)
        assert num_residuals in (6, 9)

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, padding_mode="reflect", bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        self.conv = nn.Sequential(
            make_conv3x3(64, 128, stride=2),
            make_conv3x3(128, 256, stride=2)
        )
        self.residuals = nn.Sequential(*[
            ResidualBlock(256, 256)
            for i in range(num_residuals)
        ])
        self.transposed_conv = nn.Sequential(
            make_transposed_conv3x3(256, 128),
            make_transposed_conv3x3(128, 64)
        )
        self.final_conv = nn.Conv2d(
            64, in_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        x = self.conv(x)
        x = self.residuals(x)
        x = self.transposed_conv(x)
        x = self.final_conv(x)
        x = torch.tanh(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Sequential(
            make_conv3x3(in_channels, out_channels, stride=1),
            make_conv3x3(out_channels, out_channels, stride=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


def make_conv3x3(in_channels: int, out_channels: int, stride: int):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, padding_mode="reflect",
            bias=False
        ),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(True)
    )


def make_transposed_conv3x3(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        ),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(True)
    )


if __name__ == '__main__':
    x = torch.randn((4, 3, 256, 256))
    g = Generator(3, num_residuals=9)
    y = g(x)
    print(y.shape)
    d = Discriminator(in_channels=3)
    t = d(y)
    print(t.shape)