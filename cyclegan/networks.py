import torch
import torch.nn as nn

from pix2pix.models import PatchNetDisciminator


class Discriminator(PatchNetDisciminator):
    def __init__(self, in_channels: int):
        super().__init__(in_channels=in_channels)


class Generator(nn.Module):
    def __init__(self, in_channels: int, num_downsamples: int = 2, num_residual_blocks: int = 9):
        super().__init__()

        assert in_channels in (1, 3)
        assert num_downsamples > 0

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, padding_mode="reflect", bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )

        input_channels = 64
        downsample_conv = []
        for _ in range(num_downsamples):
            out_channels = 2 * input_channels
            downsample_conv.append(make_conv3x3(input_channels, out_channels, stride=2))
            input_channels = out_channels

        self.downsample_conv = nn.Sequential(*downsample_conv)
        self.residuals = nn.Sequential(*[
            ResidualBlock(input_channels, input_channels)
            for _ in range(num_residual_blocks)
        ])

        upsample_conv = []
        for _ in range(num_downsamples):
            assert input_channels % 2 == 0
            out_channels = input_channels // 2
            upsample_conv.append(make_transposed_conv3x3(input_channels, out_channels))
            input_channels = out_channels

        self.transposed_conv = nn.Sequential(*upsample_conv)
        self.final_conv = nn.Conv2d(
            input_channels, in_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        x = self.downsample_conv(x)
        x = self.residuals(x)
        x = self.transposed_conv(x)
        x = self.final_conv(x)
        x = torch.tanh(x)
        return x

    @classmethod
    def from_image_size(cls, in_channels: int, image_size: int):
        image_size_to_params = {
            64: {"num_downsamples": 1, "num_residual_blocks": 3},
            128: {"num_downsamples": 2, "num_residual_blocks": 6},
            256: {"num_downsamples": 2, "num_residual_blocks": 9}
        }
        if image_size not in image_size_to_params:
            raise KeyError(f"{image_size} not supported. Must be one of {', '.join(map(str, image_size_to_params.keys()))}")
        params = image_size_to_params[image_size]
        return cls(in_channels=in_channels, **params)


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
    g = Generator(3, num_residual_blocks=9)
    y = g(x)
    print(y.shape)
    d = Discriminator(in_channels=3)
    t = d(y)
    print(t.shape)