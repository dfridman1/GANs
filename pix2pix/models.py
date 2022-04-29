import torch
import torch.nn as nn

from typing import Tuple


class Pix2PixGenerator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encodings = self.encoder(x)
        return self.decoder(*encodings)


class PatchNetDisciminator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64, activation=nn.LeakyReLU(0.2), batchnorm=False, dropout=False)
        self.conv2 = ConvBlock(64, 128, activation=nn.LeakyReLU(0.2), batchnorm=True, dropout=False)
        self.conv3 = ConvBlock(128, 256, activation=nn.LeakyReLU(0.2), batchnorm=True, dropout=False)
        self.conv4 = ConvBlock(256, 512, activation=nn.LeakyReLU(0.2), batchnorm=True, dropout=False)
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=64, activation=nn.LeakyReLU(0.2), batchnorm=False)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128, activation=nn.LeakyReLU(0.2))
        self.conv3 = ConvBlock(in_channels=128, out_channels=256, activation=nn.LeakyReLU(0.2))
        self.conv4 = ConvBlock(in_channels=256, out_channels=512, activation=nn.LeakyReLU(0.2))
        self.conv5 = ConvBlock(in_channels=512, out_channels=512, activation=nn.LeakyReLU(0.2))
        self.conv6 = ConvBlock(in_channels=512, out_channels=512, activation=nn.LeakyReLU(0.2))
        self.conv7 = ConvBlock(in_channels=512, out_channels=512, activation=nn.LeakyReLU(0.2))
        self.conv8 = ConvBlock(in_channels=512, out_channels=512, activation=nn.LeakyReLU(0.2))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x2 = self.conv1(x)
        x4 = self.conv2(x2)
        x8 = self.conv3(x4)
        x16 = self.conv4(x8)
        x32 = self.conv5(x16)
        x64 = self.conv6(x32)
        x128 = self.conv7(x64)
        x256 = self.conv8(x128)

        return x2, x4, x8, x16, x32, x64, x128, x256


class Decoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.conv1 = TransposeConvBlock(512, 512, dropout=True)
        self.conv2 = TransposeConvBlock(1024, 512, dropout=True)
        self.conv3 = TransposeConvBlock(1024, 512, dropout=True)
        self.conv4 = TransposeConvBlock(1024, 512, dropout=False)
        self.conv5 = TransposeConvBlock(1024, 256, dropout=False)
        self.conv6 = TransposeConvBlock(512, 256, dropout=False)
        self.conv6 = TransposeConvBlock(512, 128, dropout=False)
        self.conv7 = TransposeConvBlock(256, 64, dropout=False)
        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(
                128, in_channels, kernel_size=4, stride=2, padding=1,
                bias=True
            ),
            nn.Tanh()
        )

    def forward(self, x2, x4, x8, x16, x32, x64, x128, x256):
        decoded_x128 = self.conv1(x256)  # 2x2
        decoded_x64 = self.conv2(torch.cat([x128, decoded_x128], dim=1))  # 4x4
        decoded_x32 = self.conv3(torch.cat([x64, decoded_x64], dim=1))  # 8x8
        decoded_x16 = self.conv4(torch.cat([x32, decoded_x32], dim=1))  # 16x16
        decoded_x8 = self.conv5(torch.cat([x16, decoded_x16], dim=1))  # 32x32
        decoded_x4 = self.conv6(torch.cat([x8, decoded_x8], dim=1))  # 64x64
        decoded_x2 = self.conv7(torch.cat([x4, decoded_x4], dim=1))  # 128x128
        decoded = self.conv8(torch.cat([x2, decoded_x2], dim=1))  # 256x256
        return decoded


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 activation: nn.Module, batchnorm: bool = True, dropout: bool = False):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=4, stride=2, padding=1, bias=not batchnorm)
        ]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(activation)
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TransposeConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: bool):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)
    g = Pix2PixGenerator(in_channels=3)
    y = g(x)
    d = PatchNetDisciminator(in_channels=3)
    t = d(y)
    print(t)
