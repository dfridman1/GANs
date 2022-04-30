import torch
from dataclasses import dataclass


@dataclass(repr=False)
class TrainConfig:
    experiment_dirpath: str
    image_size: int
    in_channels: int
    overwrite: bool = True
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 128
    num_workers: int = 4
    send_every: int = 50
    show_every: int = 100
    lr: float = 2e-4
    z_dim: int = 100
    conditional_dim: int = 10  # for MNIST and CIFAR-10

    def __post_init__(self):
        assert self.in_channels in (1, 3)
