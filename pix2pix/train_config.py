import torch
from dataclasses import dataclass


@dataclass(repr=False)
class TrainConfig:
    experiment_dirpath: str
    data_dirpath: str
    in_channels: int = 3
    num_epochs: int = 100
    batch_size: int = 1
    num_workers: int = 2
    lr: float = 2e-4
    image_size: int = 256
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lam = 100  # L1 weight
    send_every: int = 10
    show_every: int = 100
