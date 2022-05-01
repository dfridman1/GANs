import torch
from dataclasses import dataclass


@dataclass(repr=False)
class TrainConfig:
    experiment_dirpath: str
    data_dirpath: str
    in_channels: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    num_workers: int = 2
    num_epochs: int = 200
    lr: float = 2e-4
    image_size: int = 256
    send_every: int = 10
    show_every: int = 100
    lam: float = 10  # weight of cyclic loss
    identity_lam: float = 0.5  # weight of identity loss
    image_pool_size: int = 50
    image_pool_proba: float = 0.5  # with this probability a 'history' image will be returned for discriminator
