import ssl

from train import normalize, generate

ssl._create_default_https_context = ssl._create_unverified_context

import os
import argparse
import shutil
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

from tqdm import tqdm

from models import discriminators
from models import generators

from train import train
from train_config import TrainConfig


def get_dataset_and_in_channels(dataset_name: str, image_size: int):
    name_to_dataset_cls = {
        "mnist": (torchvision.datasets.MNIST, 1),
        "cifar-10": (torchvision.datasets.CIFAR10, 3)
    }
    dataset_cls, in_channels = name_to_dataset_cls[dataset_name]
    dataset = dataset_cls(
        root="data/", train=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor()
        ]),
        download=True
    )
    return dataset, in_channels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name")
    parser.add_argument("--dataset", choices=["mnist", "cifar-10"],  default="mnist")
    parser.add_argument("--image_size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    exp_dir = "experiments"
    if args.exp_name is not None:
        exp_dir = f"{exp_dir}/{args.exp_name}"

    dataset, in_channels = get_dataset_and_in_channels(dataset_name=args.dataset, image_size=args.image_size)

    config = TrainConfig(
        experiment_dirpath=exp_dir,
        image_size=args.image_size,
        in_channels=in_channels
    )

    generator = generators.DCGenerator(out_channels=in_channels, z_dim=config.z_dim, img_dim=config.image_size)
    discriminator = discriminators.DCDiscriminator(in_channels=in_channels, img_dim=config.image_size)

    train(
        dataset=dataset, train_config=config, generator=generator, discriminator=discriminator
    )


if __name__ == '__main__':
    main()
