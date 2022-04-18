import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import argparse
import shutil
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torchvision

from tqdm import tqdm

from models import discriminators
from models import generators


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


def normalize(x):
    return 2 * x - 1


def generate(batch_size: int, z_dim: int, img_size: int, device: torch.device, generator: nn.Module) -> torch.Tensor:
    noise = torch.randn(batch_size, z_dim).to(device=device)
    with torch.no_grad():
        generated_images = generator(noise).view(batch_size, -1, img_size, img_size)
    return generated_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name")
    parser.add_argument("--dataset", choices=["mnist", "cifar-10"],  default="mnist")
    parser.add_argument("--image_size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    num_workers = 4
    send_every = 10
    show_every = 100
    lr = 2e-4
    epoch = 0
    global_step = 0

    exp_dir = "experiments"
    if args.exp_name is not None:
        exp_dir = f"{exp_dir}/{args.exp_name}"

    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    real_images_writer = SummaryWriter(f"{exp_dir}/real")
    fake_images_writer = SummaryWriter(f"{exp_dir}/fake")
    stats_writer = SummaryWriter(f"{exp_dir}/stats")

    z_dim = 100
    img_size = args.image_size

    dataset, in_channels = get_dataset_and_in_channels(dataset_name=args.dataset, image_size=img_size)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    num_iterations_per_epoch = len(dataset) // batch_size

    generator = generators.DCGenerator(out_channels=in_channels, z_dim=z_dim, img_dim=img_size).to(device=device).train()
    discriminator = discriminators.DCDiscriminator(in_channels=in_channels, img_dim=img_size).to(device=device).train()

    criterion = torch.nn.BCELoss()
    gen_opt = torch.optim.Adam(params=generator.parameters(), lr=lr, betas=(0.5, 0.999))
    disc_opt = torch.optim.Adam(params=discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    while True:
        for batch_idx, (real_img_batch, _) in tqdm(enumerate(dataloader), total=num_iterations_per_epoch, leave=False):
            img_batch = normalize(real_img_batch).to(device=device)

            # train discriminator
            noise = torch.randn(size=(batch_size, z_dim)).to(device=device)
            fake_img_batch = generator(noise)

            real_proba = discriminator(img_batch)
            fake_proba = discriminator(fake_img_batch.detach())

            disc_loss = (criterion(real_proba, torch.ones_like(real_proba)) +
                         criterion(fake_proba, torch.zeros_like(fake_proba)))
            disc_loss = disc_loss / 2

            disc_opt.zero_grad()
            disc_loss.backward()
            disc_opt.step()

            # train generator
            fake_proba = discriminator(fake_img_batch)
            gen_loss = criterion(fake_proba, torch.ones_like(fake_proba))

            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

            if global_step % send_every == 0:
                stats_writer.add_scalar("generator loss", gen_loss, global_step=global_step)
                stats_writer.add_scalar("discriminator loss", disc_loss, global_step=global_step)
                stats_writer.add_scalar("total loss", gen_loss + disc_loss, global_step=global_step)

            if global_step % show_every == 0:
                # visualize
                real_images_grid = torchvision.utils.make_grid(
                    real_img_batch, normalize=True
                )
                real_images_writer.add_image("real images", real_images_grid, global_step=epoch)

                generated_images = generate(
                    batch_size=real_img_batch.shape[0],
                    z_dim=z_dim, img_size=img_size, device=device,
                    generator=generator
                )
                generated_images = torchvision.utils.make_grid(
                    generated_images, normalize=True
                )
                fake_images_writer.add_image("fake images", generated_images, global_step=global_step)
            global_step += 1
        epoch += 1


if __name__ == '__main__':
    main()
