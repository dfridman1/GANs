import argparse

import torch.nn
import torchvision.transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Pix2PixDataset
from models import Pix2PixGenerator, PatchNetDisciminator


def generate(dataset: Dataset, generator: torch.nn.Module, batch_size: int, device: torch.device):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    batch = None
    for batch in dataloader:
        break
    batch = batch[1]  # extract input image
    with torch.no_grad():
        generated_images = generator(batch)
    return generated_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="pix2pix-facades")
    parser.add_argument("--data_root", default="data/kaggle/facades/facades")
    return parser.parse_args()


def main():
    args = parse_args()

    batch_size = 4
    num_workers = 2
    lr = 2e-4
    image_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lam = 1  # L1 loss weight
    send_every = 10
    show_every = 100
    experiment_dirpath = f"experiments/{args.exp_name}"

    global_step = epoch = 0

    real_images_writer = SummaryWriter(f"{experiment_dirpath}/real")
    fake_images_writer = SummaryWriter(f"{experiment_dirpath}/fake")
    stats_writer = SummaryWriter(f"{experiment_dirpath}/stats")

    train_dataset = Pix2PixDataset(
        root=args.data_root, split="train",
        transformation=torchvision.transforms.Compose([
            torchvision.transforms.Resize(286),
            torchvision.transforms.ToTensor()
        ]),
        random_crop_size=image_size
    )
    test_dataset = Pix2PixDataset(
        root=args.data_root, split="test",
        transformation=torchvision.transforms.Compose([
            torchvision.transforms.Resize(286),
            torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.ToTensor()
        ])
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    num_iterations_per_epoch = len(train_dataset) // batch_size

    in_channels = 3
    generator = Pix2PixGenerator(in_channels=in_channels).to(device=device)
    discriminator = PatchNetDisciminator(in_channels=in_channels).to(device=device)

    bce_criterion = torch.nn.BCELoss()
    l1_criterion = torch.nn.L1Loss()
    g_opt = torch.optim.Adam(params=generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(params=discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    while True:
        for batch_idx, (target_image, input_image) in tqdm(enumerate(train_dataloader), total=num_iterations_per_epoch):
            target_image, input_image = target_image.to(device=device), input_image.to(device=device)

            # train discriminator
            generated_image = generator(input_image)

            real_proba = discriminator(target_image)
            fake_proba = discriminator(generated_image.detach())

            d_loss = (bce_criterion(real_proba, torch.ones_like(real_proba)) +
                      bce_criterion(fake_proba, torch.zeros_like(fake_proba)))
            d_loss = d_loss / 2

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # train generator
            fake_proba = discriminator(generated_image)
            g_loss = bce_criterion(fake_proba, torch.ones_like(fake_proba))
            g_loss = g_loss + lam * l1_criterion(generated_image, target_image)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            if global_step % send_every == 0:
                stats_writer.add_scalar("generator loss", g_loss, global_step=global_step)
                stats_writer.add_scalar("discriminator loss", d_loss, global_step=global_step)
                stats_writer.add_scalar("total loss", g_loss + d_loss, global_step=global_step)

            if global_step % show_every == 0:
                # visualize
                real_images_grid = torchvision.utils.make_grid(
                    target_image, normalize=True
                )
                real_images_writer.add_image("real images", real_images_grid, global_step=epoch)

                generated_images = generate(dataset=test_dataset, generator=generator,
                                            batch_size=batch_size, device=device)
                generated_images = torchvision.utils.make_grid(
                    generated_images, normalize=True
                )
                fake_images_writer.add_image("fake images", generated_images, global_step=global_step)

            global_step += 1
        epoch += 1


if __name__ == '__main__':
    main()
