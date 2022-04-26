import os
import shutil
import torch
import torchvision
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from train_config import TrainConfig


def train(dataset: Dataset, train_config: TrainConfig,
          generator, discriminator):

    global_step = epoch = 0

    if train_config.overwrite and os.path.exists(train_config.experiment_dirpath):
        shutil.rmtree(train_config.experiment_dirpath)

    real_images_writer = SummaryWriter(f"{train_config.experiment_dirpath}/real")
    fake_images_writer = SummaryWriter(f"{train_config.experiment_dirpath}/fake")
    stats_writer = SummaryWriter(f"{train_config.experiment_dirpath}/stats")

    dataloader = DataLoader(
        dataset=dataset, batch_size=train_config.batch_size, shuffle=True, num_workers=train_config.num_workers
    )
    num_iterations_per_epoch = len(dataset) // train_config.batch_size

    generator = generator.to(device=train_config.device).train()
    discriminator = discriminator.to(device=train_config.device).train()

    criterion = torch.nn.BCELoss()
    gen_opt = torch.optim.Adam(params=generator.parameters(), lr=train_config.lr, betas=(0.5, 0.999))
    disc_opt = torch.optim.Adam(params=discriminator.parameters(), lr=train_config.lr, betas=(0.5, 0.999))

    while True:
        for batch_idx, (real_img_batch, _) in tqdm(enumerate(dataloader), total=num_iterations_per_epoch, leave=False):
            img_batch = normalize(real_img_batch).to(device=train_config.device)

            # train discriminator
            noise = torch.randn(size=(train_config.batch_size, train_config.z_dim)).to(device=train_config.device)
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

            if global_step % train_config.send_every == 0:
                stats_writer.add_scalar("generator loss", gen_loss, global_step=global_step)
                stats_writer.add_scalar("discriminator loss", disc_loss, global_step=global_step)
                stats_writer.add_scalar("total loss", gen_loss + disc_loss, global_step=global_step)

            if global_step % train_config.show_every == 0:
                # visualize
                real_images_grid = torchvision.utils.make_grid(
                    real_img_batch, normalize=True
                )
                real_images_writer.add_image("real images", real_images_grid, global_step=epoch)

                generated_images = generate(
                    batch_size=real_img_batch.shape[0],
                    z_dim=train_config.z_dim, img_size=train_config.image_size, device=train_config.device,
                    generator=generator
                )
                generated_images = torchvision.utils.make_grid(
                    generated_images, normalize=True
                )
                fake_images_writer.add_image("fake images", generated_images, global_step=global_step)
            global_step += 1
        epoch += 1


def normalize(x):
    return 2 * x - 1


def generate(batch_size: int, z_dim: int, img_size: int, device: torch.device, generator: nn.Module) -> torch.Tensor:
    noise = torch.randn(batch_size, z_dim).to(device=device)
    with torch.no_grad():
        generated_images = generator(noise).view(batch_size, -1, img_size, img_size)
    return generated_images