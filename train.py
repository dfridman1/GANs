import os
import shutil
import numpy as np
import torch
import torchvision
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import helpers
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
        for batch_idx, (real_img_batch, labels) in tqdm(enumerate(dataloader), total=num_iterations_per_epoch, leave=False):
            img_batch = normalize(real_img_batch)
            if train_config.conditional_dim > 0:
                conditional_input = helpers.conditional_input_encoder_discriminator(
                    labels=labels, cardinality=train_config.conditional_dim, spatial_size=train_config.image_size
                )
                img_batch = torch.cat([img_batch, conditional_input], dim=1)
            img_batch = img_batch.to(device=train_config.device)

            # train discriminator
            noise = torch.randn(size=(len(labels), train_config.z_dim))
            if train_config.conditional_dim > 0:
                conditional_input = helpers.conditional_input_encoder_generator(
                    labels=labels, cardinality=train_config.conditional_dim
                )
                noise = torch.cat([noise, conditional_input], dim=1)
            noise = noise.to(device=train_config.device)
            fake_img_batch = generator(noise)
            if train_config.conditional_dim > 0:
                conditional_input = helpers.conditional_input_encoder_discriminator(
                    labels=labels, cardinality=train_config.conditional_dim, spatial_size=train_config.image_size
                ).to(device=train_config.device)
                fake_img_batch = torch.cat([fake_img_batch, conditional_input], dim=1)

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

                generated_images = generate(train_config=train_config, generator=generator)
                generated_images = torchvision.utils.make_grid(
                    generated_images, normalize=True
                )
                fake_images_writer.add_image("fake images", generated_images, global_step=global_step)
            global_step += 1
        epoch += 1


def normalize(x):
    return 2 * x - 1


def generate(train_config: TrainConfig, generator: nn.Module) -> torch.Tensor:
    noise = torch.randn(train_config.batch_size, train_config.z_dim)
    if train_config.conditional_dim > 0:
        label = np.random.randint(low=0, high=train_config.conditional_dim)
        labels = np.asarray([label] * train_config.batch_size)
        labels = torch.from_numpy(labels)
        conditional_input = helpers.conditional_input_encoder_generator(
            labels=labels, cardinality=train_config.conditional_dim
        )
        noise = torch.cat([noise, conditional_input], dim=1)
    noise = noise.to(device=train_config.device)
    with torch.no_grad():
        generated_images = generator(noise).view(train_config.batch_size, -1, train_config.image_size, train_config.image_size)
    return generated_images
