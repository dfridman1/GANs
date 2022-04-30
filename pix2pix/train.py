import os
import shutil
import torch.nn
import torchvision.transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Pix2PixDataset
from models import Pix2PixGenerator, PatchNetDisciminator
from train_config import TrainConfig


def generate(dataset: Dataset, generator: torch.nn.Module, batch_size: int, device: torch.device, num_batches=4):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    generated_images_batch, input_images_batch, gt_batch = [], [], []
    for batch in dataloader:
      gt = batch[0].to(device=device)
      batch = batch[1].to(device=device)  # extract input image
      with torch.no_grad():
          generated_images = generator(batch)
      generated_images_batch.append(generated_images)
      input_images_batch.append(batch)
      gt_batch.append(gt)
      num_batches -= 1
      if num_batches <= 0:
        break
    f = lambda xs: torch.cat(xs, dim=0)
    generated_images_batch = f(generated_images_batch)
    input_images_batch = f(input_images_batch)
    gt_batch = f(gt_batch)
    return generated_images_batch, input_images_batch, gt_batch


def train(train_config: TrainConfig):
    if os.path.exists(train_config.experiment_dirpath):
        shutil.rmtree(train_config.experiment_dirpath)
    global_step = 0

    fake_images_writer = SummaryWriter(f"{train_config.experiment_dirpath}/fake")
    stats_writer = SummaryWriter(f"{train_config.experiment_dirpath}/stats")

    train_dataset = Pix2PixDataset(
        root=train_config.data_dirpath, split="train",
        transformation=torchvision.transforms.Compose([
            torchvision.transforms.Resize(286),
            torchvision.transforms.ToTensor()
        ]),
        random_crop_size=train_config.image_size,
        augment=True
    )
    test_dataset = Pix2PixDataset(
        root=train_config.data_dirpath, split="val",
        transformation=torchvision.transforms.Compose([
            torchvision.transforms.Resize(286),
            torchvision.transforms.CenterCrop(train_config.image_size),
            torchvision.transforms.ToTensor()
        ])
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=train_config.batch_size, shuffle=True, num_workers=train_config.num_workers
    )
    num_iterations_per_epoch = len(train_dataset) // train_config.batch_size

    generator = Pix2PixGenerator(in_channels=train_config.in_channels).to(device=train_config.device)
    discriminator = PatchNetDisciminator(in_channels=2 * train_config.in_channels).to(device=train_config.device)

    bce_criterion = torch.nn.BCELoss() if train_config.gan_mode == "vanilla" else torch.nn.MSELoss()
    l1_criterion = torch.nn.L1Loss()
    g_opt = torch.optim.Adam(params=generator.parameters(), lr=train_config.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(params=discriminator.parameters(), lr=train_config.lr, betas=(0.5, 0.999))

    for epoch in range(train_config.num_epochs):
        for batch_idx, (target_image, input_image) in tqdm(enumerate(train_dataloader), total=num_iterations_per_epoch):
            target_image, input_image = target_image.to(device=train_config.device), input_image.to(device=train_config.device)

            # train discriminator
            generated_image = generator(input_image)

            real_proba = discriminator(torch.cat([input_image, target_image], dim=1))
            fake_proba = discriminator(torch.cat([input_image, generated_image.detach()], dim=1))

            d_loss = (bce_criterion(real_proba, torch.ones_like(real_proba)) +
                      bce_criterion(fake_proba, torch.zeros_like(fake_proba)))
            d_loss = d_loss / 2

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # train generator
            fake_proba = discriminator(torch.cat([input_image, generated_image], dim=1))
            g_loss = bce_criterion(fake_proba, torch.ones_like(fake_proba))
            g_loss = g_loss + train_config.lam * l1_criterion(generated_image, target_image)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            if global_step % train_config.send_every == 0:
                stats_writer.add_scalar("generator loss", g_loss, global_step=global_step)
                stats_writer.add_scalar("discriminator loss", d_loss, global_step=global_step)
                stats_writer.add_scalar("total loss", g_loss + d_loss, global_step=global_step)

            if global_step % train_config.show_every == 0:
                generated_images, input_images, gt = generate(
                    dataset=test_dataset, generator=generator,
                    batch_size=train_config.batch_size, device=train_config.device
                )
                generated_images = torchvision.utils.make_grid(
                    generated_images, normalize=True
                )
                input_images = torchvision.utils.make_grid(
                    input_images, normalize=True
                )
                gt_images = torchvision.utils.make_grid(
                    gt, normalize=True
                )
                generated_images = torch.cat([generated_images, input_images, gt_images], dim=1)
                fake_images_writer.add_image("fake images", generated_images, global_step=global_step)
            global_step += 1
        epoch += 1


def main():
    train_config = TrainConfig(experiment_dirpath="experiments/pix2pix-facades",
                               data_dirpath="data/kaggle/facades/facades")
    train(train_config)


if __name__ == '__main__':
    main()
