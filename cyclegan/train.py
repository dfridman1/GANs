import os
import shutil
import json
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from typing import Dict
from cyclegan.train_config import TrainConfig
from cyclegan.helpers import sample_random_batch
from cyclegan import networks
from cyclegan.dataset import ImageDataset


def save_model(save_dir: str, name_to_module: Dict[str, torch.nn.Module], train_config: TrainConfig):
    os.makedirs(save_dir, exist_ok=True)
    for name, module in name_to_module.items():
        out_filepath = os.path.join(save_dir, f"{name}.pt")
        torch.save(module.state_dict(), out_filepath)
    with open(os.path.join(save_dir, "config.json"), "w") as fp:
        json.dump(train_config.__dict__, fp, indent=4, sort_keys=True)


def generate(dataset: Dataset, generator: torch.nn.Module, batch_size: int, device: torch.device, num_batches=4):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    generated_images_batch, input_images_batch = [], []
    for batch in dataloader:
      batch = batch.to(device=device)  # extract input image
      with torch.no_grad():
          generated_images = generator(batch)
      generated_images_batch.append(generated_images)
      input_images_batch.append(batch)
      num_batches -= 1
      if num_batches <= 0:
        break
    f = lambda xs: torch.cat(xs, dim=0)
    generated_images_batch = f(generated_images_batch)
    input_images_batch = f(input_images_batch)
    return generated_images_batch, input_images_batch


def train(train_config: TrainConfig):
    if os.path.exists(train_config.experiment_dirpath):
        shutil.rmtree(train_config.experiment_dirpath)

    global_step = 0

    fake_images_a_to_b_writer = SummaryWriter(f"{train_config.experiment_dirpath}/fake_a->b")
    fake_images_b_to_a_writer = SummaryWriter(f"{train_config.experiment_dirpath}/fake_b->a")
    stats_writer = SummaryWriter(f"{train_config.experiment_dirpath}/stats")

    resize_factor = 1.1
    resize_size = int(round(resize_factor * train_config.image_size))
    train_dataset_a = ImageDataset(
        root=os.path.join(train_config.data_dirpath, "trainA"),
        transforms=transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomCrop(train_config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        image_size=train_config.image_size
    )
    train_dataset_b = ImageDataset(
        root=os.path.join(train_config.data_dirpath, "trainB"),
        transforms=transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomCrop(train_config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        image_size=train_config.image_size
    )
    test_dataset_a = ImageDataset(
        root=os.path.join(train_config.data_dirpath, "testA"),
        transforms=transforms.Compose([
            transforms.Resize(286),
            transforms.CenterCrop(train_config.image_size),
            transforms.ToTensor()
        ]),
        image_size=train_config.image_size
    )
    test_dataset_b = ImageDataset(
        root=os.path.join(train_config.data_dirpath, "testB"),
        transforms=transforms.Compose([
            transforms.Resize(286),
            transforms.CenterCrop(train_config.image_size),
            transforms.ToTensor()
        ]),
        image_size=train_config.image_size
    )

    train_dataloader_a = DataLoader(
        dataset=train_dataset_a, batch_size=train_config.batch_size,
        shuffle=True, num_workers=train_config.num_workers
    )
    train_dataloader_b = DataLoader(
        dataset=train_dataset_b, batch_size=train_config.batch_size,
        shuffle=True, num_workers=train_config.num_workers
    )
    num_iterations_per_epoch = len(train_dataset_a) // train_config.batch_size

    g_a = networks.Generator(in_channels=train_config.in_channels, num_residuals=train_config.num_residual_blocks).to(device=train_config.device)
    g_b = networks.Generator(in_channels=train_config.in_channels, num_residuals=train_config.num_residual_blocks).to(device=train_config.device)
    d_a = networks.Discriminator(in_channels=train_config.in_channels).to(device=train_config.device)
    d_b = networks.Discriminator(in_channels=train_config.in_channels).to(device=train_config.device)

    l2_criterion = torch.nn.MSELoss()  # lsgan
    l1_criterion = torch.nn.L1Loss()

    create_opt = lambda parameters: torch.optim.Adam(
        parameters, lr=train_config.lr, betas=(0.5, 0.999)
    )
    g_a_opt = create_opt(g_a.parameters())
    g_b_opt = create_opt(g_b.parameters())
    d_a_opt = create_opt(d_a.parameters())
    d_b_opt = create_opt(d_b.parameters())

    for epoch in range(train_config.num_epochs):
        for real_a in tqdm(train_dataloader_a, total=num_iterations_per_epoch):
            real_b = sample_random_batch(dataset=train_dataset_b, batch_size=real_a.shape[0])
            real_a, real_b = real_a.to(device=train_config.device), real_b.to(device=train_config.device)

            # train generators (GAN loss)
            generated_b = g_a(real_a)
            fake_proba_b = d_b(generated_b)
            g_a_gan_loss = l2_criterion(fake_proba_b, torch.ones_like(fake_proba_b))

            generated_a = g_b(real_b)
            fake_proba_a = d_a(generated_a)
            g_b_gan_loss = l2_criterion(fake_proba_a, torch.ones_like(fake_proba_a))

            # train generators (recycle loss)
            recycled_a = g_b(generated_b)
            g_a_cycle_loss = train_config.lam * l1_criterion(recycled_a, real_a)
            recycled_b = g_a(generated_a)
            g_b_cycle_loss = train_config.lam * l1_criterion(recycled_b, real_b)

            # identity loss (for regularization purposes)
            identity_a = g_a(real_a)
            identity_b = g_b(real_b)
            identity_loss_a = train_config.identity_lam * l1_criterion(identity_a, real_a)
            identity_loss_b = train_config.identity_lam * l1_criterion(identity_b, real_b)

            g_loss = (g_a_gan_loss + g_b_gan_loss +
                      g_a_cycle_loss + g_b_cycle_loss +
                      identity_loss_a + identity_loss_b)

            g_a_opt.zero_grad()
            g_b_opt.zero_grad()
            g_loss.backward()
            g_a_opt.step()
            g_b_opt.step()

            # train discriminators
            # TODO: sample previously generated images
            fake_proba_b = d_b(generated_b.detach())  # detach?
            real_proba_b = d_b(real_b)
            fake_proba_a = d_a(generated_a.detach())  # detach?
            real_proba_a = d_a(real_a)

            d_loss = (
                l2_criterion(fake_proba_b, torch.zeros_like(fake_proba_b)) +
                l2_criterion(real_proba_b, torch.ones_like((real_proba_b))) +
                l2_criterion(fake_proba_a, torch.zeros_like(fake_proba_a)) +
                l2_criterion(real_proba_a, torch.ones_like(real_proba_a))
            )
            d_loss = d_loss / 2  # make training of D's slower

            d_a_opt.zero_grad()
            d_b_opt.zero_grad()
            d_loss.backward()
            d_a_opt.step()
            d_b_opt.step()

            if global_step % train_config.send_every == 0:
                stats_writer.add_scalar("generator loss", g_loss, global_step=global_step)
                stats_writer.add_scalar("discriminator loss", d_loss, global_step=global_step)
                stats_writer.add_scalar("total loss", g_loss + d_loss, global_step=global_step)

            if global_step % train_config.show_every == 0:
                # A -> B
                generated_images, input_images = generate(
                    dataset=test_dataset_a, generator=g_a,
                    batch_size=train_config.batch_size, device=train_config.device
                )
                generated_images = torchvision.utils.make_grid(
                    generated_images, normalize=True
                )
                input_images = torchvision.utils.make_grid(
                    input_images, normalize=True
                )
                generated_images = torch.cat([generated_images, input_images], dim=1)
                fake_images_a_to_b_writer.add_image("fake images (A -> B)", generated_images, global_step=global_step)

                # B -> A
                generated_images, input_images = generate(
                    dataset=test_dataset_b, generator=g_b,
                    batch_size=train_config.batch_size, device=train_config.device
                )
                generated_images = torchvision.utils.make_grid(
                    generated_images, normalize=True
                )
                input_images = torchvision.utils.make_grid(
                    input_images, normalize=True
                )
                generated_images = torch.cat([generated_images, input_images], dim=1)
                fake_images_b_to_a_writer.add_image("fake images (A -> B)", generated_images, global_step=global_step)
            global_step += 1
        save_model(
            save_dir=os.path.join(train_config.experiment_dirpath, "saved_weights"),
            name_to_module={
                "g_a": g_a, "g_b": g_b, "d_a": d_a, "d_b": d_b
            },
            train_config=train_config
        )


def main():
    config = TrainConfig(
        experiment_dirpath="experiments/cyclegan-test",
        data_dirpath="datasets/monet2photo"
    )
    train(config)


if __name__ == '__main__':
    main()
