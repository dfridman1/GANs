import os
import shutil
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torchvision


class Generator(nn.Module):
    def __init__(self, z_dim: int, img_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.gen = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, img_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, img_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.disc = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.disc(x).view(-1)


def normalize(x):
    return 2 * x - 1


def generate(batch_size: int, z_dim: int, img_size: int, device: torch.device, generator: nn.Module) -> torch.Tensor:
    back_to_train = generator.training
    generator.eval()
    noise = torch.randn(batch_size, z_dim).to(device=device)
    with torch.no_grad():
        generated_images = generator(noise).view(batch_size, 1, img_size, img_size)
    if back_to_train:
        generator.train()
    return generated_images


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    num_workers = 0
    send_every = 10
    lr = 3e-4
    epoch = 0
    global_step = 0

    exp_dir = "experiments"
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    real_images_writer = SummaryWriter(f"{exp_dir}/real")
    fake_images_writer = SummaryWriter(f"{exp_dir}/fake")
    stats_writer = SummaryWriter(f"{exp_dir}/stats")

    z_dim = 100
    img_size = 28
    img_dim = img_size * img_size * 1

    generator = Generator(z_dim=z_dim, img_dim=img_dim).to(device=device)
    discriminator = Discriminator(img_dim=img_dim).to(device=device)

    criterion = torch.nn.BCELoss()
    gen_opt = torch.optim.Adam(params=generator.parameters(), lr=lr)
    disc_opt = torch.optim.Adam(params=discriminator.parameters(), lr=lr)

    dataset = torchvision.datasets.MNIST(root="data/", train=True, transform=torchvision.transforms.ToTensor())
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    while True:
        for batch_idx, (real_img_batch, _) in enumerate(dataloader):
            img_batch = real_img_batch.view(batch_size, -1).to(device=device)
            img_batch = normalize(img_batch)

            # train discriminator
            noise = torch.randn(size=(batch_size, z_dim)).to(device=device)
            fake_img_batch = generator(noise)

            real_proba = discriminator(img_batch)
            fake_proba = discriminator(fake_img_batch.detach())

            disc_loss = (criterion(real_proba, torch.ones_like(real_proba)) +
                         criterion(fake_proba, torch.zeros_like(fake_proba)))
            disc_loss = disc_loss / 2

            disc_loss.backward()
            disc_opt.step()
            disc_opt.zero_grad()

            # train generator
            fake_proba = discriminator(fake_img_batch)
            gen_loss = criterion(fake_proba, torch.ones_like(fake_proba))

            gen_loss.backward()
            gen_opt.step()
            gen_opt.zero_grad()

            if global_step % send_every == 0:
                stats_writer.add_scalar("generator loss", gen_loss, global_step=global_step)
                stats_writer.add_scalar("discriminator loss", disc_loss, global_step=global_step)
                stats_writer.add_scalar("total loss", gen_loss + disc_loss, global_step=global_step)

            if batch_idx == 0:
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
                fake_images_writer.add_image("fake images", generated_images, global_step=epoch)
            global_step += 1
        epoch += 1


if __name__ == '__main__':
    main()
