import os

import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset
from glob import glob
from tqdm import tqdm


def random_crop(image1: torch.Tensor, image2: torch.Tensor, crop_size: int):
    assert crop_size % 2 == 0
    assert image1.shape == image2.shape
    c, h, w = image1.shape
    image = torch.zeros((2 * c, 2 * h, 2 * w), dtype=torch.float32)
    concat_image = torch.cat([image1, image2], dim=0)
    image[:, h // 2 : -h  // 2, w // 2 : -w // 2] = concat_image
    start_y = np.random.randint(0, image.shape[1] - crop_size + 1)
    start_x = np.random.randint(0, image.shape[2] - crop_size + 1)
    image = image[:, start_y:start_y + crop_size, start_x: start_x + crop_size]
    image_1, image_2 = image[:c], image[c:]
    return image_1, image_2


def random_crop_without_padding(image1: torch.Tensor, image2: torch.Tensor, crop_size: int):
    assert image1.shape == image2.shape
    c, h, w = image1.shape
    concat_image = torch.cat([image1, image2], dim=0)
    start_y = np.random.randint(0, h - crop_size + 1)
    start_x = np.random.randint(0, w - crop_size + 1)
    crop = concat_image[:, start_y: start_y + crop_size, start_x: start_x + crop_size]
    image1, image2 = crop[:c], crop[c:]
    return image1, image2


class Pix2PixDataset(Dataset):
    def __init__(self, root: str, split: str, transformation: torchvision.transforms.Compose, random_crop_size: int = -1):
        assert split in ("train", "val", "test")
        self.transformation = transformation
        self.random_crop_size = random_crop_size

        self._data = self._load_data(root, split)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        image, labels = self._data[index]
        image = Image.fromarray(image)
        labels = Image.fromarray(labels)
        image = self.transformation(image)
        labels = self.transformation(labels)
        if self.random_crop_size > 0:
            image, labels = random_crop_without_padding(
                image1=image, image2=labels, crop_size=self.random_crop_size
            )
        return image, labels

    def _load_data(self, root: str, split: str):
        data = []
        data_dirpath = os.path.join(root, split)
        filepaths = glob(os.path.join(data_dirpath, "*.jpg"))
        for f in tqdm(filepaths, desc="reading data..."):
            image = np.asarray(Image.open(f))
            width = image.shape[1]
            image, labels = image[:, :width//2], image[:, width//2:]
            data.append((image, labels))
        return data


if __name__ == '__main__':
    root = "../data/kaggle/facades/facades"
    split = "test"
    dataset = Pix2PixDataset(root, split)
    print(len(dataset))
    image, labels = dataset[0]
    print(image)
    print(labels)