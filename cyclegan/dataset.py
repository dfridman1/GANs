import os

import numpy as np
import torchvision
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, root: str, transforms: torchvision.transforms.Compose = None, image_size: int = -1):
        super().__init__()
        self.transforms = transforms or torchvision.transforms.ToTensor()
        self.image_size = image_size

        self.images = self._load_images(root)

    def __getitem__(self, index):
        image = self.images[index]
        tensor = self.transforms(image)
        return tensor

    def __len__(self):
        return len(self.images)

    def _load_images(self, root: str):
        filepaths = glob(os.path.join(root, "*"))
        images = []
        for f in tqdm(filepaths, desc="loading images"):
            image = Image.open(f)
            if self._validate(image):
                if self.image_size > 0:
                    image = image.resize((self.image_size, self.image_size))
                images.append(image)
        return images

    def _validate(self, image: Image) -> bool:
        image_array = np.asarray(image)
        return image_array.ndim == 3 and image_array.shape[-1] == 3
