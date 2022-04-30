import os
import torchvision
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, root: str, transforms: torchvision.transforms.Compose = None):
        super().__init__()
        self.transforms = transforms or torchvision.transforms.ToTensor()

        self.images = self._load_images(root)

    def __getitem__(self, index):
        image = self.images[index]
        tensor = self.transforms(image)
        return tensor

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _load_images(root: str):
        filepaths = glob(os.path.join(root, "*"))
        images = []
        for f in tqdm(filepaths, desc="loading images"):
            images.append(Image.open(f))
        return images