import os

import numpy as np
import pandas as pd
import torchvision.transforms
from torch.utils.data import Dataset
from typing import Dict
from sklearn.model_selection import train_test_split
from PIL import Image


def pad_to_square(image_array: np.ndarray) -> np.ndarray:
    h, w, c = image_array.shape
    size = max(h, w)
    padded_image_array = np.zeros(shape=(size, size, c), dtype=np.uint8)
    start_y = (size - h) // 2
    start_x = (size - w) // 2
    padded_image_array[start_y:start_y + h, start_x: start_x + w] = image_array
    return padded_image_array


class CelebaDataset(Dataset):
    def __init__(self, data_dir: str, is_train: bool, criteria: Dict[str, int],
                 transforms: torchvision.transforms.Compose = None):
        attributes_df = pd.read_csv(os.path.join(data_dir, "attributes.csv"))
        ids = sorted(attributes_df["id"].tolist())
        train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=42)
        ids = set(train_ids if is_train else test_ids)
        attributes_df = attributes_df[attributes_df["id"].isin(ids)]
        for column, value in criteria.items():
            attributes_df = attributes_df[attributes_df[column] == value]
        assert attributes_df.shape[0] > 0
        self.image_filepaths = [
            os.path.join(data_dir, "images", filename) for filename in attributes_df["filename"].tolist()
        ]

        self.transforms = transforms or torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        image = Image.open(self.image_filepaths[index])
        image = Image.fromarray(pad_to_square(np.asarray(image)))
        return self.transforms(image)

    def __len__(self):
        return len(self.image_filepaths)


if __name__ == '__main__':
    dataset = CelebaDataset(data_dir="datasets/celeba", is_train=True, criteria={"No_Beard": 1})
    print(dataset[0])