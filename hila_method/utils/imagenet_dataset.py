import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


# IMAGES_FOLDER_PATH = Path('/home/yuvalas/wolf/Transformer-Explainability/baselines/data')
# IMAGES_FOLDER_PATH = Path('/home/yuvalas/explainability/data/run_hila_3000')


# targets = list(range(3000))


class ImageNetDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.listdir = os.listdir(root_dir)
        self.targets = list(range(len(self.listdir)))

    def __len__(self):
        return len(self.listdir)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        image_name = self.listdir[idx]
        image_path = Path(self.root_dir, image_name)
        image = Image.open(image_path)
        image = image if image.mode == "RGB" else image.convert("RGB")  # Black & White images

        if self.transform:
            image = self.transform(image)
        return image, self.targets[idx]


def show_pil_image(image):
    from matplotlib import pyplot as plt
    plt.imshow(image)
    plt.show()