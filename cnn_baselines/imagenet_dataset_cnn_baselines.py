import os
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.consts import GT_VALIDATION_PATH_LABELS

class ImageNetDataset(Dataset):

    def __init__(self,
                 root_dir,
                 list_of_images_names: List[int],
                 transform,
                 resize_transform,
                 ):
        self.root_dir = root_dir
        self.transform = transform
        self.resize_transform = resize_transform
        self.listdir = sorted(os.listdir(root_dir))
        self.targets = get_gt_classes(path=GT_VALIDATION_PATH_LABELS)

        n_samples = len(self.listdir)
        self.images_name = [f"ILSVRC2012_val_{str(image_idx + 1).zfill(8)}.JPEG" for image_idx in list_of_images_names]
        self.targets = [self.targets[image_idx] for image_idx in list_of_images_names]
        self.targets = self.targets[:n_samples]
        print(f"After filter images: {len(self.images_name)}")

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        image_name = self.images_name[idx]
        image_path = Path(self.root_dir, image_name)
        image = Image.open(image_path)
        image = image if image.mode == "RGB" else image.convert("RGB")  # Black & White images
        resized_image = self.resize_transform(image)
        if self.transform:
            image = self.transform(image)
        return image, self.targets[idx], resized_image


def show_pil_image(image):
    from matplotlib import pyplot as plt
    plt.imshow(image)
    plt.show()


def get_gt_classes(path):
    with open(path, 'r') as f:
        gt_classes_list = f.readlines()
    gt_classes_list = [int(record.split()[-1].replace('\n', '')) for record in gt_classes_list]
    return gt_classes_list


resize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))

])
