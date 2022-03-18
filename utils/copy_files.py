import shutil
import random
from pathlib import Path
import torch
from torchvision.datasets import ImageNet
from torchvision import transforms
from utils.consts import DGX_IMAGES_FOLDER_PATH, DATA_PATH, EVALUATION_FOLDER_PATH
from config import config
from vit_utils import save_text_to_file

vit_config = config['vit']

NUM_PICTURES: int = 1000

if __name__ == '__main__':
    pictures_to_copy = random.sample(range(0, 50000), NUM_PICTURES)
    transform = transforms.Compose([
        transforms.Resize((vit_config['img_size'], vit_config['img_size'])),
        transforms.ToTensor(),
    ])
    save_text_to_file(path=EVALUATION_FOLDER_PATH, file_name='images_to_test', text=str(sorted(pictures_to_copy)))
    val_imagenet_ds = ImageNet(str(DATA_PATH), split='val', transform=transform)
    imagenet_ds = torch.utils.data.Subset(val_imagenet_ds, pictures_to_copy)
    for (image, target), image_idx in zip(imagenet_ds, pictures_to_copy):
        picture_name = f'{str(image_idx).zfill(8)}.JPEG'
        pic = transforms.ToPILImage()(image)
        pic.save(Path(DGX_IMAGES_FOLDER_PATH, picture_name))
