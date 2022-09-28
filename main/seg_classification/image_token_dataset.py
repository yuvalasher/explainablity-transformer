import os
from typing import Union, List

from torch.utils.data import Dataset
import torch

from feature_extractor import ViTFeatureExtractor
from pathlib import WindowsPath, Path

from utils import get_image_from_path
from utils.transformation import resize
from vit_utils import get_image_and_inputs_and_transformed_image
from config import config

vit_config = config["vit"]
print(f"TRAIN N_SAMPLES: {vit_config['seg_cls']['train_n_samples']}")
print(f"VAL N_SAMPLES: {vit_config['seg_cls']['val_n_samples']}")


class ImageSegDataset(Dataset):
    def __init__(
            self,
            images_path: Union[str, WindowsPath],
            feature_extractor: ViTFeatureExtractor,
            n_samples: int,
    ):
        self.feature_extractor = feature_extractor
        self.images_path = images_path
        print(f"Total images: {len(list(Path(images_path).iterdir()))}")
        self.images_name = list(Path(images_path).iterdir())
        n_samples = n_samples if n_samples > 0 else len(self.images_name)
        self.images_name = self.images_name[:n_samples]
        print(f"After filter images: {len(self.images_name)}")
        # print(self.images_name)

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, index: int):
        image_name = os.path.basename(self.images_name[index])
        image = get_image_from_path(path=Path(self.images_path, image_name))
        image = image if image.mode == "RGB" else image.convert("RGB")  # Black & White images
        inputs, original_transformed_image = get_image_and_inputs_and_transformed_image(
            image=image, feature_extractor=self.feature_extractor
        )
        image_resized = resize(image)

        return dict(
            image_name=image_name,
            pixel_values=inputs["pixel_values"],
            original_transformed_image=original_transformed_image,
            image=image_resized,
        )
