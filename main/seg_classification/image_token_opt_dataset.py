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


class ImageSegOptDataset(Dataset):
    def __init__(
            self,
            image_path: Union[str, WindowsPath],
            feature_extractor: ViTFeatureExtractor,
            target: int
    ):
        self.feature_extractor = feature_extractor
        self.image_path = image_path
        self.target = target

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        image = get_image_from_path(path=self.image_path)
        image = image if image.mode == "RGB" else image.convert("RGB")  # Black & White images
        inputs, resized_and_normalized_image = get_image_and_inputs_and_transformed_image(
            image=image, feature_extractor=self.feature_extractor
        )
        image_resized = resize(image)

        return dict(
            image_name=self.image_path.split('/')[-1].split('.')[0],
            pixel_values=inputs["pixel_values"],
            resized_and_normalized_image=resized_and_normalized_image,
            image=image_resized,
            target_class=torch.tensor(self.target),
        )
