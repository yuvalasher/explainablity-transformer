import os
from typing import Union, List

from torch.utils.data import Dataset
import torch

from feature_extractor import ViTFeatureExtractor
from pathlib import WindowsPath, Path

from main.seg_classification.cnns.cnn_utils import convnet_resize_center_crop_transform, convnet_preprocess
from utils import get_image_from_path
from utils.transformation import resize
from vit_utils import get_image_and_inputs_and_transformed_image
from config import config

vit_config = config["vit"]


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
        if self.feature_extractor is not None:
            inputs, resized_and_normalized_image = get_image_and_inputs_and_transformed_image(
                image=image, feature_extractor=self.feature_extractor,
                is_competitive_method_transforms=vit_config["is_competitive_method_transforms"]
            )
            image_resized = resize(image)
            inputs = inputs["pixel_values"]
        else:
            inputs = convnet_preprocess(image)
            resized_and_normalized_image = convnet_preprocess(image)
            image_resized = convnet_resize_center_crop_transform(image)

        return dict(
            image_name=self.image_path.split('/')[-1].split('.')[0],
            pixel_values=inputs,
            resized_and_normalized_image=resized_and_normalized_image,
            image=image_resized,
            target_class=torch.tensor(self.target),
        )
