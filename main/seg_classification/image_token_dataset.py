import os
from typing import Union

from torch.utils.data import Dataset
import torch

from feature_extractor import ViTFeatureExtractor
from pathlib import WindowsPath, Path

from utils import get_image_from_path
from vit_utils import get_image_and_inputs_and_transformed_image


class ImageSegDataset(Dataset):
    def __init__(
            self,
            images_path: Union[str, WindowsPath],
            feature_extractor: ViTFeatureExtractor,
    ):
        self.feature_extractor = feature_extractor
        self.images_path = images_path
        self.images_name = list(Path(images_path).iterdir())

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, index: int):
        image_name = os.path.basename(self.images_name[index])
        image = get_image_from_path(path=Path(self.images_path, image_name))
        # labels = data_row[LABEL_COLUMNS]
        inputs, original_transformed_image = get_image_and_inputs_and_transformed_image(image=image,
                                                                                        feature_extractor=self.feature_extractor)

        return dict(
            image_name=image_name,
            pixel_values=inputs['pixel_values'],
            original_transformed_image=original_transformed_image
        )
