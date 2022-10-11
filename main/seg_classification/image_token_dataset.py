import os
from typing import Union

import pandas as pd
from torch.utils.data import Dataset
import torch
from feature_extractor import ViTFeatureExtractor
from pathlib import WindowsPath, Path
from utils import get_image_from_path
from utils.transformation import resize
from vit_utils import get_image_and_inputs_and_transformed_image
from config import config

vit_config = config["vit"]
print(f"TRAIN N_SAMPLES: {vit_config['seg_cls']['train_n_label_sample'] * 1000}")
print(f"VAL N_SAMPLES: {vit_config['seg_cls']['val_n_label_sample'] * 1000}")

# FILE_PATH = "/home/yuvalas/explainability/data/samples_6000_reconstruct.txt"
FILE_PATH = "/home/amiteshel1/Projects/explainablity-transformer-cv/img_name_and_label_no_idx.csv"


class ImageSegDataset(Dataset):
    def __init__(
            self,
            images_path: Union[str, WindowsPath],
            feature_extractor: ViTFeatureExtractor,
            is_val: bool,
    ):
        self.is_val = is_val
        self.feature_extractor = feature_extractor
        self.images_path = images_path
        print(f"Total images: {len(list(Path(images_path).iterdir()))}")
        # with open(FILE_PATH, 'r') as f:
        #     lines = f.readlines()
        img_name_train, img_name_val = self.sample_balance_labels_train_val()

        if (is_val):
            self.images_name = [f"{self.images_path}/{image_name}" for image_name in img_name_val]
        else:
            self.images_name = [f"{self.images_path}/{image_name}" for image_name in img_name_train]
        # if is_val:
        #     self.images_name = list(Path(images_path).iterdir())
        #     n_samples = n_samples if n_samples > 0 else len(self.images_name)
        #     self.images_name = self.images_name[:n_samples]
        print(f"After filter images: {len(self.images_name)}")
        # print(self.images_name)

    def sample_balance_labels_train_val(self):
        df = pd.read_csv(FILE_PATH)
        df_train_sample = pd.DataFrame(columns=['img_name', 'label'])
        df_val_sample = pd.DataFrame(columns=['img_name', 'label'])

        for idx_label, val in enumerate(df.label.unique()):
            arr_img_name = df.loc[df.label == val].sample(vit_config['seg_cls']['val_n_label_sample']).img_name.values
            arr_label = df.loc[df.label == val].sample(vit_config['seg_cls']['val_n_label_sample']).label.values
            n_rows = df_val_sample.shape[0]
            for idx, img_name in enumerate(arr_img_name):
                df_val_sample.loc[n_rows + idx, 'img_name'] = arr_img_name[idx]
                df_val_sample.loc[n_rows + idx, 'label'] = arr_label[idx]

        df = df.drop(index=df_val_sample.index)
        for idx_label, val in enumerate(df.label.unique()):
            arr_img_name = df.loc[df.label == val].sample(vit_config['seg_cls']['train_n_label_sample']).img_name.values
            arr_label = df.loc[df.label == val].sample(vit_config['seg_cls']['train_n_label_sample']).label.values
            n_rows = df_train_sample.shape[0]
            for idx, img_name in enumerate(arr_img_name):
                df_train_sample.loc[n_rows + idx, 'img_name'] = arr_img_name[idx]
                df_train_sample.loc[n_rows + idx, 'label'] = arr_label[idx]

        img_name_train = df_train_sample['img_name'].values.tolist()
        img_name_val = df_val_sample['img_name'].values.tolist()
        return img_name_train, img_name_val

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, index: int):
        image_name = os.path.basename(self.images_name[index])
        image = get_image_from_path(path=Path(self.images_path, image_name))
        image = image if image.mode == "RGB" else image.convert("RGB")  # Black & White images
        inputs, resized_and_normalized_image = get_image_and_inputs_and_transformed_image(
            image=image, feature_extractor=self.feature_extractor
        )
        image_resized = resize(image)

        return dict(
            image_name=image_name,
            pixel_values=inputs["pixel_values"],
            resized_and_normalized_image=resized_and_normalized_image,
            image=image_resized,
        )
