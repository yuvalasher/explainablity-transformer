import torch
from icecream import ic
import os
import random
from typing import Union, List, Dict, Tuple
import pandas as pd
from torch.utils.data import Dataset
from feature_extractor import ViTFeatureExtractor
from pathlib import WindowsPath, Path
from utils import get_image_from_path
from utils.transformation import resize
from vit_utils import get_image_and_inputs_and_transformed_image
from config import config

vit_config = config["vit"]
print(f"TRAIN N_SAMPLES: {vit_config['seg_cls']['train_n_label_sample'] * 1000}")
print(f"VAL N_SAMPLES: {vit_config['seg_cls']['val_n_label_sample'] * 1000}")

IMAGENET_TEST_GT_BY_VIT_FILE_PATH = "/home/amiteshel1/Projects/explainablity-transformer-cv/imagenet_test_gt_by_vit.csv"
IMAGENET_VAL_GT_CSV_FILE_PATH = "/home/amiteshel1/Projects/explainablity-transformer-cv/val_ground_truth_2012.csv"
# GT_VALIDATION_PATH_LABELS = "/home/yuvalas/explainability/data/val ground truth 2012.txt"
N_IMAGES_PER_LABEL = 1000


class ImagesDataset(Dataset):
    def __init__(
            self,
            images_path: Union[str, WindowsPath],
            feature_extractor: ViTFeatureExtractor,
            images_name: List[str],
            targets: List[int]
    ):
        self.feature_extractor = feature_extractor
        self.images_name = images_name
        self.images_path = images_path
        self.targets = targets

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, index: int):
        image_name = os.path.basename(self.images_name[index])
        image = get_image_from_path(path=Path(self.images_path, image_name))
        image = image if image.mode == "RGB" else image.convert("RGB")  # Black & White images
        inputs, resized_and_normalized_image = get_image_and_inputs_and_transformed_image(
            image=image, feature_extractor=self.feature_extractor, is_wolf_transforms=vit_config["is_wolf_transforms"]
        )
        image_resized = resize(image)
        target_class = torch.tensor(self.targets[index])

        return dict(
            image_name=image_name,
            pixel_values=inputs["pixel_values"],
            resized_and_normalized_image=resized_and_normalized_image,
            image=image_resized,
            target_class=target_class,
        )


class ImageSegDataset(Dataset):
    def __init__(
            self,
            images_path: Union[str, WindowsPath],
            feature_extractor: ViTFeatureExtractor,
            is_sampled_train_data_uniformly: bool = True,
            is_sampled_val_data_uniformly: bool = True,
    ):
        self.feature_extractor = feature_extractor
        self.images_path = images_path
        print(f"Total images: {len(list(Path(images_path).iterdir()))}")
        train_n_samples = vit_config['seg_cls']['train_n_label_sample'] * 1000
        val_n_samples = vit_config['seg_cls']['val_n_label_sample'] * 1000

        datasets = self.sample_train_val_data(images_csv_path=IMAGENET_VAL_GT_CSV_FILE_PATH,
                                              train_n_samples=train_n_samples,
                                              val_n_samples=val_n_samples,
                                              is_sampled_train_data_uniformly=is_sampled_train_data_uniformly,
                                              is_sampled_val_data_uniformly=is_sampled_val_data_uniformly)
        self.train_set = datasets["train_set"]
        self.train_gt_classes = datasets["train_gt_classes"]
        self.val_set = datasets["val_set"]
        self.val_gt_classes = datasets["val_gt_classes"]

    def sample_train_val_data(self,
                              images_csv_path: str,
                              is_sampled_train_data_uniformly: bool,
                              is_sampled_val_data_uniformly: bool,
                              train_n_samples: int,
                              val_n_samples: int) -> Dict[str, List[str]]:
        df = pd.read_csv(images_csv_path)
        images_name = df.img_name.values.tolist()
        val_set, val_gt_classes = self.sample_val(df=df,
                                                  images_name=images_name,
                                                  is_sampled_val_data_uniformly=is_sampled_val_data_uniformly,
                                                  val_n_samples=val_n_samples)
        train_set, train_gt_classes = self.sample_train(df=df,
                                                        images_name=images_name,
                                                        is_sampled_train_data_uniformly=is_sampled_train_data_uniformly,
                                                        val_samples=val_set,
                                                        train_n_samples=train_n_samples)

        return dict(train_set=train_set, train_gt_classes=train_gt_classes, val_set=val_set,
                    val_gt_classes=val_gt_classes)

    def sample_train(self,
                     df: pd.DataFrame,
                     is_sampled_train_data_uniformly: bool,
                     val_samples: List[str],
                     images_name: List[str],
                     train_n_samples: int) -> Tuple[List[str], List[int]]:
        images_name_without_val = sorted(list(set(images_name) - set(val_samples)))
        if is_sampled_train_data_uniformly:
            train_sampled, train_gt_classes = self.sample_uniform(df=df, n_samples=train_n_samples)
        else:
            random.seed(config["general"]["seed"])
            train_sampled = random.sample(images_name_without_val, k=train_n_samples)
            train_gt_classes = df.query('img_name == @images_name_without_val').label.values.tolist()
        return train_sampled, train_gt_classes

    def sample_val(self,
                   df: pd.DataFrame,
                   is_sampled_val_data_uniformly: bool,
                   images_name: List[str],
                   val_n_samples: int) -> Tuple[List[str], List[int]]:
        if is_sampled_val_data_uniformly:
            val_sampled, val_gt_classes = self.sample_uniform(df=df, n_samples=val_n_samples)
        else:
            random.seed(config["general"]["seed"])
            val_sampled = random.sample(images_name, k=val_n_samples)
            val_gt_classes = df.query('img_name == @images_name_without_val').label.values.tolist()
        return val_sampled, val_gt_classes

    def sample_uniform(self, df: pd.DataFrame, n_samples: int) -> Tuple[List[str], List[int]]:
        df_sample = pd.DataFrame(columns=['img_name', 'label'])
        for idx_label, val in enumerate(df.label.unique()):
            arr_img_name = df.loc[df.label == val].sample(int(n_samples / N_IMAGES_PER_LABEL)).img_name.values
            arr_label = df.loc[df.label == val].sample(int(n_samples / N_IMAGES_PER_LABEL)).label.values
            n_rows = df_sample.shape[0]
            for idx, img_name in enumerate(arr_img_name):
                df_sample.loc[n_rows + idx, 'img_name'] = arr_img_name[idx]
                df_sample.loc[n_rows + idx, 'label'] = arr_label[idx]
        return df_sample['img_name'].values.tolist(), df_sample['label'].values.tolist()

    def sample_random_train_val(self, images_name: List[str], train_n_samples: int, val_n_samples: int):
        random.seed(config["general"]["seed"])
        val_random_sampled = random.sample(images_name, k=val_n_samples)
        l_without_val = sorted(list(set(images_name) - set(val_random_sampled)))
        random.seed(config["general"]["seed"])
        train_random_sampled = random.sample(l_without_val, k=train_n_samples)
        return dict(train_set=train_random_sampled, val_set=val_random_sampled)
