import os
import random
from typing import Union, List
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
IMAGENET_VAL_GT_FILE_PATH = "/home/amiteshel1/Projects/explainablity-transformer-cv/val_ground_truth_2012.csv"


class ImageSegDataset(Dataset):
    def __init__(
            self,
            images_path: Union[str, WindowsPath],
            feature_extractor: ViTFeatureExtractor,
            is_val: bool,
            is_sampled_data_uniformly: bool = True,
    ):
        self.is_val = is_val
        self.feature_extractor = feature_extractor
        self.images_path = images_path
        print(f"Total images: {len(list(Path(images_path).iterdir()))}")

        img_name_train, img_name_val = self.sample_balance_labels_train_val(
            is_uniform_sampling=is_sampled_data_uniformly)
        if (is_val):
            self.images_name = [f"{self.images_path}/{image_name}" for image_name in img_name_val]
        else:
            self.images_name = [f"{self.images_path}/{image_name}" for image_name in img_name_train]
        # else:
        #     train_n_samples = vit_config['seg_cls']['val_n_label_sample'] * 1000
        #     val_n_samples = vit_config['seg_cls']['val_n_label_sample'] * 1000
        #     self.images_name = sorted(os.listdir(images_path))
        #     randomly_sampled_train_val_dict = self.sample_random_train_val(images_name=self.images_name,
        #                                                                    train_n_samples=train_n_samples,
        #                                                                    val_n_samples=val_n_samples)
        #     self.images_name = randomly_sampled_train_val_dict["val_set"] if is_val else \
        #     randomly_sampled_train_val_dict["train_set"]

        ## sample first n images
        # train_n_samples = vit_config['seg_cls']['train_n_label_sample'] * 1000
        # val_n_samples = vit_config['seg_cls']['val_n_label_sample'] * 1000
        # n_samples = train_n_samples if not is_val else val_n_samples
        # self.images_name = list(Path(images_path).iterdir())
        # n_samples = n_samples if n_samples > 0 else len(self.images_name)
        # self.images_name = self.images_name[:n_samples]
        # print(is_val, len(self.images_name), n_samples)
        print(f"After filter images: {len(self.images_name)}")

    def sample_random_train_val(self, images_name: List[str], train_n_samples: int, val_n_samples: int):
        random.seed(config["general"]["seed"])
        val_random_sampled = random.sample(images_name, k=val_n_samples)
        l_without_val = sorted(list(set(images_name) - set(val_random_sampled)))
        random.seed(config["general"]["seed"])
        train_random_sampled = random.sample(l_without_val, k=train_n_samples)
        return dict(train_set=train_random_sampled, val_set=val_random_sampled)

    def sample_balance_labels_train_val(self, is_uniform_sampling: bool = True):
        df = pd.read_csv(IMAGENET_VAL_GT_FILE_PATH)
        train_n_samples = vit_config['seg_cls']['train_n_label_sample']
        val_n_samples = vit_config['seg_cls']['val_n_label_sample']

        df_train_sample = pd.DataFrame(columns=['img_name', 'label'])
        df_val_sample = pd.DataFrame(columns=['img_name', 'label'])

        for idx_label, val in enumerate(df.label.unique()):
            arr_img_name = df.loc[df.label == val].sample(val_n_samples, random_state=config["general"]["seed"]).img_name.values
            arr_label = df.loc[df.label == val].sample(val_n_samples, random_state=config["general"]["seed"]).label.values
            n_rows = df_val_sample.shape[0]
            for idx, img_name in enumerate(arr_img_name):
                df_val_sample.loc[n_rows + idx, 'img_name'] = arr_img_name[idx]
                df_val_sample.loc[n_rows + idx, 'label'] = arr_label[idx]
        img_name_val = df_val_sample['img_name'].values.tolist()
        df = df.drop(index=df_val_sample.index)

        if is_uniform_sampling:
            for idx_label, val in enumerate(df.label.unique()):
                arr_img_name = df.loc[df.label == val].sample(train_n_samples).img_name.values
                arr_label = df.loc[df.label == val].sample(train_n_samples).label.values
                n_rows = df_train_sample.shape[0]
                for idx, img_name in enumerate(arr_img_name):
                    df_train_sample.loc[n_rows + idx, 'img_name'] = arr_img_name[idx]
                    df_train_sample.loc[n_rows + idx, 'label'] = arr_label[idx]
            img_name_train = df_train_sample['img_name'].values.tolist()
        else:
            random.seed(config["general"]["seed"])
            train_random_sampled = random.sample(df.img_name.values.tolist(), k=train_n_samples * 1000)
            img_name_train = train_random_sampled
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
