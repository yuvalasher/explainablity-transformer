import torch
import os
import random
from typing import Union, List, Dict, Tuple
import pandas as pd
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset
from feature_extractor import ViTFeatureExtractor
from pathlib import WindowsPath, Path
from main.seg_classification.cnns.cnn_utils import convnet_preprocess, convnet_resize_transform
from utils import get_image_from_path
from utils.transformation import resize
from utils.vit_utils import get_image_and_inputs_and_transformed_image
from config import config
from utils.consts import IMAGENET_VAL_GT_CSV_FILE_PATH

seed_everything(config["general"]["seed"])

N_IMAGES_PER_LABEL = 1000


class ImagesDataset(Dataset):
    def __init__(
            self,
            images_path: Union[str, WindowsPath],
            images_name: List[str],
            targets: List[int],
            is_explaniee_convnet: bool,
            is_competitive_method_transforms: bool,
            feature_extractor: ViTFeatureExtractor = None,
    ):
        self.feature_extractor = feature_extractor
        self.images_name = images_name
        self.images_path = images_path
        self.targets = targets
        self.is_competitive_method_transforms = is_competitive_method_transforms
        self.is_explaniee_convnet = is_explaniee_convnet

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, index: int):
        image_name = os.path.basename(self.images_name[index])
        image = get_image_from_path(path=Path(self.images_path, image_name))
        image = image if image.mode == "RGB" else image.convert("RGB")  # Black & White images
        if not self.is_explaniee_convnet:
            inputs, resized_and_normalized_image = get_image_and_inputs_and_transformed_image(
                image=image,
                feature_extractor=self.feature_extractor,
                is_competitive_method_transforms=self.is_competitive_method_transforms,
            )
            image_resized = resize(image)
            inputs = inputs["pixel_values"]
        else:
            inputs = convnet_preprocess(image)
            resized_and_normalized_image = convnet_preprocess(image)
            image_resized = convnet_resize_transform(image)
        target_class = torch.tensor(self.targets[index])

        return dict(
            image_name=image_name,
            pixel_values=inputs,
            resized_and_normalized_image=resized_and_normalized_image,
            image=image_resized,
            target_class=target_class,
        )


class ImageSegDataset(Dataset):
    def __init__(
            self,
            images_path: Union[str, WindowsPath],
            feature_extractor: ViTFeatureExtractor,
            train_n_label_sample: int,
            val_n_label_sample: int,
            is_sampled_train_data_uniformly: bool = True,
            is_sampled_val_data_uniformly: bool = True,
            direct_load: bool = False,
            val_split: float = 0.2,
    ):
        self.feature_extractor = feature_extractor
        self.images_path = images_path
        total_images = len(list(Path(images_path).iterdir()))
        print(f"Total images: {total_images}")
        
        if direct_load and total_images <= 50:  # For small datasets, load directly
            print(f"Using direct load for small dataset with {total_images} images")
            self._load_images_directly(val_split=val_split)
        else:
            # Original sampling method
            train_n_samples = train_n_label_sample * 1000
            val_n_samples = val_n_label_sample * 1000
            datasets = self.sample_train_val_data(
                images_csv_path=IMAGENET_VAL_GT_CSV_FILE_PATH,
                train_n_samples=train_n_samples,
                val_n_samples=val_n_samples,
                is_sampled_train_data_uniformly=is_sampled_train_data_uniformly,
                is_sampled_val_data_uniformly=is_sampled_val_data_uniformly
            )
            self.train_set = datasets["train_set"]
            self.train_gt_classes = datasets["train_gt_classes"]
            self.val_set = datasets["val_set"]
            self.val_gt_classes = datasets["val_gt_classes"]
    
    def _load_images_directly(self, val_split: float = 0.2):
        """
        Load images directly from the directory for small datasets
        """
        # Get all image files from directory
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        all_files = [f for f in os.listdir(self.images_path) 
                    if os.path.isfile(os.path.join(self.images_path, f)) and 
                    os.path.splitext(f)[1].lower() in valid_extensions]
        
        # Get labels from CSV if possible
        try:
            df = pd.read_csv(IMAGENET_VAL_GT_CSV_FILE_PATH)
            targets_dict = {row['img_name']: row['label'] for _, row in df.iterrows()}
            
            # Find labels for our images
            all_images = []
            all_targets = []
            for img_file in all_files:
                if img_file in targets_dict:
                    all_images.append(img_file)
                    all_targets.append(targets_dict[img_file])
                else:
                    # Fallback - assign label 0 if not found
                    all_images.append(img_file)
                    all_targets.append(0)
        except Exception as e:
            print(f"Warning: Could not load labels from CSV: {e}")
            # Fallback - assign all images to class 0
            all_images = all_files
            all_targets = [0] * len(all_files)
        
        # Split into train/val
        num_val = max(1, int(len(all_images) * val_split))
        num_train = len(all_images) - num_val
        
        random.seed(config["general"]["seed"])
        indices = list(range(len(all_images)))
        # random.shuffle(indices)
        
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        
        self.train_set = [all_images[i] for i in train_indices]
        self.train_gt_classes = [all_targets[i] for i in train_indices]
        self.val_set = [all_images[i] for i in val_indices]
        self.val_gt_classes = [all_targets[i] for i in val_indices]
        
        print(f"Loaded {len(self.train_set)} training and {len(self.val_set)} validation images directly")

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
            df = df.query('img_name == @images_name_without_val')
            train_sampled, train_gt_classes = self.sample_uniform(df=df, n_samples=train_n_samples)
        else:
            random.seed(config["general"]["seed"])
            # Handle case where train_n_samples is larger than available samples
            available_samples = len(images_name_without_val)
            actual_n_samples = min(train_n_samples, available_samples)
            if actual_n_samples < train_n_samples:
                print(f"Warning: Requested {train_n_samples} training samples but only {available_samples} available")
            
            train_sampled = random.sample(images_name_without_val, k=actual_n_samples)
            train_gt_classes = df.query('img_name == @train_sampled').label.values.tolist()
        return train_sampled, train_gt_classes

    def sample_val(self,
                   df: pd.DataFrame,
                   is_sampled_val_data_uniformly: bool,
                   images_name: List[str],
                   val_n_samples: int) -> Tuple[List[str], List[int]]:
        if is_sampled_val_data_uniformly:
            val_sampled, val_gt_classes = self.sample_uniform(df=df, n_samples=val_n_samples, is_val=True)
        else:
            random.seed(config["general"]["seed"])
            # Handle case where val_n_samples is larger than available samples
            available_samples = len(images_name)
            actual_n_samples = min(val_n_samples, available_samples)
            if actual_n_samples < val_n_samples:
                print(f"Warning: Requested {val_n_samples} validation samples but only {available_samples} available")
                
            val_sampled = random.sample(images_name, k=actual_n_samples)
            val_gt_classes = df.query('img_name == @val_sampled').label.values.tolist()
        return val_sampled, val_gt_classes

    def sample_uniform(self, df: pd.DataFrame, n_samples: int, is_val=False) -> Tuple[List[str], List[int]]:
        df_sample = pd.DataFrame(columns=['img_name', 'label'])
        
        # Calculate how many samples we need per class
        unique_labels = df.label.unique()
        num_classes = len(unique_labels)
        
        # Adjust samples per class if we have fewer images than expected
        samples_per_class = max(1, int(n_samples / N_IMAGES_PER_LABEL))
        
        for idx_label, val in enumerate(unique_labels):
            # Get all images for this class
            class_images = df.loc[df.label == val]
            
            # If we have fewer images in this class than requested, use all of them
            actual_samples = min(samples_per_class, len(class_images))
            if actual_samples < samples_per_class:
                print(f"Warning: Requested {samples_per_class} samples for class {val} but only {actual_samples} available")
            
            arr_img_name = class_images.sample(actual_samples).img_name.values
            arr_label = class_images.sample(actual_samples).label.values
            
            n_rows = df_sample.shape[0]
            for idx, img_name in enumerate(arr_img_name):
                df_sample.loc[n_rows + idx, 'img_name'] = arr_img_name[idx]
                df_sample.loc[n_rows + idx, 'label'] = arr_label[idx]

        return df_sample['img_name'].values.tolist(), df_sample['label'].values.tolist()

    def sample_random_train_val(self, images_name: List[str], train_n_samples: int, val_n_samples: int):
        random.seed(config["general"]["seed"])
        # Handle case where requested samples exceed available images
        total_samples_needed = train_n_samples + val_n_samples
        available_samples = len(images_name)
        
        if total_samples_needed > available_samples:
            print(f"Warning: Requested {total_samples_needed} total samples but only {available_samples} available")
            # Adjust sizes maintaining the original ratio if possible
            if available_samples > 0:
                ratio = train_n_samples / total_samples_needed
                val_n_samples = min(val_n_samples, max(1, int(available_samples * (1 - ratio))))
                train_n_samples = min(train_n_samples, available_samples - val_n_samples)
            else:
                val_n_samples = 0
                train_n_samples = 0
                
        val_random_sampled = random.sample(images_name, k=val_n_samples) if val_n_samples > 0 else []
        l_without_val = sorted(list(set(images_name) - set(val_random_sampled)))
        random.seed(config["general"]["seed"])
        train_random_sampled = random.sample(l_without_val, k=train_n_samples) if train_n_samples > 0 else []
        return dict(train_set=train_random_sampled, val_set=val_random_sampled)
