PROBA_DIFF_TH = 0.3
MIN_FULL_IMAGE_PROB = 0.4

from time import sleep

import pandas as pd
from icecream import ic
from datetime import datetime as dt
import os
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
import pickle
from typing import Union, Dict, List, Tuple
from torchvision.transforms import transforms
from tqdm import tqdm
from transformers import ViTForImageClassification
from config import config
from pytorch_lightning import seed_everything
from torch.nn import functional as F
import numpy as np
import seaborn as sns
from feature_extractor import ViTFeatureExtractor
from main.seg_classification.vit_backbone_to_details import VIT_BACKBONE_DETAILS
from vit_loader.load_vit import load_vit_pretrained
import torch
import pickle

vit_config = config["vit"]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

IMAGENET_VAL_IMAGES_FOLDER_PATH = "/home/amiteshel1/Projects/explainablity-transformer/vit_data/"
GT_VALIDATION_PATH_LABELS = "/home/yuvalas/explainability/data/val ground truth 2012.txt"
# PICKLES_PATH = "/raid/yuvalas/pickles"
seed_everything(config['general']['seed'])
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


def get_pic_value(vit_for_image_classification: ViTForImageClassification,
                  inputs,
                  inputs_scatter,
                  gt_class: int,
                  ) -> Tuple[int, float, float]:
    full_image_probability_by_index = get_probability_and_class_idx_by_index(
        vit_for_image_classification(inputs).logits, index=gt_class)
    saliency_map_probability_by_index = get_probability_and_class_idx_by_index(
        vit_for_image_classification(inputs_scatter).logits, index=gt_class)

    pic_value = calculate_percentage_increase_in_confidence(
        full_image_confidence=full_image_probability_by_index,
        saliency_map_confidence=saliency_map_probability_by_index
    )
    return pic_value, full_image_probability_by_index, saliency_map_probability_by_index


def save_obj_to_disk(path, obj) -> None:
    if type(path) == str and path[-4:] != '.pkl':
        path += '.pkl'

    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def get_gt_classes(path):
    with open(path, 'r') as f:
        gt_classes_list = f.readlines()
    gt_classes_list = [int(record.split()[-1].replace('\n', '')) for record in gt_classes_list]
    return gt_classes_list


def load_obj(path):
    with open(Path(path), 'rb') as f:
        return pickle.load(f)


def get_image(path) -> Image:
    image = Image.open(path)
    return image


resize = transforms.Compose([
    transforms.Resize((config['vit']['img_size'], config['vit']['img_size'])),
    transforms.ToTensor(),
])


def plot_hila_image(image, full_confidence: float, saliancy_confidence: float) -> None:  # [1,3,224,224] or [3,224,224]
    image = image if len(image.shape) == 3 else image.squeeze(0)
    plt.imshow(image.cpu().detach().permute(1, 2, 0))
    plt.title(f"full_confidence: {full_confidence}, saliancy confidence: {saliancy_confidence}")
    plt.axis('off');
    plt.margins(x=0, y=0)
    plt.show(transparent=True);


def plot_image(image) -> None:  # [1,3,224,224] or [3,224,224]
    image = image if len(image.shape) == 3 else image.squeeze(0)
    plt.imshow(image.cpu().detach().permute(1, 2, 0))
    plt.axis('off');
    plt.margins(x=0, y=0)
    plt.show(transparent=True);


def save_image(image, image_idx: int, is_hila: bool) -> None:  # [1,3,224,224] or [3,224,224]
    image = image if len(image.shape) == 3 else image.squeeze(0)
    plt.imshow(image.cpu().detach().permute(1, 2, 0))
    plt.axis('off');
    plt.margins(x=0, y=0)
    # plt.show();
    path = f"/home/yuvalas/explainability/research/plots/our_pic_1_images/{image_idx}_hila.png" if is_hila else f"/home/yuvalas/explainability/research/plots/our_pic_1_images/{image_idx}.png"
    plt.savefig(path, dpi=2000,
                bbox_inches='tight', pad_inches=0, transparent=True)


def save_image_try(image, image_idx: int, is_hila: bool) -> None:  # [1,3,224,224] or [3,224,224]
    image = image if len(image.shape) == 3 else image.squeeze(0)
    image = transforms.ToPILImage()(image)
    image = image.resize((224, 224))
    plt.imshow(transforms.ToTensor()(image).permute(1, 2, 0))
    plt.axis('off');
    # plt.show();
    path = f"/home/yuvalas/explainability/research/plots/our_pic_1_images/{image_idx}_hila.png" if is_hila else f"/home/yuvalas/explainability/research/plots/our_pic_1_images/{image_idx}.png"
    plt.margins(x=0, y=0)
    plt.savefig(path, dpi=1500,
                bbox_inches='tight', pad_inches=0, transparent=True)



def show_mask(mask, model_type='N/A', auc='N/A'):  # [1, 1, 224, 224]
    mask = mask if len(mask.shape) == 3 else mask.squeeze(0)
    _ = plt.imshow(mask.squeeze(0).cpu().detach())
    # plt.title(f'model: {model_type}, auc: {auc}')
    plt.axis('off');
    plt.margins(x=0)
    plt.show()
    return


def scatter_image_by_mask(image, mask):
    return image * mask


def normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def get_probability_by_logits(logits):
    return F.softmax(logits, dim=1)[0]


def calculate_percentage_increase_in_confidence(full_image_confidence: float, saliency_map_confidence: float) -> int:
    """
    Higher is better
    """
    return 1 if full_image_confidence < saliency_map_confidence else 0


def read_image_and_mask_from_pickls_by_path(image_path, mask_path, device):
    masks_listdir = os.listdir(mask_path)[:40000]
    # print(f"Total images: {len(masks_listdir)}")
    for idx in range(len(masks_listdir)):
        pkl_path = Path(mask_path, f"{idx}.pkl")  # pkl are zero-based
        loaded_obj = load_obj(pkl_path)
        image = get_image(Path(image_path, f'ILSVRC2012_val_{str(idx + 1).zfill(8)}.JPEG'))  # images are one-based
        image = image if image.mode == "RGB" else image.convert("RGB")
        image_resized = resize(image).unsqueeze(0)
        yield dict(image_resized=image_resized.to(device),
                   image_mask=loaded_obj["vis"].to(device),
                   auc=loaded_obj["auc"])


def get_probability_and_class_idx_by_index(logits, index: int) -> float:
    probability_distribution = F.softmax(logits[0], dim=-1)
    predicted_probability_by_idx = probability_distribution[index].item()
    return predicted_probability_by_idx


def infer_pic(vit_for_image_classification: ViTForImageClassification,
              images_and_masks,
              gt_classes_list: List[int],
              is_hila: bool = False,
              ) -> List[Union[str, int]]:
    pic_1_images = []

    for image_idx, image_and_mask in tqdm(enumerate(images_and_masks), total=len(gt_classes_list)):
        if image_idx in [7436, 11616, 17721, 16167] or is_hila:
            image, mask = image_and_mask["image_resized"], image_and_mask["image_mask"]  # [1,3,224,224], [1,1,224,224]
            norm_original_image = normalize(image.clone())
            scattered_image = scatter_image_by_mask(image=image, mask=mask)
            scattered_image_to_plot = scattered_image.clone()
            norm_scattered_image = normalize(scattered_image)
            gt_class = image_and_mask["target"] if "target" in image_and_mask.keys() else gt_classes_list[image_idx]
            pic_value, full_image_probability_by_index, saliency_map_probability_by_index, = get_pic_value(
                vit_for_image_classification=vit_for_image_classification,
                inputs=norm_original_image,
                inputs_scatter=norm_scattered_image,
                gt_class=gt_class,
            )
            if not is_hila:
                if pic_value == 1 and full_image_probability_by_index > MIN_FULL_IMAGE_PROB and saliency_map_probability_by_index - full_image_probability_by_index > PROBA_DIFF_TH:
                    print(
                        f"image_idx: {image_idx}, gt_class: {gt_class}, full_image_probability_by_index: {full_image_probability_by_index}, saliency_map_probability_by_index: {saliency_map_probability_by_index}, pic_value: {pic_value}")

                    pic_1_images.append(image_idx)
                    save_image_try(image=image_and_mask["image_resized"], image_idx=image_idx, is_hila=is_hila)
                    # save_image_try(image=scattered_image_to_plot, image_idx=image_idx, is_hila=is_hila)
                    # plot_image(image)
                    # plot_image(scattered_image_to_plot)
                    # show_mask(mask)
            if is_hila:
                print(image_idx, full_image_probability_by_index, saliency_map_probability_by_index)
                save_image_try(image=scattered_image_to_plot, image_idx=image_idx, is_hila=is_hila)

                # plot_hila_image(scattered_image_to_plot, full_confidence=round(full_image_probability_by_index, 5),
                #                 srealiancy_confidence=round(saliency_map_probability_by_index, 5))
                # plot_image(scattered_image_to_plot)
            print(1)
    return pic_1_images


def run_evaluations(pkl_path,
                    exp_name: str,
                    is_base_model: bool,
                    target_or_predicted_model: str,
                    backbone_name: str,
                    imagenet_val_images_folder_path,
                    device):
    print(f"backbone_name: {backbone_name}")
    print(f"is_base_model: {is_base_model}")
    print(f"pkl_path: {pkl_path}")

    NAME = f'{"Base" if is_base_model else "Opt"} Model + {target_or_predicted_model} - {backbone_name}'
    print(NAME)
    images_and_masks = read_image_and_mask_from_pickls_by_path(image_path=imagenet_val_images_folder_path,
                                                               mask_path=pkl_path, device=device)

    pic_values = infer_pic(vit_for_image_classification=vit_for_image_classification,
                           images_and_masks=images_and_masks,
                           gt_classes_list=gt_classes_list)
    print("images with pic value of 1")
    print(pic_values)
    # print(np.where(np.array(pic_values) == 1)[0])
    print('###')


if __name__ == '__main__':
    gt_classes_list = get_gt_classes(GT_VALIDATION_PATH_LABELS)
    target_or_predicted_model = "predicted"
    backbone_name = "google/vit-base-patch16-224"
    HOME_BASE_PATH = VIT_BACKBONE_DETAILS[backbone_name]["experiment_base_path"][target_or_predicted_model]
    OPTIMIZATION_PKL_PATH = Path(HOME_BASE_PATH)
    OPTIMIZATION_PKL_PATH_OPT = Path(OPTIMIZATION_PKL_PATH, "opt_model", "objects_pkl")
    feature_extractor = ViTFeatureExtractor.from_pretrained(backbone_name)
    vit_for_image_classification, _ = load_vit_pretrained(model_name=backbone_name)
    vit_for_image_classification = vit_for_image_classification.to(device)
    images_and_masks = read_image_and_mask_from_pickls_by_path(image_path=IMAGENET_VAL_IMAGES_FOLDER_PATH,
                                                               mask_path=OPTIMIZATION_PKL_PATH_OPT, device=device)

    if len(os.listdir(OPTIMIZATION_PKL_PATH_OPT)) == 50000:
        run_evaluations(pkl_path=OPTIMIZATION_PKL_PATH_OPT,
                        exp_name=HOME_BASE_PATH,
                        is_base_model=False,
                        target_or_predicted_model=target_or_predicted_model,
                        backbone_name=backbone_name,
                        imagenet_val_images_folder_path=IMAGENET_VAL_IMAGES_FOLDER_PATH,
                        device=device)
