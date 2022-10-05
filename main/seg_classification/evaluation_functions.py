from datetime import datetime as dt
import os
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
import pickle
from typing import Union, Dict, List
from torchvision.transforms import transforms
from tqdm import tqdm
from transformers import ViTForImageClassification
from config import config
from pytorch_lightning import seed_everything
from torch.nn import functional as F
import numpy as np
from evaluation.perturbation_tests.seg_cls_perturbation_tests import eval_perturbation_test
from utils import get_gt_classes
from utils.consts import GT_VALIDATION_PATH_LABELS, IMAGENET_VAL_IMAGES_FOLDER_PATH
from vit_loader.load_vit import load_vit_pretrained
import torch
from enum import Enum

seed_everything(config['general']['seed'])
device = torch.device(type='cuda', index=config["general"]["gpu_index"])


class VisClass(Enum):
    TOP = "TOP"
    TARGET = "TARGET"


class PerturbationType(Enum):
    POS = "POS"
    NEG = "NEG"


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


def plot_image(image) -> None:  # [1,3,224,224] or [3,224,224]
    image = image if len(image.shape) == 3 else image.squeeze(0)
    plt.imshow(image.cpu().detach().permute(1, 2, 0))
    plt.show();


def show_mask(mask):  # [1, 1, 224, 224]
    mask = mask if len(mask.shape) == 3 else mask.squeeze(0)
    _ = plt.imshow(mask.squeeze(0).cpu().detach())
    plt.show()


def get_probability_by_logits(logits):
    return F.softmax(logits, dim=1)[0]


def calculate_avg_drop_percentage(full_image_confidence: float, saliency_map_confidence: float) -> float:
    """
    Lower is better
    """
    return max(0, full_image_confidence - saliency_map_confidence) / full_image_confidence


def calculate_percentage_increase_in_confidence(full_image_confidence: float, saliency_map_confidence: float) -> float:
    """
    Higher is better
    """
    return 1 if full_image_confidence < saliency_map_confidence else 0


def get_probability_and_class_idx_by_index(logits, index: int) -> Dict[str, Union[int, float]]:
    probability_distribution = F.softmax(logits[0], dim=-1)
    if index == 0:
        predicted_class_by_idx = torch.argmax(probability_distribution).item()
        predicted_probability_by_idx = torch.max(probability_distribution).item()
    else:
        predicted_class_by_idx = None
        predicted_probability_by_idx = probability_distribution[index].item()
    return dict(predicted_class_by_idx=predicted_class_by_idx,
                predicted_probability_by_idx=predicted_probability_by_idx)


def run_evaluation_metrics(vit_for_image_classification: ViTForImageClassification,
                           inputs,
                           inputs_scatter,
                           gt_class: int,
                           is_compared_by_target: bool):
    if is_compared_by_target:
        full_image_probability_and_class_idx_by_index = get_probability_and_class_idx_by_index(
            vit_for_image_classification(inputs).logits, index=gt_class)
        saliency_map_probability_and_class_idx_by_index = get_probability_and_class_idx_by_index(
            vit_for_image_classification(inputs_scatter).logits,
            index=gt_class)

    else:
        full_image_probability_and_class_idx_by_index = get_probability_and_class_idx_by_index(
            vit_for_image_classification(inputs).logits, index=0)

        saliency_map_probability_and_class_idx_by_index = get_probability_and_class_idx_by_index(
            vit_for_image_classification(inputs_scatter).logits,
            index=full_image_probability_and_class_idx_by_index["predicted_class_by_idx"])

    avg_drop_percentage = calculate_avg_drop_percentage(
        full_image_confidence=full_image_probability_and_class_idx_by_index["predicted_probability_by_idx"],
        saliency_map_confidence=saliency_map_probability_and_class_idx_by_index["predicted_probability_by_idx"])

    percentage_increase_in_confidence_indicators = calculate_percentage_increase_in_confidence(
        full_image_confidence=full_image_probability_and_class_idx_by_index["predicted_probability_by_idx"],
        saliency_map_confidence=saliency_map_probability_and_class_idx_by_index["predicted_probability_by_idx"])
    return dict(avg_drop_percentage=avg_drop_percentage,
                percentage_increase_in_confidence_indicators=percentage_increase_in_confidence_indicators)


def infer_adp_and_pic(vit_for_image_classification: ViTForImageClassification,
                      images_and_masks: List[Dict],
                      gt_classes_list: List[int],
                      ADP_PIC_config: Dict[str, bool],
                      ):
    adp_values, pic_values = [], []
    is_compared_by_target: bool = ADP_PIC_config["IS_COMPARED_BY_TARGET"]
    is_clamp_between_0_to_1: bool = ADP_PIC_config["IS_CLAMP_BETWEEN_0_TO_1"]

    for image_idx, image_and_mask in enumerate(images_and_masks):
        image, mask = image_and_mask["image_resized"], image_and_mask["image_mask"]  # [1,3,224,224], [1,1,224,224]
        # plot_image(image)
        # show_mask(mask)
        norm_original_image = normalize(image.clone())
        # plot_image(norm_original_image)
        norm_mask = normalize_mask_values(mask=mask, clamp_between_0_to_1=is_clamp_between_0_to_1)
        # show_mask(norm_mask)
        scattered_image = scatter_image_by_mask(image=image, mask=norm_mask)  # TODO
        # plot_image(scattered_image)
        norm_scattered_image = normalize(scattered_image)
        # plot_image(norm_scattered_image)
        metrics = run_evaluation_metrics(vit_for_image_classification=vit_for_image_classification,
                                         inputs=norm_original_image,
                                         inputs_scatter=norm_scattered_image,
                                         gt_class=gt_classes_list[image_idx],
                                         is_compared_by_target=is_compared_by_target)
        adp_values.append(metrics["avg_drop_percentage"])
        pic_values.append(metrics["percentage_increase_in_confidence_indicators"])

    percentage_increase_in_confidence = 100 * np.mean(pic_values)
    averaged_drop_percentage = 100 * np.mean(adp_values)
    return dict(percentage_increase_in_confidence=percentage_increase_in_confidence,
                averaged_drop_percentage=averaged_drop_percentage)


def normalize_mask_values(mask, clamp_between_0_to_1: bool = False):
    if clamp_between_0_to_1:
        norm_mask = torch.clamp(mask, min=0, max=1)
    else:
        norm_mask = (mask - mask.min()) / (mask.max() - mask.min())
    return norm_mask


def scatter_image_by_mask(image, mask):
    return image * mask


def normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def read_image_and_mask_from_pickls_by_path(image_path, mask_path, device) -> List[Dict]:
    objects = []

    masks_listdir = os.listdir(mask_path)
    for idx in range(len(masks_listdir)):
        pkl_path = Path(mask_path, f"{idx}.pkl")  # pkl are zero-based
        loaded_obj = load_obj(pkl_path)
        image = get_image(Path(image_path, f'ILSVRC2012_val_{str(idx + 1).zfill(8)}.JPEG'))  # images are one-based
        image = image if image.mode == "RGB" else image.convert("RGB")
        image_resized = resize(image).unsqueeze(0)
        objects.append(
            dict(image_resized=image_resized.to(device), image_mask=loaded_obj["vis"].to(device)))
    return objects


def infer_perturbation_tests(images_and_masks: List[Dict], vit_for_image_classification,
                             perturbation_config: Dict[str, PerturbationType], gt_classes_list: List[int]):
    """
    :param config: contains the configuration of the perturbation test:
        * neg: True / False
        * vis_class: TARGET / TOP (predicted top-1)
    """
    aucs = []
    vis_class = perturbation_config["vis_class"].name
    perturbation_type = perturbation_config["perturbation_type"].name
    for image_idx, image_and_mask in enumerate(images_and_masks):
        image, mask = image_and_mask["image_resized"], image_and_mask["image_mask"]  # [1,3,224,224], [1,1,224,224]
        outputs = [{'image_resized': image, 'image_mask': mask}]
        auc = eval_perturbation_test(experiment_dir=Path(""), model=vit_for_image_classification, outputs=outputs,
                                     perturbation_type=perturbation_type, vis_class=vis_class,
                                     target_class=gt_classes_list[image_idx])
        aucs.append(auc)
    # print(aucs)
    return np.mean(aucs)


if __name__ == '__main__':
    OPTIMIZATION_PKL_PATH = "/home/yuvalas/explainability/research/experiments/seg_cls/ft_50000/opt_objects"

    vit_for_image_classification, _ = load_vit_pretrained(model_name=config["vit"]["model_name"])
    vit_for_image_classification = vit_for_image_classification.to(device)
    images_and_masks = read_image_and_mask_from_pickls_by_path(image_path=IMAGENET_VAL_IMAGES_FOLDER_PATH,
                                                               mask_path=OPTIMIZATION_PKL_PATH, device=device)
    gt_classes_list = get_gt_classes(GT_VALIDATION_PATH_LABELS)

    # Perturbation tests
    perturbation_config = {'vis_class': VisClass.TOP, 'perturbation_type': PerturbationType.POS}
    start_time = dt.now()
    auc = infer_perturbation_tests(images_and_masks=images_and_masks[:100],
                                   vit_for_image_classification=vit_for_image_classification,
                                   perturbation_config=perturbation_config, gt_classes_list=gt_classes_list)
    print(f"timing: {(dt.now() - start_time).total_seconds()}")
    print(f'Mean AUC: {auc} for {perturbation_config["vis_class"]}; {perturbation_config["perturbation_type"]}. data: /home/yuvalas/explainability/research/experiments/seg_cls/ft_50000/opt_objects')

    """
    # ADP & PIC metrics
    ADP_PIC_config = {'IS_CLAMP_BETWEEN_0_TO_1': False, 'IS_COMPARED_BY_TARGET': True}
    print(
        f'Evaluation Params: IS_COMPARED_BY_TARGET: {ADP_PIC_config["IS_COMPARED_BY_TARGET"]}, IS_CLAMP_BETWEEN_0_TO_1: {ADP_PIC_config["IS_CLAMP_BETWEEN_0_TO_1"]}')
    evaluation_metrics = infer_adp_and_pic(vit_for_image_classification=vit_for_image_classification,
                                           images_and_masks=images_and_masks,
                                           gt_classes_list=gt_classes_list, ADP_PIC_config=ADP_PIC_config)
    print(
        f'PIC (% Increase in Confidence - Higher is better): {round(evaluation_metrics["percentage_increase_in_confidence"], 4)}%; ADP (Average Drop % - Lower is better): {round(evaluation_metrics["averaged_drop_percentage"], 4)}%')
    """

    """
    assert calculate_avg_drop_percentage(full_image_confidence=0.8, saliency_map_confidence=0.4) == 0.5
    assert calculate_percentage_increase_in_confidence(full_image_confidence=0.8, saliency_map_confidence=0.4) == 0
    assert calculate_percentage_increase_in_confidence(full_image_confidence=0.4, saliency_map_confidence=0.8) == 1
    """
