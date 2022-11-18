from time import sleep
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
from evaluation.perturbation_tests.seg_cls_perturbation_tests import eval_perturbation_test
from feature_extractor import ViTFeatureExtractor
from main.seg_classification.vit_backbone_to_details import VIT_BACKBONE_DETAILS
from vit_loader.load_vit import load_vit_pretrained
import torch
from enum import Enum
import pickle
from utils.consts import IMAGENET_VAL_IMAGES_FOLDER_PATH, GT_VALIDATION_PATH_LABELS

vit_config = config["vit"]

seed_everything(config['general']['seed'])
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


def get_gt_classes(path):
    with open(path, 'r') as f:
        gt_classes_list = f.readlines()
    gt_classes_list = [int(record.split()[-1].replace('\n', '')) for record in gt_classes_list]
    return gt_classes_list


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


def show_mask(mask, model_type='N/A', auc='N/A'):  # [1, 1, 224, 224]
    mask = mask if len(mask.shape) == 3 else mask.squeeze(0)
    _ = plt.imshow(mask.squeeze(0).cpu().detach())
    plt.title(f'model: {model_type}, auc: {auc}')
    plt.show()
    return


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


def get_probability_by_logits(logits):
    return F.softmax(logits, dim=1)[0]


def calculate_average_change_percentage(full_image_confidence: float, saliency_map_confidence: float) -> float:
    """
    Higher is better
    """
    return (saliency_map_confidence - full_image_confidence) / full_image_confidence


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


def read_image_and_mask_from_pickls_by_path(image_path, mask_path, device):
    masks_listdir = os.listdir(mask_path)
    for idx in range(len(masks_listdir)):
        pkl_path = Path(mask_path, f"{idx}.pkl")  # pkl are zero-based
        loaded_obj = load_obj(pkl_path)
        image = get_image(Path(image_path, f'ILSVRC2012_val_{str(idx + 1).zfill(8)}.JPEG'))  # images are one-based
        image = image if image.mode == "RGB" else image.convert("RGB")
        image_resized = resize(image).unsqueeze(0)
        yield dict(image_resized=image_resized.to(device),
                   image_mask=loaded_obj["vis"].to(device),
                   auc=loaded_obj["auc"])


def infer_perturbation_tests(images_and_masks,
                             vit_for_image_classification,
                             perturbation_config: Dict[str, Union[PerturbationType, bool]],
                             gt_classes_list: List[int],
                             ) -> Tuple[List[float], List[float]]:
    """
    :param config: contains the configuration of the perturbation test:
        * neg: True / False
    """
    aucs_perturbation = []
    aucs_auc_deletion_insertion = []
    perturbation_type = perturbation_config["perturbation_type"].name
    is_calculate_deletion_insertion = perturbation_config["is_calculate_deletion_insertion"]
    for image_idx, image_and_mask in tqdm(enumerate(images_and_masks)):
        image, mask = image_and_mask["image_resized"], image_and_mask["image_mask"]  # [1,3,224,224], [1,1,224,224]
        outputs = [
            {'image_resized': image, 'image_mask': mask,
             'target_class': torch.tensor([gt_classes_list[image_idx]])}]
        auc_perturbation, auc_deletion_insertion = eval_perturbation_test(experiment_dir=Path(""),
                                                                          model=vit_for_image_classification,
                                                                          outputs=outputs,
                                                                          perturbation_type=perturbation_type,
                                                                          is_calculate_deletion_insertion=is_calculate_deletion_insertion)
        aucs_perturbation.append(auc_perturbation)
        aucs_auc_deletion_insertion.append(auc_deletion_insertion)
    return aucs_perturbation, aucs_auc_deletion_insertion


def get_probability_and_class_idx_by_index(logits, index: int) -> float:
    probability_distribution = F.softmax(logits[0], dim=-1)
    predicted_probability_by_idx = probability_distribution[index].item()
    return predicted_probability_by_idx


def run_evaluation_metrics(vit_for_image_classification: ViTForImageClassification,
                           inputs,
                           inputs_scatter,
                           gt_class: int,
                           ):
    full_image_probability_by_index = get_probability_and_class_idx_by_index(
        vit_for_image_classification(inputs).logits, index=gt_class)
    saliency_map_probability_by_index = get_probability_and_class_idx_by_index(
        vit_for_image_classification(inputs_scatter).logits, index=gt_class)

    avg_drop_percentage = calculate_avg_drop_percentage(
        full_image_confidence=full_image_probability_by_index,
        saliency_map_confidence=saliency_map_probability_by_index)

    percentage_increase_in_confidence_indicators = calculate_percentage_increase_in_confidence(
        full_image_confidence=full_image_probability_by_index,
        saliency_map_confidence=saliency_map_probability_by_index
    )

    avg_change_percentage = calculate_average_change_percentage(
        full_image_confidence=full_image_probability_by_index,
        saliency_map_confidence=saliency_map_probability_by_index
    )

    return dict(avg_drop_percentage=avg_drop_percentage,
                percentage_increase_in_confidence_indicators=percentage_increase_in_confidence_indicators,
                avg_change_percentage=avg_change_percentage)


def infer_adp_pic_acp(vit_for_image_classification: ViTForImageClassification,
                      images_and_masks,
                      gt_classes_list: List[int],
                      ):
    adp_values, pic_values, acp_values = [], [], []

    for image_idx, image_and_mask in tqdm(enumerate(images_and_masks), total=len(gt_classes_list)):
        image, mask = image_and_mask["image_resized"], image_and_mask["image_mask"]  # [1,3,224,224], [1,1,224,224]
        norm_original_image = normalize(image.clone())
        scattered_image = scatter_image_by_mask(image=image, mask=mask)
        norm_scattered_image = normalize(scattered_image)
        metrics = run_evaluation_metrics(vit_for_image_classification=vit_for_image_classification,
                                         inputs=norm_original_image,
                                         inputs_scatter=norm_scattered_image,
                                         gt_class=gt_classes_list[image_idx],
                                         )
        adp_values.append(metrics["avg_drop_percentage"])
        pic_values.append(metrics["percentage_increase_in_confidence_indicators"])
        acp_values.append(metrics["avg_change_percentage"])

    averaged_drop_percentage = 100 * np.mean(adp_values)
    percentage_increase_in_confidence = 100 * np.mean(pic_values)
    averaged_change_percentage = 100 * np.mean(acp_values)

    return dict(percentage_increase_in_confidence=percentage_increase_in_confidence,
                averaged_drop_percentage=averaged_drop_percentage,
                averaged_change_percentage=averaged_change_percentage,
                )


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

    # ADP & PIC metrics
    evaluation_metrics = infer_adp_pic_acp(vit_for_image_classification=vit_for_image_classification,
                                           images_and_masks=images_and_masks,
                                           gt_classes_list=gt_classes_list)
    print(
        f'PIC (% Increase in Confidence - Higher is better): {round(evaluation_metrics["percentage_increase_in_confidence"], 4)}%; ADP (Average Drop % - Lower is better): {round(evaluation_metrics["averaged_drop_percentage"], 4)}%; ACP (% Average Change Percentage - Higher is better): {round(evaluation_metrics["averaged_change_percentage"], 4)}%;')

    # Perturbation + Deletion & Insertion tests
    for perturbation_type in [PerturbationType.POS, PerturbationType.NEG]:
        images_and_masks = read_image_and_mask_from_pickls_by_path(image_path=imagenet_val_images_folder_path,
                                                                   mask_path=pkl_path, device=device)

        perturbation_config = {'perturbation_type': perturbation_type,
                               "is_calculate_deletion_insertion": True}

        print(
            f'Perturbation tests - {perturbation_config["perturbation_type"].name}')

        auc_perturbation_list, auc_deletion_insertion_list = infer_perturbation_tests(
            images_and_masks=images_and_masks,
            vit_for_image_classification=vit_for_image_classification,
            perturbation_config=perturbation_config,
            gt_classes_list=gt_classes_list)
        auc_perturbation, auc_deletion_insertion = np.mean(auc_perturbation_list), np.mean(auc_deletion_insertion_list)

        print(
            f'{"Base" if is_base_model else "Opt"} + {target_or_predicted_model} Model; Perturbation tests {perturbation_config["perturbation_type"].name}, {PERTURBATION_DELETION_INSERTION_MAPPING[perturbation_config["perturbation_type"]]} test. pkl_path: {pkl_path}')
        print(
            f'Mean {perturbation_type} Perturbation AUC: {auc_perturbation}; Mean {PERTURBATION_DELETION_INSERTION_MAPPING[perturbation_config["perturbation_type"]]} AUC: {auc_deletion_insertion}')
        print('************************************************************************************')


if __name__ == '__main__':
    PERTURBATION_DELETION_INSERTION_MAPPING = {PerturbationType.POS: "Deletion", PerturbationType.NEG: "Insertion"}
    gt_classes_list = get_gt_classes(GT_VALIDATION_PATH_LABELS)

    for backbone_name in VIT_BACKBONE_DETAILS.keys():
        for target_or_predicted_model in ["predicted", "target"]:
            HOME_BASE_PATH = VIT_BACKBONE_DETAILS[backbone_name]["experiment_base_path"][target_or_predicted_model]
            OPTIMIZATION_PKL_PATH = Path(HOME_BASE_PATH)
            OPTIMIZATION_PKL_PATH_BASE = Path(OPTIMIZATION_PKL_PATH, "base_model", "objects_pkl")
            OPTIMIZATION_PKL_PATH_OPT = Path(OPTIMIZATION_PKL_PATH, "opt_model", "objects_pkl")
            try:
                feature_extractor = ViTFeatureExtractor.from_pretrained(backbone_name)
                if backbone_name in ["google/vit-base-patch16-224"]:
                    vit_for_image_classification, _ = load_vit_pretrained(
                        model_name=backbone_name)
                else:
                    vit_for_image_classification = ViTForImageClassification.from_pretrained(backbone_name)
            except Exception as e:
                print(e)
                sleep(60)
                feature_extractor = ViTFeatureExtractor.from_pretrained(backbone_name)
                if backbone_name in ["google/vit-base-patch16-224"]:
                    vit_for_image_classification, _ = load_vit_pretrained(
                        model_name=backbone_name)
                else:
                    vit_for_image_classification = ViTForImageClassification.from_pretrained(backbone_name)
            vit_for_image_classification = vit_for_image_classification.to(device)

            if len(os.listdir(OPTIMIZATION_PKL_PATH_BASE)) == 50000:
                run_evaluations(pkl_path=OPTIMIZATION_PKL_PATH_BASE,
                                exp_name=HOME_BASE_PATH,
                                is_base_model=True,
                                target_or_predicted_model=target_or_predicted_model,
                                backbone_name=backbone_name,
                                imagenet_val_images_folder_path=IMAGENET_VAL_IMAGES_FOLDER_PATH,
                                device=device)

            if len(os.listdir(OPTIMIZATION_PKL_PATH_OPT)) == 50000:
                run_evaluations(pkl_path=OPTIMIZATION_PKL_PATH_OPT,
                                exp_name=HOME_BASE_PATH,
                                is_base_model=False,
                                target_or_predicted_model=target_or_predicted_model,
                                backbone_name=backbone_name,
                                imagenet_val_images_folder_path=IMAGENET_VAL_IMAGES_FOLDER_PATH,
                                device=device)
