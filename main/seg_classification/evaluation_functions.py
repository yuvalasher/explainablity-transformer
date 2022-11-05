import pandas as pd
from icecream import ic
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
import seaborn as sns
from evaluation.perturbation_tests.seg_cls_perturbation_tests import eval_perturbation_test
# from utils.utils_functions import get_gt_classes
# from utils.consts import GT_VALIDATION_PATH_LABELS, IMAGENET_VAL_IMAGES_FOLDER_PATH
# from main.seg_classification.image_classification_with_token_classification_model import prediction_loss
from main.seg_classification.vit_backbone_to_details import VIT_BACKBONE_DETAILS
from vit_loader.load_vit import load_vit_pretrained
import torch
from enum import Enum

IMAGENET_VAL_IMAGES_FOLDER_PATH = "/home/amiteshel1/Projects/explainablity-transformer/vit_data/"
GT_VALIDATION_PATH_LABELS = "/home/yuvalas/explainability/data/val ground truth 2012.txt"
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
    print(f"Total images: {len(masks_listdir)}")
    for idx in range(len(masks_listdir)):
        pkl_path = Path(mask_path, f"{idx}.pkl")  # pkl are zero-based
        loaded_obj = load_obj(pkl_path)
        image = get_image(Path(image_path, f'ILSVRC2012_val_{str(idx + 1).zfill(8)}.JPEG'))  # images are one-based
        image = image if image.mode == "RGB" else image.convert("RGB")
        image_resized = resize(image).unsqueeze(0)
        yield dict(image_resized=image_resized.to(device),
                   image_mask=loaded_obj["vis"].to(device),
                   auc=loaded_obj["auc"])


def infer_perturbation_tests(images_and_masks, vit_for_image_classification,
                             perturbation_config: Dict[str, Union[PerturbationType, bool]], gt_classes_list: List[int]):
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
        outputs = [{'image_resized': image, 'image_mask': mask, 'target_class': gt_classes_list[image_idx]}]
        auc_perturbation, auc_deletion_insertion = eval_perturbation_test(experiment_dir=Path(""),
                                                                          model=vit_for_image_classification,
                                                                          outputs=outputs,
                                                                          perturbation_type=perturbation_type,
                                                                          is_calculate_deletion_insertion=is_calculate_deletion_insertion)
        # print(1)
        aucs_perturbation.append(auc_perturbation)
        aucs_auc_deletion_insertion.append(auc_deletion_insertion)
    return np.mean(aucs_perturbation), np.mean(aucs_auc_deletion_insertion)


# def calculate_logitis_prob_loss_per_image(images_and_masks, vit_for_image_classification):
#     q_arr = np.arange(1, 4.5, 0.5)
#     flag = True
#     df_a = pd.DataFrame(columns=[f'{q_val}' for q_val in q_arr])
#     df_b = pd.DataFrame(columns=[f'{q_val}' for q_val in q_arr])
#     df_c = pd.DataFrame(columns=[f'{q_val}' for q_val in q_arr])
#     for image_idx, image_and_mask in enumerate(images_and_masks):
#
#         pred_logitis_arr, pred_probs_arr, pred_loss_arr = [], [], []
#         for q in q_arr:
#             image, mask = image_and_mask["image_resized"], image_and_mask["image_mask"]  # [1,3,224,224], [1,1,224,224]
#             # original:
#             if flag:
#                 plot_image(image)
#                 show_mask(mask)
#                 flag = False
#             _norm_img = normalize(image)
#             inputs_org = {'pixel_values': _norm_img}
#             out_org = vit_for_image_classification(**inputs_org)
#             org_probs = torch.softmax(out_org.logits, dim=1)
#             gt_idx = out_org.logits.argmax().item()
#
#             mask = mask ** q
#             if image_idx == 0:
#                 show_mask(mask)
#
#             # pred
#             data = image * mask
#             _norm_data = normalize(data)
#             inputs = {'pixel_values': _norm_data}
#             out = vit_for_image_classification(**inputs)
#             pred_probs = torch.softmax(out.logits, dim=1)
#
#             # calculate_loss
#             pred_loss = prediction_loss(output=out.logits, target=out_org.logits)
#             pred_logitis_arr.append(out.logits[0][gt_idx].cpu().item())
#             pred_probs_arr.append(pred_probs[0][gt_idx].cpu().item())
#             pred_loss_arr.append(pred_loss.cpu().item())
#             df_a.loc[image_idx, f'{q}'] = out.logits[0][gt_idx].cpu().item()
#             df_b.loc[image_idx, f'{q}'] = pred_probs[0][gt_idx].cpu().item()
#             df_c.loc[image_idx, f'{q}'] = pred_loss.cpu().item()
#         plot_metric_vs_power(q_arr, pred_logitis_arr, 'logitis', image_idx)
#         plot_metric_vs_power(q_arr, pred_probs_arr, 'proba', image_idx)
#         plot_metric_vs_power(q_arr, pred_loss_arr, 'pred_loss_ce', image_idx)
#         # sns.boxplot(data=df_a)
#         flag = True
#     sns.boxplot(data=df_a)
#     plt.title('logits')
#     plt.show()
#     sns.boxplot(data=df_b)
#     plt.title('proba')
#     plt.show()
#     sns.boxplot(data=df_c)
#     plt.title('ce_loss')
#     plt.show()
#     return
#
"""
def compare_opt_vs_base_per_image(images_and_masks__dict, vit_for_image_classification, perturbation_config,
                                  gt_classes_list):
    vis_class = perturbation_config["vis_class"].name
    perturbation_type = perturbation_config["perturbation_type"].name

    for image_idx, image_and_mask in enumerate(images_and_masks__dict):

        keys = [k for k in images_and_masks__dict.keys()]
        for idx, x in enumerate(zip(images_and_masks__dict[keys[0]], images_and_masks__dict[keys[1]])):
            if idx >= 4:
                img_opt, mask_opt = x[0]['image_resized'], x[0]['image_mask']  # [1,3,224,224], [1,1,224,224]
                img_base, mask_base = x[1]['image_resized'], x[1]['image_mask']  # [1,3,224,224], [1,1,224,224]

                outputs = [{'image_resized': img_opt, 'image_mask': mask_opt}]
                auc_opt = eval_perturbation_test(experiment_dir=Path(""), model=vit_for_image_classification,
                                                 outputs=outputs,
                                                 perturbation_type=perturbation_type, vis_class=vis_class,
                                                 target_class=gt_classes_list[image_idx], model_type='opt',
                                                 auc_score=x[0]['auc'])

                outputs = [{'image_resized': img_base, 'image_mask': mask_base}]
                auc_base = eval_perturbation_test(experiment_dir=Path(""), model=vit_for_image_classification,
                                                  outputs=outputs,
                                                  perturbation_type=perturbation_type, vis_class=vis_class,
                                                  target_class=gt_classes_list[image_idx], model_type='base',
                                                  auc_score=x[1]['auc'])

            # original:
            # if flag:
            #     plot_image(image)
            #     show_mask(mask)
            #     flag = False
            # _norm_img = normalize(image)
            # inputs_org = {'pixel_values': _norm_img}
            # out_org = vit_for_image_classification(**inputs_org)
            # org_probs = torch.softmax(out_org.logits, dim=1)
            # gt_idx = out_org.logits.argmax().item()
            #
            # if image_idx == 0:
            #     show_mask(mask)
            #
            # # pred
            # data = image * mask
            # _norm_data = normalize(data)
            # inputs = {'pixel_values': _norm_data}
            # out = vit_for_image_classification(**inputs)
            # pred_probs = torch.softmax(out.logits, dim=1)
            #
            # # calculate_loss
            # pred_loss = prediction_loss(output=out.logits, target=out_org.logits)

    return


def plot_metric_vs_power(q_arr, metric_a, metric_name, image_idx):
    plt.plot(q_arr, metric_a)
    plt.grid()
    plt.title(f'img_idx = {image_idx} - metric = {metric_name}')
    plt.show()
    return

"""


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
            vit_for_image_classification(inputs_scatter).logits, index=gt_class)

    else:  # Predicted Top 1
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

    avg_change_percentage = calculate_average_change_percentage(
        full_image_confidence=full_image_probability_and_class_idx_by_index["predicted_probability_by_idx"],
        saliency_map_confidence=saliency_map_probability_and_class_idx_by_index["predicted_probability_by_idx"])

    return dict(avg_drop_percentage=avg_drop_percentage,
                percentage_increase_in_confidence_indicators=percentage_increase_in_confidence_indicators,
                avg_change_percentage=avg_change_percentage)


def infer_adp_pic_acp(vit_for_image_classification: ViTForImageClassification,
                      images_and_masks,
                      gt_classes_list: List[int],
                      ):
    adp_values_predicted, pic_values_predicted, acp_values_predicted = [], [], []
    adp_values_target, pic_values_target, acp_values_target = [], [], []

    for image_idx, image_and_mask in tqdm(enumerate(images_and_masks), total=len(gt_classes_list)):
        image, mask = image_and_mask["image_resized"], image_and_mask["image_mask"]  # [1,3,224,224], [1,1,224,224]
        # plot_image(image)
        # show_mask(mask)
        norm_original_image = normalize(image.clone())
        # plot_image(norm_original_image)
        # norm_mask = normalize_mask_values(mask=mask, clamp_between_0_to_1=is_clamp_between_0_to_1) # masks are already normalized
        # show_mask(norm_mask)
        scattered_image = scatter_image_by_mask(image=image, mask=mask)  # TODO
        # plot_image(scattered_image)
        norm_scattered_image = normalize(scattered_image)
        # plot_image(norm_scattered_image)
        metrics_predicted = run_evaluation_metrics(vit_for_image_classification=vit_for_image_classification,
                                                   inputs=norm_original_image,
                                                   inputs_scatter=norm_scattered_image,
                                                   gt_class=gt_classes_list[image_idx],
                                                   is_compared_by_target=False)
        adp_values_predicted.append(metrics_predicted["avg_drop_percentage"])
        pic_values_predicted.append(metrics_predicted["percentage_increase_in_confidence_indicators"])
        acp_values_predicted.append(metrics_predicted["avg_change_percentage"])

        metrics_target = run_evaluation_metrics(vit_for_image_classification=vit_for_image_classification,
                                                inputs=norm_original_image,
                                                inputs_scatter=norm_scattered_image,
                                                gt_class=gt_classes_list[image_idx],
                                                is_compared_by_target=True)
        adp_values_target.append(metrics_target["avg_drop_percentage"])
        pic_values_target.append(metrics_target["percentage_increase_in_confidence_indicators"])
        acp_values_target.append(metrics_target["avg_change_percentage"])

    ic(len(adp_values_predicted), len(pic_values_predicted), len(acp_values_predicted))
    ic(len(adp_values_target), len(pic_values_target), len(acp_values_target))
    averaged_drop_percentage_predicted = 100 * np.mean(adp_values_predicted)
    percentage_increase_in_confidence_predicted = 100 * np.mean(pic_values_predicted)
    averaged_change_percentage_predicted = 100 * np.mean(acp_values_predicted)

    averaged_drop_percentage_target = 100 * np.mean(adp_values_target)
    percentage_increase_in_confidence_target = 100 * np.mean(pic_values_target)
    averaged_change_percentage_target = 100 * np.mean(acp_values_target)

    return dict(percentage_increase_in_confidence_predicted=percentage_increase_in_confidence_predicted,
                averaged_drop_percentage_predicted=averaged_drop_percentage_predicted,
                averaged_change_percentage_predicted=averaged_change_percentage_predicted,
                percentage_increase_in_confidence_target=percentage_increase_in_confidence_target,
                averaged_drop_percentage_target=averaged_drop_percentage_target,
                averaged_change_percentage_target=averaged_change_percentage_target
                )


if __name__ == '__main__':
    HOME_BASE_PATH = "/home/yuvalas/explainability/research/experiments/seg_cls"
    EXP_NAME = "direct_opt_ckpt_27_model_google_vit-base-patch16-224_train_uni_True_val_unif_True_activation_sigmoid__norm_by_max_p_False_pred_1_mask_l_bce_50__train_n_samples_6000_lr_0.002_mlp_classifier_True__bs_32"

    OPTIMIZATION_PKL_PATH = Path(HOME_BASE_PATH, EXP_NAME)
    OPTIMIZATION_PKL_PATH_BASE = Path(OPTIMIZATION_PKL_PATH, "base_model", "objects_pkl")
    OPTIMIZATION_PKL_PATH_OPT = Path(OPTIMIZATION_PKL_PATH, "opt_model", "objects_pkl")

    vit_for_image_classification, _ = load_vit_pretrained(model_name=config["vit"]["model_name"])
    vit_for_image_classification = vit_for_image_classification.to(device)
    gt_classes_list = get_gt_classes(GT_VALIDATION_PATH_LABELS)
    start_time = dt.now()
    # images_and_masks= read_image_and_mask_from_pickls_by_path(image_path=IMAGENET_VAL_IMAGES_FOLDER_PATH,
    #                                                            mask_path=OPTIMIZATION_PKL_PATH, device=device)

    """
    # ADP & PIC metrics
    evaluation_metrics = infer_adp_pic_acp(vit_for_image_classification=vit_for_image_classification,
                                           images_and_masks=images_and_masks,
                                           gt_classes_list=gt_classes_list)
    ic(evaluation_metrics)
    print(
        f'Predicted - PIC (% Increase in Confidence - Higher is better): {round(evaluation_metrics["percentage_increase_in_confidence_predicted"], 4)}%; ADP (Average Drop % - Lower is better): {round(evaluation_metrics["averaged_drop_percentage_predicted"], 4)}%; ACP (% Average Change Percentage - Higher is better): {round(evaluation_metrics["averaged_change_percentage_predicted"], 4)}%;')

    """

    # Perturbation tests
    # # TODO - Do it with loop of perturbation_type
    perturbation_config = {'perturbation_type': PerturbationType.POS,
                           "is_calculate_deletion_insertion": True}

    print('************************************************************************************')
    PKL_PATH = OPTIMIZATION_PKL_PATH_OPT
    print(
        f'Perturbation tests {perturbation_config["perturbation_type"]}. data: {PKL_PATH}')
    images_and_masks_opt = read_image_and_mask_from_pickls_by_path(image_path=IMAGENET_VAL_IMAGES_FOLDER_PATH,
                                                                   mask_path=PKL_PATH, device=device)

    auc_perturbation_opt, auc_deletion_insertion_opt = infer_perturbation_tests(images_and_masks=images_and_masks_opt,
                                                                        vit_for_image_classification=vit_for_image_classification,
                                                                        perturbation_config=perturbation_config,
                                                                        gt_classes_list=gt_classes_list)

    print(
        f'Opt Model; Perturbation tests {perturbation_config["perturbation_type"]}. data: {PKL_PATH}')
    print(
        f'Mean Perturbation AUC: {auc_perturbation_opt}; Mean Deletion-Insertion AUC: {auc_deletion_insertion_opt} for {perturbation_config["perturbation_type"]}. data: {PKL_PATH}')

    PERTURBATION_DELETION_INSERTION_MAPPING = {PerturbationType.POS: "Deletion", PerturbationType.NEG: "Insertion"}
    for target_or_predicted_model in ["predicted", "target"]:
        HOME_BASE_PATH = VIT_BACKBONE_DETAILS[vit_config["model_name"]]["experiment_base_path"][target_or_predicted_model]
        model_type = HOME_BASE_PATH.split("model_")[1].split("_train")[0]
        OPTIMIZATION_PKL_PATH = Path(HOME_BASE_PATH)
        OPTIMIZATION_PKL_PATH_BASE = Path(OPTIMIZATION_PKL_PATH, "base_model", "objects_pkl")
        OPTIMIZATION_PKL_PATH_OPT = Path(OPTIMIZATION_PKL_PATH, "opt_model", "objects_pkl")

        feature_extractor = ViTFeatureExtractor.from_pretrained(vit_config["model_name"])
        if vit_config["model_name"] in ["google/vit-base-patch16-224"]:
            vit_for_image_classification, _ = load_vit_pretrained(
                model_name=vit_config["model_name"])
        else:
            vit_for_image_classification = ViTForImageClassification.from_pretrained(vit_config["model_name"])

    auc_perturbation_base, auc_deletion_insertion_base = infer_perturbation_tests(images_and_masks=images_and_masks_base,
                                                                        vit_for_image_classification=vit_for_image_classification,
                                                                        perturbation_config=perturbation_config,
                                                                        gt_classes_list=gt_classes_list)
    print(f"timing: {(dt.now() - start_time).total_seconds()}")
    print(
        f'Mean Perturbation AUC: {auc_perturbation_base}; Mean Deletion-Insertion AUC: {auc_deletion_insertion_base} for {perturbation_config["perturbation_type"]}. data: {PKL_PATH}')

    """
     images_and_masks = [images_and_masks[i] for i in [1, 2, 4, 7, 10, 12, 13, 15, 18, 19, 20, 22, 24, 27]]
    for i in range(len(images_and_masks)):
        plot_image(images_and_masks[i]["image_resized"])
        show_mask(images_and_masks[i]["image_mask"])
    print(1)
    auc = infer_perturbation_tests(images_and_masks=images_and_masks,
                                   vit_for_image_classification=vit_for_image_classification,
                                   perturbation_config=perturbation_config, gt_classes_list=gt_classes_list)
    """
