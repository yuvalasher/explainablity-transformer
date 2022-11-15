import json
import random
from copy import deepcopy

import cv2
import pandas as pd
from matplotlib import pyplot as plt
import os

from torchvision import transforms

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers import ViTForImageClassification

from config import config
import torch

from feature_extractor import ViTFeatureExtractor
from main.seg_classification.evaluation_functions import get_image
from main.seg_classification.vit_backbone_to_details import VIT_BACKBONE_DETAILS
from utils import save_obj_to_disk
from utils.consts import IMAGENET_VAL_IMAGES_FOLDER_PATH
from utils.transformation import resize
import datetime
from datetime import datetime as dt

device = torch.device(type='cuda', index=config["general"]["gpu_index"])
from tqdm import tqdm
from icecream import ic
from pathlib import Path
import pickle
import numpy as np
from collections import Counter

from evaluation.perturbation_tests.seg_cls_perturbation_tests import run_perturbation_test_opt, eval_perturbation_test
from vit_loader.load_vit import load_vit_pretrained
from tabulate import tabulate

vit_config = config["vit"]
BEST_AUC_VALUE = 6


def normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def load_obj(path: str):
    with open(Path(path), 'rb') as f:
        return pickle.load(f)


def get_precentage_counter(c):
    return sorted([(i, str(round(count / sum(c.values()) * 100.0, 3)) + '%') for i, count in c.most_common()])


def calculate_mean_auc(n_samples: int, path):
    aucs = []
    print(f"{n_samples} samples")
    for image_idx in tqdm(range(n_samples)):
        image_path = Path(path, f"{str(image_idx)}.pkl")
        loaded_obj = load_obj(image_path)
        aucs.append(loaded_obj['auc'])

    # print(f'AUCS: {aucs}')
    print(f"{len(aucs)} samples")
    print(f"Mean AUC: {np.mean(aucs)}")
    # df, df_stats = calculate_count__and_prec_auc(aucs)
    # print(tabulate(df_stats, headers='keys'))
    # plot_perturbations_vs_num_of_images(df)

    # counter = Counter(aucs)
    # print(sorted(counter.items()))
    # print(get_precentage_counter(counter))


def plot_perturbations_vs_num_of_images(df):
    plt.plot(range(len(df)), df.auc_value.expanding().mean().values)
    plt.title('Perturbations as function of num_of_images')
    plt.xlabel('Num of images')
    plt.ylabel('Perturbations ')
    plt.grid()
    plt.show()


def calculate_count__and_prec_auc(aucs):
    aucs_np = np.array(aucs).reshape(-1, 1)
    df = pd.DataFrame(data=aucs_np, columns=['auc_value'])
    prec_auc = df.auc_value.value_counts() / len(df)
    value_auc = df.auc_value.value_counts()
    df_stats = pd.DataFrame(data=[prec_auc, value_auc]).T
    df_stats.columns = ['count', 'percentage']
    return df, df_stats


def statistics_expected_run_time(path, start_time):
    n_samples_already_run = len(os.listdir(path))
    avg_seconds_per_image = (dt.now() - start_time).total_seconds() / n_samples_already_run
    expected_run_time_hours = (avg_seconds_per_image * 50000) / 3600
    expected_run_time_days = expected_run_time_hours / 24
    expected_datetime = start_time + datetime.timedelta(hours=expected_run_time_hours)
    print(
        f"N_samples: {n_samples_already_run}; Avg. seconds per image: {avg_seconds_per_image}; Expected run time (days): {expected_run_time_days}; Data: {expected_datetime}")


def show_mask(mask, idx, auc):  # [1, 1, 224, 224]
    mask = mask if len(mask.shape) == 3 else mask.squeeze(0)
    _ = plt.imshow(mask.squeeze(0).cpu().detach())
    plt.title(f'{idx} auc = {auc}')
    # plt.savefig(f"/home/amiteshel1/Projects/explainablity-transformer-cv/amit_png_del/mask_{idx}.png")
    plt.show()


def save_mask(mask, image_idx):
    mask = mask if len(mask.shape) == 3 else mask.squeeze(0)
    plt.axis('off')
    plt.imshow(mask.squeeze(0).cpu().detach())
    path = f"/home/yuvalas/explainability/research/plots/our_masks/{image_idx}_mask.png"
    plt.margins(x=0, y=0)
    plt.savefig(path, dpi=1500, bbox_inches='tight', pad_inches=0, transparent=True)


def plot_image(image, idx, auc) -> None:  # [1,3,224,224] or [3,224,224]
    image = image if len(image.shape) == 3 else image.squeeze(0)
    plt.imshow(image.cpu().detach().permute(1, 2, 0))
    plt.title(f'idx = {idx} auc = {auc}')
    # plt.savefig(f"/home/amiteshel1/Projects/explainablity-transformer-cv/amit_png_del/image_{idx}.png")
    plt.show()


def plot_visualizations_and_images(path_to_exp_pickles: str, vit_for_image_classification):
    """
    Can calculate mean aucs from pickels
    """
    aucs = []
    for idx in tqdm(range(100)):
        loaded_obj = load_obj(
            Path(path_to_exp_pickles, f'{str(idx)}.pkl'))
        image = get_image(Path(IMAGENET_VAL_IMAGES_FOLDER_PATH,
                               f'ILSVRC2012_val_{str(idx + 1).zfill(8)}.JPEG'))  # images are one-based
        image = image if image.mode == "RGB" else image.convert("RGB")
        image_resized = resize(image).unsqueeze(0)
        outputs = [
            {"image_resized": image_resized.to(device), "image_mask": loaded_obj["vis"].to(device)}]

        auc = eval_perturbation_test(experiment_dir=Path(""), model=vit_for_image_classification, outputs=outputs)
        aucs.append(auc)
        # plot_image(image_resized, idx, auc)
        # show_mask(loaded_obj["vis"], idx, auc)
    df, df_stats = calculate_count__and_prec_auc(aucs)
    print(tabulate(df_stats, headers='keys'))
    plot_perturbations_vs_num_of_images(df)
    print(f'AUC: {round(np.mean(aucs), 4)}% for {len(aucs)} records')
    print(aucs)


def check_all_images_in_pickles(path):
    for idx in tqdm(range(len(os.listdir(path)))):
        loaded_obj = load_obj(Path(path, f'{str(idx)}.pkl'))
        loaded_image = loaded_obj["original_image"].clone()

        image = get_image(Path(IMAGENET_VAL_IMAGES_FOLDER_PATH,
                               f'ILSVRC2012_val_{str(idx + 1).zfill(8)}.JPEG'))  # images are one-based
        image = image if image.mode == "RGB" else image.convert("RGB")
        image_resized = resize(image).unsqueeze(0)
        if not torch.equal(image_resized.to(loaded_image.device), loaded_image):
            print(idx)

def find_perturbation_interesting_images_one_stage(model_pickles_path, path_dir):
    images_idx_by_auc_model = {5: [], 10: [], 15: [], 20: [], 25: [], 30: [], 35: [], 40: [], 45: []}
    # n_images = len(os.listdir(base_model_pickels_path))
    n_images = 50000
    random.seed(config["general"]["seed"])
    image_indices = range(n_images)
    for idx in tqdm(image_indices):
        loaded_obj = load_obj(Path(model_pickles_path, f'{str(idx)}.pkl'))
        auc = loaded_obj["auc"]
        if auc in images_idx_by_auc_model.keys():
            images_idx_by_auc_model[auc].append(idx)
    save_obj_to_disk(path=path_dir, obj=images_idx_by_auc_model)



def find_perturbation_interesting_images_stage_a_stage_b(base_model_pickles_path, opt_model_pickles_path, path_dir):
    images_idx_by_auc_diff_base_opt_model = {5: [], 10: [], 15: [], 20: [], 25: [], 30: [], 35: [], 40: [], 45: [],
                                             50: [], 60: []}
    # n_images = len(os.listdir(base_model_pickels_path))
    n_images = 50000
    random.seed(config["general"]["seed"])
    # image_indices = sorted(random.sample(range(1, n_images), n_imagesn_images)
    # print(sorted(rands))
    image_indices = range(n_images)
    for idx in tqdm(image_indices):
        base_model_loaded_obj = load_obj(Path(base_model_pickles_path, f'{str(idx)}.pkl'))
        opt_model_loaded_obj = load_obj(Path(opt_model_pickles_path, f'{str(idx)}.pkl'))
        base_model_auc = base_model_loaded_obj["auc"]
        opt_model_auc = opt_model_loaded_obj["auc"]
        base_opt_model_diff_auc = int(base_model_auc) - int(opt_model_auc)
        if base_opt_model_diff_auc in images_idx_by_auc_diff_base_opt_model.keys():
            images_idx_by_auc_diff_base_opt_model[base_opt_model_diff_auc].append(idx)
    save_obj_to_disk(path=path_dir, obj=images_idx_by_auc_diff_base_opt_model)


def find_perturbation_interesting_images_models_a_b_hila(images_idx_by_auc_diff_base_opt_model, hila_auc_by_img_idx,
                                                         base_model_pickles_path, opt_model_pickles_path):
    interesting_indices_to_look = []
    for key, images_indices in tqdm(images_idx_by_auc_diff_base_opt_model.items()):
        for idx in images_indices:
            base_model_loaded_obj = load_obj(Path(base_model_pickles_path, f'{str(idx)}.pkl'))
            opt_model_loaded_obj = load_obj(Path(opt_model_pickles_path, f'{str(idx)}.pkl'))
            base_model_auc = base_model_loaded_obj["auc"]
            opt_model_auc = opt_model_loaded_obj["auc"]
            base_opt_model_diff_auc = int(base_model_auc) - int(opt_model_auc)
            opt_hila_model_diff_auc = int(opt_model_auc) - int(hila_auc_by_img_idx[idx])
            if base_opt_model_diff_auc > 10 and opt_hila_model_diff_auc < 0:
                interesting_indices_to_look.append(idx)
    print(interesting_indices_to_look)
    return interesting_indices_to_look


def plot_image_and_masks(images_indices, images_idx_by_auc_diff_base_opt_model, base_model_pickles_path,
                         opt_model_pickles_path,
                         vit_for_image_classification, feature_extractor):
    PLOTS_OUTPUT_PATH = "/home/yuvalas/explainability/pickles/comparison_base_opt_models"
    # for key in images_idx_by_auc_diff_base_opt_model.keys():
    #     ic(key, len(images_idx_by_auc_diff_base_opt_model[key]))
    # for auc_diff in images_idx_by_auc_diff_base_opt_model.keys():
    for idx in images_indices:
        # os.makedirs(Path(PLOTS_OUTPUT_PATH, f"auc_diff_{str(auc_diff)}"), exist_ok=True)
        base_model_loaded_obj = load_obj(Path(base_model_pickles_path, f'{str(idx)}.pkl'))
        opt_model_loaded_obj = load_obj(Path(opt_model_pickles_path, f'{str(idx)}.pkl'))
        image = get_image(Path(IMAGENET_VAL_IMAGES_FOLDER_PATH,
                               f'ILSVRC2012_val_{str(idx + 1).zfill(8)}.JPEG'))  # images are one-based
        image = image if image.mode == "RGB" else image.convert("RGB")
        image_copy_resized = resize(deepcopy(image))
        image_inputs = feature_extractor(image, return_tensors="pt").to(vit_for_image_classification.device)
        class_idx = vit_for_image_classification(**image_inputs).logits.argmax().item()
        predicted_class_name = vit_for_image_classification.config.id2label[class_idx]
        base_vis = plot_vis_on_image(original_image=image_copy_resized, mask=base_model_loaded_obj["vis"])
        opt_vis = plot_vis_on_image(original_image=image_copy_resized, mask=opt_model_loaded_obj["vis"])
        plt.imshow(base_vis)
        plt.title(f'base-pred: {predicted_class_name} - idx = {idx} auc = {int(base_model_loaded_obj["auc"])}')
        plt.axis('off')
        plt.savefig(fname=Path(PLOTS_OUTPUT_PATH, f"{idx}_base.png"))
        # plt.savefig(fname=Path(PLOTS_OUTPUT_PATH, f"auc_diff_{str(auc_diff)}", f"{idx}_base.png"))

        plt.imshow(opt_vis)
        plt.title(f'opt-pred: {predicted_class_name} - idx = {idx} auc = {int(opt_model_loaded_obj["auc"])}')
        plt.axis('off')
        plt.savefig(fname=Path(PLOTS_OUTPUT_PATH, f"{idx}_opt.png"))
        # plt.savefig(fname=Path(PLOTS_OUTPUT_PATH, f"auc_diff_{str(auc_diff)}", f"{idx}_opt.png"))


def plot_masks_on_image_by_image_indices(images_indices, pickles_path):
    for idx in tqdm(images_indices):
        loaded_obj = load_obj(Path(pickles_path, f'{str(idx)}.pkl'))
        image = get_image(Path(IMAGENET_VAL_IMAGES_FOLDER_PATH,
                               f'ILSVRC2012_val_{str(int(idx) + 1).zfill(8)}.JPEG'))  # images are one-based
        image = image if image.mode == "RGB" else image.convert("RGB")
        image_copy_resized = resize(deepcopy(image))
        # save_only_image(image=image_copy_resized, image_idx=idx, is_with_vis=False)
        # save_only_image(image=image_copy_resized * loaded_obj["vis"].cpu(), image_idx=idx, is_with_vis=True)
        # save_mask(mask=loaded_obj["vis"], image_idx=idx)
        vis = plot_vis_on_image(original_image=image_copy_resized, mask=loaded_obj["vis"])
        save_original_image_mask(image=vis, image_idx=idx)

        # plt.imshow(base_vis)
        # plt.title(f'base-pred: {predicted_class_name} - idx = {idx} auc = {int(base_model_loaded_obj["auc"])}')
        # plt.axis('off')
        # plt.savefig(fname=Path(PLOTS_OUTPUT_PATH, f"{idx}_base.png"))


t = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def save_only_image(image, image_idx: int, is_with_vis: bool = False) -> None:  # [1,3,224,224] or [3,224,224]
    image = image if len(image.shape) == 3 else image.squeeze(0)
    image = transforms.ToPILImage()(image)
    image = image.resize((224, 224))
    plt.imshow(transforms.ToTensor()(image).permute(1, 2, 0))
    plt.axis('off');
    # plt.show();
    image_name = f"{image_idx}_image_dot_vis" if is_with_vis else f"{image_idx}_image"
    path = f"/home/yuvalas/explainability/research/plots/our_masks/{image_name}.png"
    print(path)
    plt.margins(x=0, y=0)
    plt.savefig(path, dpi=1500, bbox_inches='tight', pad_inches=0, transparent=True)


def save_original_image_mask(image, image_idx: int) -> None:  # [1,3,224,224] or [3,224,224]
    image = image if len(image.shape) == 3 else image.squeeze(0)
    plt.imshow(t(image).permute(1, 2, 0))
    plt.axis('off');
    # plt.show();
    path = f"/home/yuvalas/explainability/research/plots/additional/ours/{image_idx}_ours.png"
    plt.margins(x=0, y=0)
    plt.savefig(path, dpi=900, bbox_inches='tight', pad_inches=0, transparent=True)


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def plot_vis_on_image(original_image, mask):
    """
    :param original_image.shape: [3, 224, 224]
    :param mask.shape: [1,1, 224, 224]:
    """
    mask = mask.data.squeeze(0).squeeze(0).cpu().numpy()  # [1,1,224,224]
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
            image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(img=image_transformer_attribution, mask=mask)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


if __name__ == '__main__':
    # START_RUN_TIME = dt(2022, 10, 29, 21, 35)  # start time of the experiment for calculating expected end time
    # IMAGES_IDX_BY_AUC_DIFF_BASE_OPT_MODEL_PATH = "/home/yuvalas/explainability/pickles/images_idx_by_auc_diff_base_opt_model.pkl"
    IMAGES_IDX_BY_AUC_TARGET_OPT_MODEL_PATH = "/home/yuvalas/explainability/pickles/images_idx_by_auc_target_opt.pkl"
    HILA_AUC_BY_IMG_IDX_PATH = "/home/yuvalas/explainability/pickles/hila_auc_by_img_idx.pkl"

    for backbone_type in VIT_BACKBONE_DETAILS.keys():
        for target_or_predicted_model in ["predicted", "target"]:
            if target_or_predicted_model == "predicted" and backbone_type == "google/vit-base-patch16-224":
                HOME_BASE_PATH = VIT_BACKBONE_DETAILS[backbone_type]["experiment_base_path"][target_or_predicted_model]
                OPTIMIZATION_PKL_PATH = Path(HOME_BASE_PATH)
                OPTIMIZATION_PKL_PATH_BASE = Path(OPTIMIZATION_PKL_PATH, "base_model", "objects_pkl")
                OPTIMIZATION_PKL_PATH_OPT = Path(OPTIMIZATION_PKL_PATH, "opt_model", "objects_pkl")
                # find_perturbation_interesting_images_one_stage(model_pickles_path=OPTIMIZATION_PKL_PATH_OPT, path_dir=IMAGES_IDX_BY_AUC_TARGET_OPT_MODEL_PATH)
                """
                ic(OPTIMIZATION_PKL_PATH_BASE)
                ic(len(os.listdir(OPTIMIZATION_PKL_PATH_BASE)))
                calculate_mean_auc(n_samples=len(os.listdir(OPTIMIZATION_PKL_PATH_BASE)), path=OPTIMIZATION_PKL_PATH_BASE)
                ic(OPTIMIZATION_PKL_PATH_OPT)
                ic(len(os.listdir(OPTIMIZATION_PKL_PATH_OPT)))
                
                """
                HILA_AUC_WORST_THAN_IMAGE_INDICES_PATH = "/home/yuvalas/explainability/pickles/hila_image_indices_worst_than_us.pkl"
                hila_1_auc_worst_than_ours_image_indices = load_obj(path=HILA_AUC_WORST_THAN_IMAGE_INDICES_PATH)
                # image_names = [5943]
                image_names = hila_1_auc_worst_than_ours_image_indices
                plot_masks_on_image_by_image_indices(images_indices=image_names,
                                                     pickles_path=OPTIMIZATION_PKL_PATH_BASE)
                # calculate_mean_auc(n_samples=len(os.listdir(OPTIMIZATION_PKL_PATH_OPT)), path=OPTIMIZATION_PKL_PATH_OPT)
                # check_all_images_in_pickles(path=OPTIMIZATION_PKL_PATH_OPT)
                # calculate_mean_auc(n_samples=len(os.listdir(OPTIMIZATION_PKL_PATH_OPT)), path=OPTIMIZATION_PKL_PATH_OPT)

    # images_idx_by_auc_diff_base_opt_model = load_obj(path=IMAGES_IDX_BY_AUC_DIFF_BASE_OPT_MODEL_PATH)
    # hila_auc_by_img_idx = load_obj(path=HILA_AUC_BY_IMG_IDX_PATH)

    # find_perturbation_interesting_images_stage_a_stage_b(base_model_pickles_path=OPTIMIZATION_PKL_PATH_BASE,
    #                                                      opt_model_pickles_path=OPTIMIZATION_PKL_PATH_OPT)
    # interesting_indices_to_look = find_perturbation_interesting_images_models_a_b_hila(
    #     images_idx_by_auc_diff_base_opt_model=images_idx_by_auc_diff_base_opt_model,
    #     hila_auc_by_img_idx=hila_auc_by_img_idx,
    #     base_model_pickles_path=OPTIMIZATION_PKL_PATH_BASE,
    #     opt_model_pickles_path=OPTIMIZATION_PKL_PATH_OPT)

    # plot_image_and_masks(images_indices=interesting_indices_to_look,
    #                      images_idx_by_auc_diff_base_opt_model=images_idx_by_auc_diff_base_opt_model,
    #                      base_model_pickles_path=OPTIMIZATION_PKL_PATH_BASE,
    #                      opt_model_pickles_path=OPTIMIZATION_PKL_PATH_OPT,
    #                      vit_for_image_classification=vit_for_image_classification,
    #                      feature_extractor=feature_extractor)
    # """
