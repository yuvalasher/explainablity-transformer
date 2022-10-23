import pandas as pd
from matplotlib import pyplot as plt
import os
from config import config
import torch

from main.seg_classification.evaluation_functions import get_image
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

BEST_AUC_VALUE = 6


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
    df, df_stats = calculate_count__and_prec_auc(aucs)
    print(tabulate(df_stats, headers='keys'))
    plot_perturbations_vs_num_of_images(df)
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
    plt.title(f'idx = {idx} auc = {auc}')
    # plt.savefig(f"/home/amiteshel1/Projects/explainablity-transformer-cv/amit_png_del/mask_{idx}.png")
    plt.show()


def plot_image(image, idx, auc) -> None:  # [1,3,224,224] or [3,224,224]
    image = image if len(image.shape) == 3 else image.squeeze(0)
    plt.imshow(image.cpu().detach().permute(1, 2, 0))
    plt.title(f'idx = {idx} auc = {auc}')
    # plt.savefig(f"/home/amiteshel1/Projects/explainablity-transformer-cv/amit_png_del/image_{idx}.png")
    plt.show()



def plot_visualizations_and_images(path_to_exp_pickles: str, vit_for_image_classification, ):
    """
    Can calculate mean aucs from pickels
    """
    aucs = []
    for idx in tqdm(range(4400,4450)):
        # for idx in range(10):

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


if __name__ == '__main__':
    # OPTIMIZATION_PKL_PATH = "/home/yuvalas/explainability/research/experiments/seg_cls/ft_pasten/opt_objects"
    # opt
    OPTIMIZATION_PKL_PATH = "/home/amiteshel1/Projects/explainablity-transformer-cv/research/experiments/seg_cls/ft_50000_new_model_only_opt/opt_model/objects_pkl"
    # base
    # OPTIMIZATION_PKL_PATH = "/home/amiteshel1/Projects/explainablity-transformer-cv/research/experiments/seg_cls/ft_50000_new_model_only_base/base_model/objects_pkl/"

    # seg - opt
    # OPTIMIZATION_PKL_PATH = "/home/amiteshel1/Projects/explainablity-transformer-cv/research/experiments/seg_cls/ft_50000_new_model_seg_only_opt/opt_model/objects_pkl/"

    # seg - base
    # OPTIMIZATION_PKL_PATH = "/home/amiteshel1/Projects/explainablity-transformer-cv/research/experiments/seg_cls/ft_50000_new_model_seg_only_base/opt_model/objects_pkl/"

    print(OPTIMIZATION_PKL_PATH)

    # Calculation of statistics on pickles - can be during running experiment - calculating by the last pickle
    n_samples = len(os.listdir(OPTIMIZATION_PKL_PATH))
    calculate_mean_auc(n_samples=n_samples, path=OPTIMIZATION_PKL_PATH)
    statistics_expected_run_time(path=OPTIMIZATION_PKL_PATH, start_time=dt(2022, 10, 11, 19,
                                                                           24))  # start time of the experiment for calculating expected end time

    # plotting visualizations with images by saved pickles

    # vit_for_image_classification, _ = load_vit_pretrained(model_name="google/vit-base-patch16-224")
    # vit_for_image_classification = vit_for_image_classification.to(device)
    # plot_visualizations_and_images(path_to_exp_pickles=OPTIMIZATION_PKL_PATH,
    #                                vit_for_image_classification=vit_for_image_classification)
