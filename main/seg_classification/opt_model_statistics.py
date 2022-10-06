from matplotlib import pyplot as plt
import os
from config import config
import torch

from main.seg_classification.evaluation_functions import get_image
from utils.consts import IMAGENET_VAL_IMAGES_FOLDER_PATH
from utils.transformation import resize

device = torch.device(type='cuda', index=config["general"]["gpu_index"])
from tqdm import tqdm
from icecream import ic
from pathlib import Path
import pickle
import numpy as np
from collections import Counter

from evaluation.perturbation_tests.seg_cls_perturbation_tests import run_perturbation_test_opt, eval_perturbation_test
from vit_loader.load_vit import load_vit_pretrained

BEST_AUC_VALUE = 6


def load_obj(path: str):
    with open(Path(path), 'rb') as f:
        return pickle.load(f)


def calculate_stats_base_and_opt(n_samples: int, base_path, opt_path):
    s_to_5 = 0
    s_smaller_opt_than_baseline = 0
    s_wrong = 0
    for image_idx in range(n_samples):
        image_base_model_path = Path(base_path, f"{str(image_idx)}.pkl")
        image_optimization_path = Path(opt_path, f"{str(image_idx)}.pkl")

        base_model_loaded_obj = load_obj(image_base_model_path)
        optimization_loaded_obj = load_obj(image_optimization_path)

        # ic(base_model_loaded_obj['auc'], optimization_loaded_obj['auc'])
        if optimization_loaded_obj['auc'] > base_model_loaded_obj['auc']:
            s_wrong += 1
        if optimization_loaded_obj['auc'] < BEST_AUC_VALUE and base_model_loaded_obj['auc'] > BEST_AUC_VALUE:
            s_to_5 += 1
        if optimization_loaded_obj['auc'] < base_model_loaded_obj['auc']:
            s_smaller_opt_than_baseline += 1
    ic(s_to_5)
    ic(100 * (s_to_5 / n_samples))
    ic(s_smaller_opt_than_baseline)
    ic(100 * (s_smaller_opt_than_baseline / n_samples))
    ic(s_wrong)


def get_precentage_counter(c):
    return sorted([(i, str(round(count / sum(c.values()) * 100.0, 3)) + '%') for i, count in c.most_common()])


def calculate_mean_auc(n_samples: int, path):
    aucs = []
    for image_idx in tqdm(range(n_samples)):
        image_path = Path(path, f"{str(image_idx)}.pkl")
        loaded_obj = load_obj(image_path)
        aucs.append(loaded_obj['auc'])

    # print(f'AUCS: {aucs}')
    counter = Counter(aucs)
    print(sorted(counter.items()))
    print(get_precentage_counter(counter))
    print(f"{len(aucs)} samples")
    print(f"Mean AUC: {np.mean(aucs)}")


def statistics_run_time(path):
    import datetime
    from datetime import datetime as dt

    n_samples_already_run = len(os.listdir(path))
    start_time = dt(2022, 10, 1, 21, 15)
    avg_seconds_per_image = (dt.now() - start_time).total_seconds() / n_samples_already_run
    expected_run_time_hours = (avg_seconds_per_image * 50000) / 3600
    expected_run_time_days = expected_run_time_hours / 24
    expected_datetime = start_time + datetime.timedelta(hours=expected_run_time_hours)
    print(
        f"N_samples: {n_samples_already_run}; Avg. seconds per image: {avg_seconds_per_image}; Expected run time (days): {expected_run_time_days}; Data: {expected_datetime}")


def show_mask(mask):  # [1, 1, 224, 224]
    mask = mask if len(mask.shape) == 3 else mask.squeeze(0)
    _ = plt.imshow(mask.squeeze(0).cpu().detach())
    plt.show()


def plot_image(image) -> None:  # [1,3,224,224] or [3,224,224]
    image = image if len(image.shape) == 3 else image.squeeze(0)
    plt.imshow(image.cpu().detach().permute(1, 2, 0))
    plt.show();


if __name__ == '__main__':
    # OPTIMIZATION_PKL_PATH = "/home/yuvalas/explainability/research/experiments/seg_cls/ft_3000/opt_objects"
    # BASE_MODEL_PKL_PATH = "/home/yuvalas/explainability/research/experiments/seg_cls/ft_3000/base_model/opt_objects"
    # calculate_stats_base_and_opt(n_samples=n_samples, base_path=BASE_MODEL_PKL_PATH, opt_path=OPTIMIZATION_PKL_PATH)
    OPTIMIZATION_PKL_PATH = "/home/yuvalas/explainability/research/experiments/seg_cls/ft_50000/opt_objects"

    n_samples = len(os.listdir(OPTIMIZATION_PKL_PATH)) - 5
    calculate_mean_auc(n_samples=n_samples, path=OPTIMIZATION_PKL_PATH)
    statistics_run_time(path=OPTIMIZATION_PKL_PATH)
    """
    vit_for_image_classification, _ = load_vit_pretrained(model_name="google/vit-base-patch16-224")
    vit_for_image_classification = vit_for_image_classification.to(device)
    # for idx in [15,16,17]:#,18,19,20,21:
    print('Base model')
    aucs = []
    for idx in [1, 2, 4, 7, 10, 12, 13, 15, 18, 19, 20, 22, 24, 27]:
        loaded_obj = load_obj(
            Path("/home/yuvalas/explainability/research/experiments/seg_cls/ft_50000/base_model/opt_objects",
                 f'{str(idx)}.pkl'))
        image = get_image(Path(IMAGENET_VAL_IMAGES_FOLDER_PATH,
                               f'ILSVRC2012_val_{str(idx + 1).zfill(8)}.JPEG'))  # images are one-based
        image = image if image.mode == "RGB" else image.convert("RGB")
        image_resized = resize(image).unsqueeze(0)
        outputs = [
            {"image_resized": image_resized.to(device), "image_mask": loaded_obj["vis"].to(device)}]
        plot_image(image_resized)
        show_mask(loaded_obj["vis"])
        print(1)
        # auc = eval_perturbation_test(experiment_dir=Path(""), model=vit_for_image_classification, outputs=outputs)
        # aucs.append(auc)
    print(aucs)
    """
