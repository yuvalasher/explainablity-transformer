import json
import random
from copy import deepcopy

import cv2
import pandas as pd
from matplotlib import pyplot as plt
import os

from transformers import ViTForImageClassification

from config import config
import torch

from feature_extractor import ViTFeatureExtractor
from main.seg_classification.evaluation_functions import get_image
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
    plt.title(f'{idx} auc = {auc}')
    # plt.savefig(f"/home/amiteshel1/Projects/explainablity-transformer-cv/amit_png_del/mask_{idx}.png")
    plt.show()


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


def find_perturbation_interesting_images(base_model_pickles_path, opt_model_pickles_path):
    images_idx_by_auc_diff_base_opt_model = {5: [], 10: [], 15: [], 20: [], 25: [], 30: [], 35: [], 40: [], 45: [],
                                             50: [], 60: []}
    # n_images = len(os.listdir(base_model_pickels_path))
    n_images = 20000
    random.seed(config["general"]["seed"])
    rands = sorted(random.sample(range(1, 39000), n_images))
    # print(sorted(rands))

    for idx in tqdm(rands):
        base_model_loaded_obj = load_obj(Path(base_model_pickles_path, f'{str(idx)}.pkl'))
        opt_model_loaded_obj = load_obj(Path(opt_model_pickles_path, f'{str(idx)}.pkl'))
        base_model_auc = base_model_loaded_obj["auc"]
        opt_model_auc = opt_model_loaded_obj["auc"]
        base_opt_model_diff_auc = int(base_model_auc) - int(opt_model_auc)
        if base_opt_model_diff_auc in images_idx_by_auc_diff_base_opt_model.keys():
            images_idx_by_auc_diff_base_opt_model[base_opt_model_diff_auc].append(idx)
    save_obj_to_disk(path=IMAGES_IDX_BY_AUC_DIFF_BASE_OPT_MODEL_PATH, obj=images_idx_by_auc_diff_base_opt_model)


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
    IMAGES_IDX_BY_AUC_DIFF_BASE_OPT_MODEL_PATH = "/home/yuvalas/explainability/pickles/images_idx_by_auc_diff_base_opt_model.pkl"
    HILA_AUC_BY_IMG_IDX_PATH = "/home/yuvalas/explainability/pickles/hila_auc_by_img_idx.pkl"

    HOME_BASE_PATH = "/home/yuvalas/explainability/research/experiments/seg_cls"
    EXP_NAME = "direct_opt_ckpt_27_model_google_vit-base-patch16-224_train_uni_True_val_unif_True_activation_sigmoid__norm_by_max_p_False_pred_1_mask_l_bce_50__train_n_samples_6000_lr_0.002_mlp_classifier_True__bs_32"
    OPTIMIZATION_PKL_PATH = Path(HOME_BASE_PATH, EXP_NAME)
    OPTIMIZATION_PKL_PATH_BASE = Path(OPTIMIZATION_PKL_PATH, "base_model", "objects_pkl")
    OPTIMIZATION_PKL_PATH_OPT = Path(OPTIMIZATION_PKL_PATH, "opt_model", "objects_pkl")
    images_idx_by_auc_diff_base_opt_model = load_obj(path=IMAGES_IDX_BY_AUC_DIFF_BASE_OPT_MODEL_PATH)
    hila_auc_by_img_idx = load_obj(path=HILA_AUC_BY_IMG_IDX_PATH)

    feature_extractor = ViTFeatureExtractor.from_pretrained(vit_config["model_name"])
    if vit_config["model_name"] in ["google/vit-base-patch16-224"]:
        vit_for_image_classification, _ = load_vit_pretrained(
            model_name=vit_config["model_name"])
    else:
        vit_for_image_classification = ViTForImageClassification.from_pretrained(vit_config["model_name"])

    vit_for_image_classification = vit_for_image_classification.to(device)

    # find_perturbation_interesting_images(base_model_pickles_path=OPTIMIZATION_PKL_PATH_BASE, opt_model_pickles_path=OPTIMIZATION_PKL_PATH_OPT)
    # interesting_indices_to_look = find_perturbation_interesting_images_models_a_b_hila(
    #     images_idx_by_auc_diff_base_opt_model=images_idx_by_auc_diff_base_opt_model,
    #     hila_auc_by_img_idx=hila_auc_by_img_idx,
    #     base_model_pickles_path=OPTIMIZATION_PKL_PATH_BASE,
    #     opt_model_pickles_path=OPTIMIZATION_PKL_PATH_OPT)
    interesting_indices_to_look = [2982, 3232, 8425, 9458, 9683, 9955, 10902, 14239, 15683, 18863, 19926, 21284, 28272,
                                   31552, 32010, 34402, 34406,
                                   18, 46, 118, 141, 165, 212, 250, 370, 413, 471, 504, 507, 522, 530, 609, 631, 639,
                                   681, 753, 777, 803, 834, 845,
                                   851, 864, 891, 904, 942, 973, 974, 1021, 1031, 1044, 1054, 1069, 1076, 1129, 1152,
                                   1159, 1196, 1227, 1246, 1251,
                                   1252, 1254, 1291, 1307, 1318, 1326, 1390, 1496, 1499, 1567, 1603, 1606, 1681, 1689,
                                   1691, 1746, 1773, 1784, 1807,
                                   1846, 1901, 1924, 1939, 1984, 1985, 2015, 2065, 2073, 2082, 2132, 2143, 2144, 2258,
                                   2281, 2292, 2294, 2340, 2366,
                                   2395, 2472, 2503, 2551, 2559, 2564, 2644, 2645, 2668, 2679, 2688, 2705, 2769, 2775,
                                   2778, 2789, 2814, 2874, 2887,
                                   2907, 2940, 2972, 2981, 3008, 3009, 3010, 3030, 3050, 3057, 3084, 3089, 3130, 3133,
                                   3263, 3307, 3310, 3368, 3370,
                                   3383, 3389, 3407, 3421, 3501, 3505, 3519, 3562, 3571, 3576, 3585, 3599, 3613, 3646,
                                   3668, 3681, 3700, 3757, 3759,
                                   3778, 3786, 3805, 3808, 3845, 3878, 3884, 3919, 3925, 3926, 3943, 3945, 4030, 4051,
                                   4055, 4136, 4180, 4190, 4213,
                                   4224, 4254, 4298, 4314, 4335, 4435, 4444, 4454, 4470, 4527, 4540, 4554, 4621, 4649,
                                   4669, 4716, 4744, 4782, 4783,
                                   4784, 4788, 4809, 4818, 4819, 4854, 4861, 4907, 4954, 4966, 5088, 5120, 5125, 5163,
                                   5176, 5188, 5213, 5227, 5244,
                                   5258, 5262, 5312, 5318, 5334, 5340, 5384, 5415, 5444, 5474, 5513, 5529, 5605, 5611,
                                   5690, 5747, 5762, 5774, 5792,
                                   5798, 5807, 5834, 5855, 5878, 5887, 5928, 5934, 5943, 5960, 5988, 5998, 6067, 6081,
                                   6086, 6096, 6140, 6188, 6193,
                                   6214, 6227, 6233, 6238, 6245, 6250, 6262, 6323, 6365, 6386, 6388, 6445, 6450, 6466,
                                   6509, 6521, 6580, 6618, 6657,
                                   6717, 6727, 6733, 6784, 6797, 6813, 6823, 6868, 6874, 6876, 6910, 6982, 6983, 6995,
                                   7069, 7111, 7137, 7147, 7156,
                                   7158, 7170, 7220, 7244, 7252, 7311, 7360, 7398, 7409, 7443, 7483, 7555, 7556, 7682,
                                   7729, 7748, 7870, 7894, 7952,
                                   7958, 7962, 8027, 8039, 8049, 8087, 8097, 8099, 8199, 8209, 8231, 8238, 8264, 8451,
                                   8461, 8498, 8505, 8523, 8530,
                                   8533, 8555, 8572, 8587, 8621, 8627, 8660, 8701, 8719, 8761, 8813, 8823, 8870, 8958,
                                   9001, 9010, 9048, 9105, 9109,
                                   9165, 9179, 9213, 9248, 9254, 9257, 9262, 9275, 9277, 9310, 9316, 9344, 9436, 9463,
                                   9482, 9507, 9509, 9573, 9578,
                                   9617, 9723, 9727, 9741, 9764, 9766, 9780, 9807, 9833, 9850, 9868, 9894, 9909, 9915,
                                   9931, 9940, 9951, 9959, 9974,
                                   9992, 10079, 10110, 10130, 10136, 10169, 10179, 10188, 10198, 10245, 10265, 10285,
                                   10293, 10317, 10319, 10362,
                                   10363, 10384, 10389, 10413, 10418, 10434, 10440, 10445, 10462, 10467, 10469, 10474,
                                   10507, 10635, 10650, 10657,
                                   10703, 10709, 10711, 10747, 10748, 10804, 10817, 10822, 10838, 10851, 10874, 10881,
                                   10913, 10974, 10978, 10992,
                                   11005, 11007, 11033, 11046, 11078, 11137, 11156, 11208, 11269, 11277, 11289, 11296,
                                   11298, 11316, 11351, 11363,
                                   11489, 11492, 11514, 11515, 11536, 11547, 11565, 11652, 11667, 11700, 11707, 11712,
                                   11725, 11730, 11741, 11753,
                                   11823, 11828, 11838, 11843, 11859, 11875, 11906, 11929, 12053, 12058, 12086, 12111,
                                   12124, 12179, 12204, 12264,
                                   12271, 12276, 12291, 12311, 12320, 12334, 12364, 12408, 12463, 12479, 12519, 12520,
                                   12558, 12585, 12589, 12595,
                                   12612, 12631, 12758, 12778, 12801, 12803, 12825, 12875, 12878, 12899, 12933, 12979,
                                   13015, 13027, 13066, 13080,
                                   13097, 13105, 13138, 13164, 13188, 13206, 13217, 13285, 13287, 13306, 13326, 13340,
                                   13343, 13350, 13404, 13461,
                                   13488, 13503, 13510, 13522, 13530, 13545, 13555, 13665, 13680, 13687, 13710, 13719,
                                   13730, 13747, 13793, 13848,
                                   13933, 13935, 13957, 13963, 13996, 14006, 14043, 14079, 14104, 14155, 14158, 14174,
                                   14230, 14250, 14275, 14276,
                                   14281, 14282, 14319, 14335, 14336, 14349, 14356, 14371, 14384, 14446, 14470, 14480,
                                   14518, 14560, 14579, 14596,
                                   14655, 14678, 14702, 14709, 14772, 14775, 14820, 14823, 14848, 14884, 14969, 14974,
                                   14990, 14993, 15001, 15016,
                                   15032, 15035, 15087, 15099, 15108, 15120, 15146, 15184, 15203, 15206, 15271, 15319,
                                   15427, 15478, 15486, 15540,
                                   15552, 15673, 15768, 15805, 15871, 15883, 15892, 15939, 15947, 15986, 16028, 16113,
                                   16154, 16259, 16271, 16276,
                                   16386, 16390, 16395, 16399, 16407, 16434, 16438, 16458, 16477, 16497, 16522, 16523,
                                   16532, 16548, 16574, 16641,
                                   16645, 16655, 16705, 16732, 16815, 16823, 16920, 16949, 17027, 17051, 17054, 17081,
                                   17084, 17090, 17104, 17121,
                                   17168, 17180, 17232, 17237, 17256, 17259, 17267, 17287, 17289, 17331, 17339, 17370,
                                   17481, 17530, 17563, 17578,
                                   17590, 17628, 17651, 17661, 17683, 17728, 17738, 17741, 17781, 17793, 17840, 17853,
                                   17890, 17911, 17917, 17931,
                                   17948, 17972, 17997, 18030, 18050, 18080, 18120, 18125, 18172, 18229, 18241, 18256,
                                   18257, 18404, 18450, 18475,
                                   18482, 18528, 18538, 18568, 18588, 18620, 18636, 18645, 18655, 18683, 18687, 18751,
                                   18834, 18847, 18876, 18888,
                                   18954, 19000, 19030, 19058, 19157, 19162, 19166, 19231, 19258, 19271, 19317, 19328,
                                   19358, 19359, 19369, 19417,
                                   19440, 19444, 19537, 19602, 19615, 19620, 19704, 19709, 19730, 19765, 19770, 19823,
                                   19826, 19876, 19941, 19947,
                                   20001, 20024, 20034, 20090, 20097, 20116, 20142, 20160, 20164, 20218, 20250, 20284,
                                   20313, 20329, 20341, 20345,
                                   20445, 20466, 20480, 20482, 20523, 20541, 20546, 20559, 20591, 20692, 20714, 20802,
                                   20818, 20845, 20855, 20876,
                                   20909, 20923, 20939, 21003, 21013, 21018, 21032, 21058, 21090, 21092, 21105, 21110,
                                   21155, 21182, 21204, 21207,
                                   21209, 21213, 21221, 21224, 21227, 21253, 21264, 21287, 21307, 21308, 21356, 21363,
                                   21378, 21419, 21430, 21435,
                                   21459, 21475, 21503, 21509, 21533, 21622, 21647, 21686, 21758, 21801, 21802, 21803,
                                   21808, 21809, 21843, 21966,
                                   21988, 22029, 22042, 22071, 22073, 22083, 22160, 22164, 22186, 22197, 22217, 22237,
                                   22247, 22260, 22264, 22304,
                                   22305, 22322, 22428, 22496, 22518, 22535, 22571, 22577, 22612, 22625, 22677, 22704,
                                   22709, 22785, 22793, 22818,
                                   22839, 22847, 22860, 22896, 22899, 22923, 22944, 22964, 23008, 23035, 23052, 23064,
                                   23065, 23105, 23244, 23262,
                                   23284, 23329, 23338, 23432, 23487, 23488, 23494, 23533, 23560, 23564, 23592, 23602,
                                   23639, 23668, 23673, 23683,
                                   23692, 23700, 23760, 23767, 23807, 23853, 23868, 23888, 23897, 23898, 23920, 23926,
                                   23949, 23991, 24016, 24031,
                                   24057, 24075, 24082, 24090, 24118, 24119, 24156, 24167, 24172, 24303, 24325, 24332,
                                   24343, 24361, 24370, 24440,
                                   24467, 24477, 24531, 24580, 24610, 24619, 24643, 24646, 24651, 24654, 24687, 24690,
                                   24740, 24748, 24765, 24776,
                                   24793, 24803, 24820, 24823, 24849, 24857, 24877, 24925, 24963, 24993, 25014, 25051,
                                   25059, 25090, 25104, 25105,
                                   25120, 25197, 25223, 25226, 25262, 25315, 25369, 25409, 25430, 25433, 25547, 25571,
                                   25577, 25647, 25668, 25685,
                                   25720, 25832, 25865, 25868, 25871, 25876, 25884, 25905, 25916, 25930, 25940, 25984,
                                   25996, 26038, 26042, 26071,
                                   26132, 26141, 26178, 26186, 26226, 26248, 26270, 26274, 26281, 26318, 26328, 26347,
                                   26379, 26427, 26483, 26516,
                                   26543, 26561, 26576, 26603, 26612, 26634, 26693, 26701, 26741, 26780, 26782, 26828,
                                   26840, 26869, 26886, 26946,
                                   26975, 26979, 27050, 27104, 27128, 27174, 27184, 27192, 27205, 27210, 27247, 27250,
                                   27273, 27275, 27282, 27323,
                                   27341, 27356, 27376, 27409, 27458, 27466, 27494, 27524, 27542, 27594, 27607, 27649,
                                   27687, 27697, 27713, 27749,
                                   27790, 27808, 27869, 27872, 27900, 27907, 27952, 27967, 27971, 27995, 28010, 28053,
                                   28086, 28104, 28107, 28115,
                                   28153, 28207, 28237, 28242, 28260, 28265, 28286, 28303, 28343, 28356, 28360, 28384,
                                   28421, 28427, 28430, 28431,
                                   28533, 28547, 28567, 28577, 28599, 28635, 28706, 28720, 28728, 28739, 28751, 28775,
                                   28792, 28811, 28835, 28866,
                                   28906, 28921, 28988, 29034, 29057, 29076, 29092, 29097, 29115, 29136, 29156, 29171,
                                   29184, 29252, 29265, 29276,
                                   29277, 29293, 29300, 29308, 29313, 29314, 29391, 29449, 29475, 29507, 29518, 29554,
                                   29562, 29572, 29603, 29646,
                                   29668, 29678, 29715, 29727, 29768, 29789, 29841, 29914, 29958, 29974, 30020, 30021,
                                   30064, 30072, 30087, 30089,
                                   30119, 30167, 30168, 30203, 30207, 30218, 30227, 30233, 30247, 30257, 30259, 30277,
                                   30309, 30325, 30348, 30383,
                                   30419, 30461, 30477, 30492, 30525, 30529, 30542, 30553, 30620, 30675, 30719, 30731,
                                   30740, 30781, 30834, 30851,
                                   30857, 30871, 30935, 30939, 31008, 31022, 31043, 31048, 31058, 31239, 31241, 31274,
                                   31297, 31299, 31343, 31365,
                                   31380, 31400, 31420, 31423, 31518, 31527, 31537, 31540, 31573, 31651, 31668, 31672,
                                   31734, 31808, 31853, 31906,
                                   31912, 31914, 31960, 31961, 32014, 32017, 32026, 32168, 32203, 32252, 32257, 32340,
                                   32347, 32386, 32427, 32491,
                                   32496, 32500, 32509, 32515, 32562, 32590, 32614, 32655, 32710, 32729, 32748, 32757,
                                   32804, 32807, 32833, 32882,
                                     32890, 32901, 32941, 32943, 32944, 32974, 32975, 32980, 32999, 33028, 33036, 33089,
                                   33111, 33191, 33206, 33209,
                                   33266, 33313, 33386, 33405, 33433, 33439, 33457, 33515, 33519, 33522, 33528, 33531,
                                   33536, 33541, 33568, 33570,
                                   33578, 33727, 33731, 33799, 33858, 33904, 33923, 33937, 33952, 34074, 34075, 34109,
                                   34119, 34138, 34169, 34192,
                                   34231, 34267, 34325, 34327, 34332, 34452, 34530, 34561, 34654, 34696, 34720, 34780,
                                   34785, 34797, 34868, 34915,
                                   34924, 34976, 34983, 35069, 35072, 35075, 35091, 35101, 35114, 35119, 35129, 35209,
                                   35223, 35235, 35254, 35296,
                                   35344, 35400, 35420, 35424, 35466, 35516, 35535, 35544, 35574, 35595, 35602, 35659,
                                   35711, 35755, 35761, 35815,
                                   35868, 35883, 35954, 35963, 35990, 36011, 36040, 36052, 36065, 36069, 36071, 36113,
                                   36133, 36144, 36163, 36199,
                                   36213, 36215, 36250, 36252, 36282, 36287, 36322, 36348, 36444, 36484, 36486, 36493,
                                   36523, 36531, 36571, 36600,
                                   36602, 36697, 36719, 36748, 36755, 36793, 36851, 36856, 36886, 36943, 36947, 37006,
                                   37017, 37018, 37028, 37056,
                                   37060, 37107, 37138, 37158, 37159, 37207, 37276, 37307, 37318, 37322, 37357, 37386,
                                   37465, 37501, 37553, 37584,
                                   37623, 37638, 37690, 37695, 37719, 37777, 37783, 37807, 37839, 37958, 38085, 38086,
                                   38138, 38146, 38266, 38360,
                                   38387, 38389, 38471, 38482, 38494, 38535, 38582, 38620, 38685, 38686, 38718, 38780,
                                   38819, 38820, 38859, 38866,
                                   38886, 38891, 38899, 38950, 38960, 6751, 8858, 11161, 11483, 15910, 16264, 16627,
                                   18708, 19242, 21629, 21996,
                                   25049, 25155, 28807, 28864, 29188, 30232, 88, 137, 167, 287, 306, 340, 402, 463, 537,
                                   657, 717, 808, 830, 916,
                                   1077, 1094, 1124, 1132, 1175, 1242, 1303, 1368, 1379, 1966, 2003, 2090, 2187, 2253,
                                   2517, 2689, 2752, 2754, 2791,
                                   2857, 2860, 3020, 3149, 3166, 3246, 3287, 3294, 3478, 3515, 3551, 3635, 3640, 3715,
                                   3796, 3829, 3989, 4029, 4201,
                                   4225, 4284, 4405, 4411, 4499, 4509, 4551, 4652, 4667, 4837, 4881, 4923, 4993, 5037,
                                   5101, 5175, 5191, 5254, 5292,
                                   5351, 5407, 5583, 5587, 5595, 5645, 5695, 5802, 5854, 5890, 5952, 6014, 6032, 6057,
                                   6139, 6172, 6252, 6322, 6369,
                                   6378, 6444, 6875, 6883, 6891, 6938, 7014, 7033, 7100, 7179, 7180, 7206, 7219, 7243,
                                   7289, 7307, 7377, 7413, 7420,
                                   7513, 7520, 7675, 7702, 7723, 7806, 7940, 7943, 8108, 8160, 8175, 8275, 8317, 8529,
                                   8689, 8707, 8907, 9093, 9118,
                                   9281, 9558, 9577, 9639, 9660, 9685, 9705, 9895, 10017, 10066, 10104, 10333, 10456,
                                   10615, 10938, 10939, 10956,
                                   11009, 11044, 11187, 11281, 11313, 11350, 11520, 11538, 11544, 11626, 11631, 11674,
                                   11816, 11832, 11855, 12016,
                                   12028, 12030, 12033, 12037, 12052, 12277, 12345, 12517, 12518, 12752, 12841, 12913,
                                   12984, 13047, 13153, 13154,
                                   13162, 13183, 13184, 13227, 13260, 13398, 13533, 13600, 13618, 13657, 14100, 14181,
                                   14214, 14372, 14529, 14538,
                                   14562, 14565, 14833, 14853, 14872, 14878, 14913, 14916, 15011, 15030, 15119, 15157,
                                   15226, 15267, 15360, 15417,
                                   15471, 15473, 15506, 15513, 15519, 15600, 15608, 15660, 15839, 15909, 15941, 15946,
                                   16031, 16206, 16231, 16269,
                                   16491, 16529, 16738, 16841, 16867, 16889, 16966, 17055, 17072, 17169, 17196, 17284,
                                   17328, 17449, 17571, 17617,
                                   17747, 17801, 17863, 17894, 17971, 18017, 18136, 18192, 18265, 18328, 18413, 18452,
                                   18461, 18470, 18737, 18757,
                                   18781, 18849, 18869, 19036, 19146, 19265, 19339, 19391, 19622, 19818, 20004, 20079,
                                   20275, 20289, 20291, 20446,
                                   20450, 20462, 20681, 20741, 20777, 20834, 20847, 20862, 20865, 20987, 21108, 21167,
                                   21198, 21323, 21330, 21358,
                                   21399, 21461, 21516, 21590, 21591, 21603, 21610, 21702, 21722, 21933, 21970, 22031,
                                   22034, 22103, 22110, 22111,
                                   22254, 22438, 22461, 22637, 22649, 22850, 22935, 22943, 22969, 22980, 23016, 23066,
                                   23082, 23266, 23311, 23680,
                                   23739, 23761, 24004, 24100, 24111, 24181, 24250, 24291, 24319, 24328, 24358, 24514,
                                   24548, 24587, 24672, 24878,
                                   24901, 24933, 24975, 25024, 25100, 25126, 25183, 25193, 25233, 25235, 25243, 25245,
                                   25451, 25656, 25866, 25917,
                                   26001, 26048, 26066, 26334, 26342, 26406, 26545, 26553, 26581, 26722, 26744, 26761,
                                   26894, 26982, 27019, 27034,
                                   27092, 27161, 27180, 27226, 27317, 27434, 27436, 27439, 27583, 27660, 27701, 27721,
                                   27759, 27880, 27886, 28222,
                                   28243, 28268, 28369, 28441, 28536, 28587, 28602, 28637, 28646, 28712, 28732, 28827,
                                   28946, 28981, 28990, 29038,
                                   29046, 29194, 29226, 29230, 29274, 29354, 29442, 29496, 29541, 29599, 29633, 29782,
                                   29820, 29953, 30029, 30163,
                                   30261, 30328, 30340, 30374, 30382, 30442, 30472, 30601, 30761, 30879, 30880, 30937,
                                   30987, 31027, 31029, 31083,
                                   31101, 31142, 31374, 31528, 31584, 31600, 31725, 31735, 31910, 32000, 32022, 32023,
                                   32064, 32070, 32107, 32165,
                                   32350, 32374, 32413, 32475, 32700, 32741, 32759, 32761, 32771, 32799, 33021, 33134,
                                   33145, 33198, 33207, 33267,
                                   33318, 33328, 33391, 33482, 33791, 33958, 33975, 34029, 34134, 34147, 34161, 34188,
                                   34301, 34363, 34463, 34507,
                                   34546, 34567, 34592, 34603, 34631, 34647, 34741, 34787, 34802, 34861, 34928, 34960,
                                   34986, 35087, 35115, 35283,
                                   35305, 35403, 35469, 35571, 35681, 35686, 35869, 35905, 35936, 35955, 35967, 36181,
                                   36278, 36307, 36349, 36362,
                                   36399, 36429, 36516, 36565, 36622, 36643, 36722, 36759, 36774, 36821, 36853, 36950,
                                   36956, 36995, 37070, 37085,
                                   37135, 37139, 37257, 37312, 37579, 37642, 37662, 37717, 37728, 37853, 37921, 38002,
                                   38212, 38228, 38251, 38285,
                                   38300, 38317, 38363, 38585, 38592, 38701, 38776, 38821, 38828, 38881, 38887, 38918,
                                   38940, 38951, 38972, 38987,
                                   1720, 9062, 11511, 14508, 15459, 18714, 20639, 21406, 456, 710, 729, 754, 1056, 1332,
                                   1450, 1671, 1700, 1740, 1831,
                                   1859, 1980, 2534, 2566, 2691, 2700, 2750, 3104, 3105, 3206, 3209, 3414, 4144, 4282,
                                   4741, 4795, 4928, 5022, 5078,
                                   5102, 5237, 5596, 5939, 6027, 6324, 6877, 7268, 7341, 8048, 8600, 8693, 8741, 8971,
                                   9043, 9085, 9319, 9324, 9510,
                                   9920, 10019, 10911, 11410, 11543, 12471, 12490, 12718, 12871, 13293, 13566, 14014,
                                   14217, 14225, 14620, 15236,
                                   15602, 15615, 15856, 15911, 15987, 16044, 16049, 16230, 16296, 16316, 16583, 16629,
                                   16704, 16723, 16753, 17202,
                                   17455, 17528, 17565, 18063, 18092, 18357, 18489, 18624, 18739, 18797, 18818, 18946,
                                   18958, 18982, 19101, 19113,
                                   19153, 19346, 19446, 19493, 19535, 19586, 19604, 19610, 19644, 19839, 19842, 20000,
                                   20068, 20653, 20658, 20733,
                                   20795, 20981, 21666, 21780, 22011, 22040, 22055, 22301, 22778, 22830, 23359, 23568,
                                   23581, 23676, 23966, 24761,
                                   25293, 26118, 26162, 26245, 26316, 26363, 26518, 26689, 26692, 26708, 26880, 26992,
                                   26998, 27193, 27543, 27645,
                                   27863, 28016, 28058, 28696, 28791, 28917, 29424, 29823, 29899, 29932, 29956, 30345,
                                   30595, 30683, 30863, 30869,
                                   30918, 30934, 31208, 31468, 31483, 31603, 31973, 32212, 32650, 32752, 32826, 32848,
                                   33038, 33040, 33080, 33150,
                                   33351, 33785, 33885, 33965, 33969, 34160, 34488, 34490, 35155, 35187, 35192, 35244,
                                   35550, 35641, 35805, 35950,
                                   36064, 36530, 36675, 36790, 36919, 37055, 37300, 37410, 37540, 37983, 38066, 38361,
                                   38653, 38834, 38903, 38945,
                                   27566, 29845, 1918, 2313, 3941, 4707, 5064, 5287, 6045, 6225, 7635, 9053, 11307,
                                   11846, 12573, 15436, 16471, 16868,
                                   17835, 19148, 19306, 20759, 21173, 22032, 22582, 23128, 23811, 24115, 24369, 24460,
                                   26817, 26849, 27005, 27239,
                                   27980, 28251, 28510, 28556, 28594, 28867, 29133, 29177, 29228, 29466, 29639, 30449,
                                   30545, 30991, 31500, 31517,
                                   31631, 32312, 33144, 33662, 33945, 33968, 34152, 34420, 34632, 34904, 35728, 36734,
                                   36994, 37104, 37147, 37164,
                                   37996, 38108, 38350, 102, 351, 6732, 7817, 7883, 9901, 10753, 12506, 17004, 17466,
                                   21113, 21373, 21576, 22046,
                                   24599, 26207, 29377, 31488, 38928]

    plot_image_and_masks(images_indices=interesting_indices_to_look,
                         images_idx_by_auc_diff_base_opt_model=images_idx_by_auc_diff_base_opt_model,
                         base_model_pickles_path=OPTIMIZATION_PKL_PATH_BASE,
                         opt_model_pickles_path=OPTIMIZATION_PKL_PATH_OPT,
                         vit_for_image_classification=vit_for_image_classification,
                         feature_extractor=feature_extractor)

    # START_RUN_TIME = dt(2022, 10, 31, 13, 19)
    # """
    # ic(OPTIMIZATION_PKL_PATH_BASE)
    # calculate_mean_auc(n_samples=len(os.listdir(OPTIMIZATION_PKL_PATH_BASE)), path=OPTIMIZATION_PKL_PATH_BASE)
    # statistics_expected_run_time(path=OPTIMIZATION_PKL_PATH_BASE, start_time=START_RUN_TIME)
    #
    # """
    # ic(OPTIMIZATION_PKL_PATH_OPT)
    # calculate_mean_auc(n_samples=len(os.listdir(OPTIMIZATION_PKL_PATH_OPT)), path=OPTIMIZATION_PKL_PATH_OPT)
    # statistics_expected_run_time(path=OPTIMIZATION_PKL_PATH_OPT, start_time=START_RUN_TIME)

    # plotting visualizations with images by saved pickles

    # OPTIMIZATION_PKL_PATH_BASE = Path(OPTIMIZATION_PKL_PATH, "base_model", "objects_pkl")
    # OPTIMIZATION_PKL_PATH_OPT = Path(OPTIMIZATION_PKL_PATH, "opt_model", "objects_pkl")
    """
    for idx in range(10):
        loaded_obj = load_obj(Path(OPTIMIZATION_PKL_PATH_OPT, f'{str(idx)}.pkl'))
        image = get_image(Path(IMAGENET_VAL_IMAGES_FOLDER_PATH,
                               f'ILSVRC2012_val_{str(idx + 1).zfill(8)}.JPEG'))  # images are one-based
        image = image if image.mode == "RGB" else image.convert("RGB")
        image_resized = resize(image).unsqueeze(0)
        plot_image(image_resized, idx, loaded_obj['auc'])
        show_mask(loaded_obj["vis"], idx, loaded_obj['auc'])
    """
