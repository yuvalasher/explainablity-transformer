import gc
import os
import sys

os.chdir('/home/amiteshel1/Projects/explainablity-transformer-cv/')
sys.path.append('/home/amiteshel1/Projects/explainablity-transformer-cv/')

from pathlib import Path
from typing import Tuple

import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from data.detection_lst import SINGLE_BBOX_IMAGE_INDICES
from data.imagenet import DetectionImageNetDataset, GTImageNetDataset
from datasets.imagenet_results_dataset import ImagenetResults
from main.seg_classification.vit_backbone_to_details import VIT_BACKBONE_DETAILS
from main.segmentation_eval.segmentation_utils import print_segmentation_results
from utils.metrices import batch_pix_accuracy, batch_intersection_union, get_ap_scores, get_f1_scores
import xml.etree.ElementTree as ET

import warnings
import logging

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger('checkpoint').setLevel(0)
logging.getLogger('lightning').setLevel(0)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def eval_results_per_bacth(Res, labels, q=-1):
    Res = (Res - Res.min()) / (Res.max() - Res.min())

    # ret = th_torch
    if q == -1:
        ret = Res.mean()
    else:
        ret = torch.quantile(Res, q=q)

    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())
    Res_1_AP = Res
    Res_0_AP = 1 - Res
    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0
    Res_1_AP[Res_1_AP != Res_1_AP] = 0
    Res_0_AP[Res_0_AP != Res_0_AP] = 0
    # TEST
    pred = Res.clamp(min=0.0) / Res.max()
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()
    output = torch.cat((Res_0, Res_1), 1)
    output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)
    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
    batch_ap, batch_f1 = 0, 0
    # Segmentation resutls
    correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
    inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)
    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union
    # print("output", output.shape)
    # print("ap labels", labels.shape)
    # ap = np.nan_to_num(get_ap_scores(output, labels))
    ap = np.nan_to_num(get_ap_scores(output_AP, labels))
    f1 = np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0]))
    batch_ap += ap
    batch_f1 += f1

    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, pred, target


def eval_results_per_bacth_cuda(Res, labels, q=-1):
    Res = (Res - Res.min()) / (Res.max() - Res.min())

    # ret = th_torch
    if q == -1:
        ret = Res.mean()
    else:
        ret = torch.quantile(Res, q=q)

    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())
    Res_1_AP = Res
    Res_0_AP = 1 - Res
    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0
    Res_1_AP[Res_1_AP != Res_1_AP] = 0
    Res_0_AP[Res_0_AP != Res_0_AP] = 0
    # TEST
    pred = Res.clamp(min=0.0) / Res.max()
    pred = pred.view(-1).data
    target = labels.view(-1).data
    output = torch.cat((Res_0, Res_1), 1)
    output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)
    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
    batch_ap, batch_f1 = 0, 0
    # Segmentation resutls
    correct, labeled = batch_pix_accuracy(output[0].data, labels[0])
    inter, union = batch_intersection_union(output[0].data, labels[0], 2)
    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union
    # print("output", output.shape)
    # print("ap labels", labels.shape)
    # ap = np.nan_to_num(get_ap_scores(output, labels))
    ap = np.nan_to_num(get_ap_scores(output_AP, labels))
    f1 = np.nan_to_num(get_f1_scores(output[0, 1].data, labels[0]))
    batch_ap += ap
    batch_f1 += f1

    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, pred, target


def load_mask(path_results, exp_name):
    if 'HILA' in exp_name:
        imagenet_ds = ImagenetResults(path_results)
    else:
        imagenet_ds = DetectionImageNetDataset(pkl_path=path_results)

    return imagenet_ds


def get_xmin_xmax_y_min_y_max(mask, width, height, q=-1, IS_VERBOSE=False) -> Tuple[int, int, int, int]:
    """
    :param mask.shape: [224, 224]
    :return:
    """
    if q == -1:
        ret = mask.mean()
    else:
        ret = torch.quantile(mask, q=q)
    mask = torch.where(mask >= ret, torch.tensor(1.).to(mask.device), torch.tensor(0.).to(mask.device)).cpu()
    mask = cv2.resize(mask[0].cpu().clone().numpy(), (width, height), interpolation=cv2.INTER_AREA)
    gx, gy = np.gradient(mask)
    temp_edge = gy * gy + gx * gx
    temp_edge[temp_edge != 0.0] = 255.0
    edged = np.asarray(temp_edge, dtype=np.uint8)
    if IS_VERBOSE:
        plt.imshow(edged)
        plt.show()

    im2, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.waitKey(1)
    # print("Number of Contours found = " + str(len(contours)))
    coordinates = [cv2.boundingRect(contour) for contour in contours]
    max_rec_idx = np.array([coordinate[2] * coordinate[3] for coordinate in coordinates]).argmax()
    c = contours[max_rec_idx]
    x, y, w, h = cv2.boundingRect(c)
    x_min, y_min, x_max, y_max = x, y, x + w, y + h
    mask.astype('uint8')

    if IS_VERBOSE:
        plt.imshow(mask.astype('uint8'))
        plt.title('mask')
        plt.show()
        edged_new = cv2.rectangle(mask.astype('uint8') * 255, (x_min, y_min), (x_max, y_max), (255, 0, 0), 5)
        plt.imshow(edged_new, cmap="gray")
        plt.title('bbox')
        plt.show()

        # edged_new_gt = cv2.rectangle(mask.astype('uint8'),
        #                              (gt_x_min, gt_y_min), (gt_x_max, gt_y_max), (255, 0, 0), 5)
        # plt.imshow(edged_new_gt, cmap="gray")
        # plt.title('bbox')
        # plt.show()

    return x_min, x_max, y_min, y_max


def get_size_from_gt_root(root):
    sizes = root.findall("size")[0]
    width, height = int(sizes[0].text), int(sizes[1].text)
    return width, height


def get_bbox_from_gt(root):
    for box_idx, item in enumerate(root.findall("object")):
        x_min, y_min, x_max, y_max = [int(val.text) for val in item[-1]]
    return x_min, y_min, x_max, y_max


def get_x_min_y_min_x_max_y_max_width_height_parse_xml_gt_file(path):
    tree = ET.parse(path)
    root = tree.getroot()
    width, height = get_size_from_gt_root(root=root)
    x_min, y_min, x_max, y_max = get_bbox_from_gt(root=root)
    return x_min, x_max, y_min, y_max, width, height


def run(path, exp_name):
    print('exp_name', exp_name)
    print('PATH', path)
    imagenet_ds = load_mask(path, exp_name)  # 50K

    total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    total_ap, total_f1 = 0, 0
    total_ap2, total_f2 = [], []
    count_idx = 0
    for idx in tqdm(SINGLE_BBOX_IMAGE_INDICES[:20000]):
        torch.cuda.empty_cache()
        gc.collect()
        count_idx += 1
        mask = imagenet_ds[idx]
        ############### GT ###############

        x_min_gt, x_max_gt, y_min_gt, y_max_gt, width, height = get_x_min_y_min_x_max_y_max_width_height_parse_xml_gt_file(
            gt_ds[idx])
        labels = torch.zeros((1, width, height))
        labels[x_min_gt: x_max_gt, y_min_gt: y_max_gt] = 1

        ############### OUR MASKS ###############
        x_min, x_max, y_min, y_max = get_xmin_xmax_y_min_y_max(mask.clone(), width, height, q=0.95,
                                                               IS_VERBOSE=False)  # 224,224
        Res = torch.zeros((1, 1, width, height))
        Res[0][0][x_min:x_max, y_min:y_max] = 1

        assert Res.shape == (1, 1, width, height)
        assert labels.shape == (1, width, height)

        correct, labeled, inter, union, ap, f1, pred, target = eval_results_per_bacth_cuda(Res.to('cuda'),
                                                                                           labels.to('cuda'))

        total_correct += correct.astype('int64')
        total_label += labeled.astype('int64')
        total_inter += inter.astype('int64')
        total_union += union.astype('int64')
        ap = np.pad(ap, (0, 1 - len(ap)), 'constant')
        f1 = np.pad(f1, (0, 1 - len(ap)), 'constant')
        total_ap += np.sum(ap)
        total_f1 += np.mean([f1])  # CHECK IF OK f1 is a list, total_f1 list of lists
        # total_ap2 += [ap]
        # total_f2 += [f1]
        pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
        IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
        mIoU = IoU.mean()
        mAp = total_ap / count_idx
        mF1 = total_f1 / count_idx
        # mAP2 = np.mean(total_ap)
        # mF2 = np.mean(total_f2)
        if (count_idx % 500 == 0):
            print('Image number = ', idx)
            print_segmentation_results(pixAcc=pixAcc, mAp=mAp, mIoU=mIoU, mF1=mF1)
        del Res
        del labels
        del mask
    print_segmentation_results(pixAcc=pixAcc, mAp=mAp, mIoU=mIoU, mF1=mF1)


if __name__ == "__main__":
    path_gt = "/home/amiteshel1/Projects/explainablity-transformer/imagenet_detection_gt/val"
    gt_ds = GTImageNetDataset(path_gt)
    # ~~~~~~~~~~~ VIT BASE STAGE A OURS~~~~~~~~~~~~~~~~~~~~~~~~~~

    # exp_name = 'VIT BASE STAGE A'
    # HOME_BASE_PATH = Path(VIT_BACKBONE_DETAILS['google/vit-base-patch16-224']["experiment_base_path"]['predicted'])
    # path_results = Path(HOME_BASE_PATH, "base_model", "objects_pkl")
    #
    # run(path_results, exp_name)

    # # ~~~~~~~~~~~ VIT BASE STAGE B OURS~~~~~~~~~~~~~~~~~~~~~~~~~~
    # exp_name = 'VIT BASE STAGE B OURS'
    # HOME_BASE_PATH = Path(VIT_BACKBONE_DETAILS['google/vit-base-patch16-224']["experiment_base_path"]['predicted'])
    # path_results = Path(HOME_BASE_PATH, "opt_model", "objects_pkl")
    #
    # run(path_results, exp_name)
    # # ~~~~~~~~~~~ VIT SMALL STAGE A OURS~~~~~~~~~~~~~~~~~~~~~~~~~~
    # exp_name = 'VIT SMALL STAGE A OURS'
    # HOME_BASE_PATH = Path(VIT_BACKBONE_DETAILS['WinKawaks/vit-small-patch16-224']["experiment_base_path"]['predicted'])
    # path_results = Path(HOME_BASE_PATH, "base_model", "objects_pkl")
    #
    # run(path_results, exp_name)

    # # ~~~~~~~~~~~ VIT SMALL STAGE B OURS~~~~~~~~~~~~~~~~~~~~~~~~~~
    # exp_name = 'VIT SMALL STAGE B OURS'
    # HOME_BASE_PATH = Path(VIT_BACKBONE_DETAILS['WinKawaks/vit-small-patch16-224']["experiment_base_path"]['predicted'])
    # path_results = Path(HOME_BASE_PATH, "opt_model", "objects_pkl")
    #
    # run(path_results, exp_name)
    #
    # ~~~~~~~~~~~ HILA 1 BASE~~~~~~~~~~~~~~~~~~~~~~~~~~
    # exp_name = 'HILA 1 BASE'
    # path_results = '/home/amiteshel1/Projects/Hila_explain/baselines/ViT/visualizations/transformer_attribution/top/not_ablation/results_hila_17.hdf5'
    #
    # run(path_results, exp_name)

    # # ~~~~~~~~~~~ HILA 1 SMALL ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # exp_name = 'HILA 1 SMALL'
    # path_results = '/home/amiteshel1/Projects/Hila_explain/baselines/ViT/visualizations/transformer_attribution/top/not_ablation/results_vit_small.hdf5'
    #
    # run(path_results, exp_name)
    # #
    # # ~~~~~~~~~~~ HILA 2 BASE~~~~~~~~~~~~~~~~~~~~~~~~~~
    # exp_name = 'HILA 2 BASE'
    # path_results = '/home/amiteshel1/Projects/Hila_explain/baselines/ViT/visualizations/hila_2/top/not_ablation/results.hdf5'
    #
    # run(path_results, exp_name)
    #
    # # ~~~~~~~~~~~ HILA 2 SMALL ~~~~~~~~~~~~~~~~~~~~~~~~~~
    exp_name = 'HILA 2 SMALL'
    path_results = '/home/amiteshel1/Projects/Hila_explain/baselines/ViT/visualizations/hila_2_small/top/not_ablation/results.hdf5'

    run(path_results, exp_name)
