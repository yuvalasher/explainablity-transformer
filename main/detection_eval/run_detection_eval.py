from typing import Tuple

from matplotlib import pyplot as plt
from pathlib import Path
import torch
from torchvision import transforms
from tqdm import tqdm

from main.detection_eval.single_bbox_image_indices import SINGLE_BBOX_IMAGE_INDICES
from main.detection_eval.detection_metrics import calculate_AP
from main.detection_eval.detection_dataset import DetectionImageNetDataset
import cv2
import numpy as np
from torch.utils.data import DataLoader
from main.seg_classification.vit_backbone_to_details import VIT_BACKBONE_DETAILS
import pickle
import os
from podm.box import Box, intersection_over_union

IMAGENET_VALIDATION_PATH = "/home/amiteshel1/Projects/explainablity-transformer/vit_data/"
IOU_PICKLE_PATH = "/home/yuvalas/explainability/pickles/detection"

METHOD = "ours_stage_b"
VIT_TYPE = "vit_small"

transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def save_obj_to_disk(path, obj) -> None:
    if type(path) == str and path[-4:] != '.pkl':
        path += '.pkl'

    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path: str):
    with open(Path(path), 'rb') as f:
        return pickle.load(f)


IS_VERBOSE = False


def get_xmin_xmax_y_min_y_max(mask) -> Tuple[int, int, int, int]:
    """
    :param mask.shape: [224, 224]
    :return:
    """
    gx, gy = np.gradient(mask)
    temp_edge = gy * gy + gx * gx
    temp_edge[temp_edge != 0.0] = 255.0
    edged = np.asarray(temp_edge, dtype=np.uint8)
    if IS_VERBOSE:
        plt.imshow(edged)

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

        edged_new_gt = cv2.rectangle(mask.astype('uint8'),
                                     (gt_x_min, gt_y_min), (gt_x_max, gt_y_max), (255, 0, 0), 5)
        plt.imshow(edged_new_gt, cmap="gray")
        plt.title('bbox')
        plt.show()
    return x_min, y_min, x_max, y_max


if __name__ == '__main__':
    ious = []
    for backbone_name in VIT_BACKBONE_DETAILS.keys():
        for target_or_predicted_model in ["predicted"]:
            # if backbone_name == "google/vit-base-patch16-224":
            if backbone_name == "WinKawaks/vit-small-patch16-224":
                gt_bboxs = []  # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
                preds_bbox = []  # [xmin, ymin, xmax, ymax, class_id, confidence]
                HOME_BASE_PATH = VIT_BACKBONE_DETAILS[backbone_name]["experiment_base_path"][target_or_predicted_model]
                OPTIMIZATION_PKL_PATH = Path(HOME_BASE_PATH)
                OPTIMIZATION_PKL_PATH_BASE = Path(OPTIMIZATION_PKL_PATH, "base_model", "objects_pkl")
                # OPTIMIZATION_PKL_PATH_OPT = Path(OPTIMIZATION_PKL_PATH, "opt_model", "objects_pkl")

                images_indices = sorted(SINGLE_BBOX_IMAGE_INDICES)
                imagenet_ds = DetectionImageNetDataset(root_dir=IMAGENET_VALIDATION_PATH,
                                                       pkl_path=OPTIMIZATION_PKL_PATH_BASE,
                                                       list_of_images_names=images_indices,
                                                       transform=transform)
                sample_loader = DataLoader(
                    imagenet_ds,
                    batch_size=1,
                    shuffle=False,
                    num_workers=1
                )
                for datum_idx, datum in enumerate(tqdm(sample_loader)):
                    mask, x_y_gt_dict = datum
                    gt_x_min, gt_y_min, gt_x_max, gt_y_max = x_y_gt_dict["x_min"].item(), x_y_gt_dict["y_min"].item(), \
                                                             x_y_gt_dict["x_max"].item(), x_y_gt_dict["y_max"].item()

                    mask = mask.squeeze(0).squeeze(0)
                    mask_mean = mask.mean()
                    mask = torch.where(mask >= mask_mean, torch.tensor(1.).to(mask.device),
                                       torch.tensor(0.).to(mask.device)).cpu()
                    mask = cv2.resize(mask.numpy(), (x_y_gt_dict["width"].item(), x_y_gt_dict["height"].item()),
                                      interpolation=cv2.INTER_AREA)

                    x_min, y_min, x_max, y_max = get_xmin_xmax_y_min_y_max(mask=mask)
                    our_box = Box.of_box(xtl=x_min, ytl=y_min, xbr=x_max, ybr=y_max)
                    gt_box = Box.of_box(xtl=gt_x_min, ytl=gt_y_min, xbr=gt_x_max, ybr=gt_y_max)
                    iou = intersection_over_union(our_box, gt_box)
                    ious.append(iou)
                    gt_bboxs.append([gt_x_min, gt_y_min, gt_x_max, gt_y_max, 0, 0, 0])
                    preds_bbox.append([x_min, y_min, x_max, y_max, 0, iou])

                OUTPUT_PATH = Path(IOU_PICKLE_PATH, VIT_TYPE, METHOD)
                print(OUTPUT_PATH)
                save_obj_to_disk(path=Path(OUTPUT_PATH, f"gt_preds_bboxs.pkl"),
                                 obj={"gt_boxes": gt_bboxs, "preds_bbox": preds_bbox})
                # save_obj_to_disk(path=Path(OUTPUT_PATH, f"iou_list_end.pkl"), obj=ious)
                # print(ious)
