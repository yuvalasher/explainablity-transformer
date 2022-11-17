import os
from typing import List

import numpy as np
from pathlib import Path

from tqdm import tqdm

from main.detection_eval.detection_metrics import calculate_AP
import pickle
import numpy as np
from mean_average_precision import MetricBuilder

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def load_obj(path: str):
    with open(Path(path), 'rb') as f:
        return pickle.load(f)

def calculate_ap(IOUs: List[float]) -> None:
    ap_05 = calculate_AP(IOUs=ious, iou_threshold=0.5)
    ap_075 = calculate_AP(IOUs=ious, iou_threshold=0.75)

    aps = []
    for threshold in tqdm(np.arange(0.5, 0.95, 0.05)):
        ap_by_threshold = calculate_AP(IOUs=ious, iou_threshold=threshold)
        aps.append(ap_by_threshold)

    print(f"AP0.5: {100 * round(ap_05, 4)}%")
    print(f"AP0.75: {100 * round(ap_075, 4)}%")
    print(f"mAP: {100 * round(np.mean(aps), 4)}%")

if __name__ == '__main__':
    IOU_PICKLE_PATH = "/home/yuvalas/explainability/pickles/detection/vit_base/ours_stage_a"
    print(IOU_PICKLE_PATH)
    ious = load_obj(Path(IOU_PICKLE_PATH, "iou_list_end.pkl"))
    gt_preds_bboxes =load_obj(Path(IOU_PICKLE_PATH, "gt_preds_bboxs.pkl"))
    i = len(ious)
    # i = 15000
    print(i)
    gt = np.array(gt_preds_bboxes["gt_boxes"])[:i, :]
    preds = np.array(gt_preds_bboxes["preds_bbox"])[:i, :]

    # print list of available metrics
    print(MetricBuilder.get_metrics_list())

    # create metric_fn
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)

    # add some samples to evaluation
    for i in range(1): # TODO ??
        metric_fn.add(preds, gt)

    # compute PASCAL VOC metric
    print(f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")

    # compute PASCAL VOC metric at the all points
    print(f"VOC PASCAL 0.5 mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}")
    print(f"VOC PASCAL 0.75 mAP in all points: {metric_fn.value(iou_thresholds=0.75)['mAP']}")

    # compute metric COCO metric
    print(
        f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")

    # ious = [0.7,0.3,0.5,0.6,0.55,0.9]
    # ap_05 = calculate_AP(IOUs=ious, iou_threshold=0.5)
    # print(ap_05)
    # print(1)


