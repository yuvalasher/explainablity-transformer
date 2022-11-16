import numpy as np
from pathlib import Path

from tqdm import tqdm

from main.detection_eval.detection_metrics import calculate_AP
import pickle


def load_obj(path: str):
    with open(Path(path), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    IOU_PICKLE_PATH = "/home/yuvalas/explainability/pickles/detection/vit_base/ours_stage_a"
    print(IOU_PICKLE_PATH)

    ious = load_obj(Path(IOU_PICKLE_PATH, "iou_list_end.pkl"))
    ap_05 = calculate_AP(IOUs=ious, iou_threshold=0.5)
    ap_075 = calculate_AP(IOUs=ious, iou_threshold=0.75)

    aps = []
    for threshold in tqdm(np.arange(0.5, 0.95, 0.05)):
        ap_by_threshold = calculate_AP(IOUs=ious, iou_threshold=threshold)
        aps.append(ap_by_threshold)

    print(f"AP0.5: {100 * round(ap_05, 4)}%")
    print(f"AP0.75: {100 * round(ap_075, 4)}%")
    print(f"mAP: {100 * round(np.mean(aps), 4)}%")
    print(1)