import numpy as np
import os
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

GT_VALIDATION_PATH_LABELS = "/home/yuvalas/explainability/data/val ground truth 2012.txt"
DETECTION_GT_PATH = "/home/amiteshel1/Projects/explainablity-transformer/imagenet_detection_gt/val"
DEFAULT_LIST_OF_IMAGES = list(range(50000))

import pickle


class DetectionImageNetDataset(Dataset):

    def __init__(self, root_dir, pkl_path, list_of_images_names: List[int] = DEFAULT_LIST_OF_IMAGES, transform=None):
        self.root_dir = root_dir
        self.xlm_gt_path = DETECTION_GT_PATH
        self.pkl_path = pkl_path
        self.transform = transform
        self.listdir = sorted(os.listdir(root_dir))
        self.images_name = [f"ILSVRC2012_val_{str(image_idx + 1).zfill(8)}.JPEG" for image_idx in list_of_images_names]
        print(f"After filter images: {len(self.images_name)}")

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        image_name = self.images_name[idx]
        image_xml_gt_path = Path(self.xlm_gt_path, image_name.replace("JPEG", "xml"))
        image_path = Path(self.root_dir, image_name)
        loaded_obj = load_obj(Path(self.pkl_path, f"{idx}.pkl"))
        image_mask = loaded_obj["vis"]
        image = Image.open(image_path)
        image = image if image.mode == "RGB" else image.convert("RGB")  # Black & White images

        x_y_width_height_dict = get_x_min_y_min_x_max_y_max_width_height_parse_xml_gt_file(path=image_xml_gt_path)
        if self.transform:
            image = self.transform(image)  # TODO - maybe not to resize!
        return image, image_mask.squeeze(0).squeeze(0), x_y_width_height_dict


def get_size_from_gt_root(root):
    sizes = root.findall("size")[0]
    width, height = int(sizes[0].text), int(sizes[1].text)
    return dict(width=width, height=height)


def get_x_min_y_min_x_max_y_max_width_height_parse_xml_gt_file(path):
    tree = ET.parse(path)
    root = tree.getroot()
    width_height_dict = get_size_from_gt_root(root=root)
    x_y_dict = get_biggest_bbox_from_gt(root=root)
    return dict(x_min=x_y_dict["x_min"], y_min=x_y_dict["y_min"], x_max=x_y_dict["x_max"], y_max=x_y_dict["y_max"],
                width=width_height_dict["width"], height=width_height_dict["height"])


def get_biggest_bbox_from_gt(root):
    bboxes = {}
    areas = []
    for box_idx, item in enumerate(root.findall("object")):
        x_min, y_min, x_max, y_max = [int(val.text) for val in item[-1]]
        area = (x_max - x_min) * (y_max - y_min)
        areas.append(area)
        bboxes[box_idx] = {"area": area,
                           "x_y": [x_min, y_min, x_max, y_max]
                           }
    biggest_box_idx = np.array(areas).argmax()
    x_min, y_min, x_max, y_max = bboxes[biggest_box_idx]["x_y"]
    return dict(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def show_pil_image(image):
    from matplotlib import pyplot as plt
    plt.imshow(image)
    plt.show()


def get_gt_classes(path):
    with open(path, 'r') as f:
        gt_classes_list = f.readlines()
    gt_classes_list = [int(record.split()[-1].replace('\n', '')) for record in gt_classes_list]
    return gt_classes_list


def load_obj(path: str):
    with open(Path(path), 'rb') as f:
        return pickle.load(f)
