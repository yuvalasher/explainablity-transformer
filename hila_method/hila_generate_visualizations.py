import os
from itertools import chain
import pickle
from datetime import datetime as dt
from icecream import ic
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification
from hila_method.utils.ViT_LRP import vit_base_patch16_224 as vit_LRP
from hila_method.utils.ViT_explanation_generator import LRP
from hila_method.utils.imagenet_dataset import ImageNetDataset
import torch
from main.seg_classification.evaluation_functions import read_image_and_mask_from_pickls_by_path, infer_adp_pic_acp
from vit_loader.load_vit import load_vit_pretrained
from config import config

device = torch.device(type='cuda', index=config["general"]["gpu_index"])
torch.cuda.empty_cache()
from torchvision.transforms import transforms
from pytorch_lightning import seed_everything
import numpy as np
from torch import Tensor
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List
from pathlib import Path
from evaluation.perturbation_tests.seg_cls_perturbation_tests import eval_perturbation_test

seed_everything(config["general"]["seed"])

IMAGENET_VALIDATION_PATH = "/home/amiteshel1/Projects/explainablity-transformer/vit_data/"
GT_VALIDATION_PATH_LABELS = "/home/yuvalas/explainability/data/val ground truth 2012.txt"
HILA_VISUAILZATION_PATH = "/home/yuvalas/explainability/research/plots/hila"


def load_obj(path: str):
    with open(Path(path), 'rb') as f:
        return pickle.load(f)


def save_obj_to_disk(path, obj) -> None:
    if type(path) == str and path[-4:] != '.pkl':
        path += '.pkl'

    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def get_gt_classes(path):
    with open(path, 'r') as f:
        gt_classes_list = f.readlines()
    gt_classes_list = [int(record.split()[-1].replace('\n', '')) for record in gt_classes_list]
    return gt_classes_list


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def visu(original_image, transformer_attribution, file_name: str):
    """
    :param original_image: shape: [3, 224, 224]
    :param transformer_attribution: shape: [n_patches, n_patches] = [14, 14]
    :param file_name:
    :return:
    """
    if type(transformer_attribution) == np.ndarray:
        transformer_attribution = torch.tensor(transformer_attribution)
    transformer_attribution = transformer_attribution.reshape(1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(
        transformer_attribution.unsqueeze(0), scale_factor=16, mode="bilinear"
    )
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
            transformer_attribution.max() - transformer_attribution.min()
    )
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (
                                            image_transformer_attribution - image_transformer_attribution.min()
                                    ) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    plt.imsave(fname=Path(f"{file_name}.png"), arr=vis, format="png")


def normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype)  # , device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype)  # , device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def compute_saliency_and_save(dataloader: DataLoader) -> List[Dict[str, Tensor]]:
    outputs: List[Dict[str, Tensor]] = []
    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        target = target.to(device)
        resized_image = data.clone()
        data = normalize(data)
        data = data.to(device)
        data.requires_grad_()
        Res_patches = lrp.generate_LRP(data, start_layer=1, method="grad", index=None).reshape(data.shape[0], 1, 14,
                                                                                               14)
        with torch.no_grad():
            Res = torch.nn.functional.interpolate(Res_patches, scale_factor=16, mode='bilinear').cpu()  # .cuda()
            Res = (Res - Res.min()) / (Res.max() - Res.min())
            # Res_np = Res.data.cpu().numpy()
            data = data.cpu()
            # Res = Res.data.cpu()
            Res_patches = Res_patches.cpu()
            # ic(data.device, Res.device, Res_patches.device)
            # target = target.cpu()
            # ic(target.device)
        outputs.append(
            {'image_resized': resized_image, 'image_mask': Res, 'patches_mask': Res_patches, 'target_class': target})
    return outputs


def compute_saliency_generator(dataloader: DataLoader):
    for batch_idx, (data, target) in enumerate(dataloader):
        # target = target.to(device)
        resized_image = data.clone()
        data = normalize(data)
        data = data.to(device)
        data.requires_grad_()
        Res_patches = lrp.generate_LRP(data, start_layer=1, method="grad", index=None).reshape(data.shape[0], 1, 14,
                                                                                               14)
        # with torch.no_grad():
        Res = torch.nn.functional.interpolate(Res_patches, scale_factor=16, mode='bilinear').cpu()  # .cuda()
        Res = (Res - Res.min()) / (Res.max() - Res.min())
        # Res_np = Res.data.cpu().numpy()
        data = data.cpu()
        # Res = Res.data.cpu()
        Res_patches = Res_patches.cpu()
        # ic(data.device, Res.device, Res_patches.device)
        # target = target.cpu()
        # ic(target.device)
        yield dict(image_resized=resized_image.to(device), image_mask=Res.to(device))
        # outputs.append({'image_resized': resized_image, 'image_mask': Res, 'patches_mask': Res_patches})


def visualize_outputs(outputs):
    for image_idx, output in tqdm(enumerate(outputs)):
        for idx, (image, mask) in enumerate(
                zip(output["image_resized"].detach().cpu(), output["patches_mask"].detach().cpu())):
            visu(
                original_image=image,
                transformer_attribution=mask,
                file_name=Path(HILA_VISUAILZATION_PATH,
                               f"{str(output['image_idx'])}_hila_auc_{int(output['auc'])}").resolve(),
            )


def run_adp_pic_tests_hila(vit_for_image_classification, images_and_masks):
    gt_classes_list = get_gt_classes(GT_VALIDATION_PATH_LABELS)
    evaluation_metrics = infer_adp_pic_acp(vit_for_image_classification=vit_for_image_classification,
                                           images_and_masks=images_and_masks,
                                           gt_classes_list=gt_classes_list)

    print(
        f'PIC (% Increase in Confidence - Higher is better): {round(evaluation_metrics["percentage_increase_in_confidence"], 4)}%; ADP (Average Drop % - Lower is better): {round(evaluation_metrics["averaged_drop_percentage"], 4)}%; ACP (% Average Change Percentage - Higher is better): {round(evaluation_metrics["averaged_change_percentage"], 4)}%;')



if __name__ == '__main__':
    BATCH_SIZE = 1
    MODEL_NAME = config["vit"]["model_name"]
    model_LRP = vit_LRP(pretrained=True).to(device)
    model_LRP.eval()
    lrp = LRP(model_LRP)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    vit_for_classification_image, _ = load_vit_pretrained(model_name=MODEL_NAME)
    vit_for_classification_image = vit_for_classification_image.to(device)

    IMAGES_IDX_BY_AUC_DIFF_BASE_OPT_MODEL_PATH = "/home/yuvalas/explainability/pickles/images_idx_by_auc_diff_base_opt_model.pkl"
    HILA_AUC_BY_IMG_IDX_PATH = "/home/yuvalas/explainability/pickles/hila_auc_by_img_idx.pkl"
    # PLOTS_OUTPUT_PATH = "/home/yuvalas/explainability/pickles/comparison_base_opt_models"

    images_idx_by_auc_diff_base_opt_model = load_obj(path=IMAGES_IDX_BY_AUC_DIFF_BASE_OPT_MODEL_PATH)

    # images_indices = []
    # for key in images_idx_by_auc_diff_base_opt_model.keys():
    #     images_indices.extend(images_idx_by_auc_diff_base_opt_model[key])
    # interesting_indices_to_look = [218, 245, 456, 457, 549, 710, 729, 754, 878, 1037, 1056, 1234, 1332, 1450, 1654,
    #                                1671,1700, 1740, 1831, 1859, 1875, 1980, 2534, 2566, 2691, 2700, 2750, 2763, 2808,
    #                                2820,2985,3032, 3083, 3104, 3105, 3206, 3209, 3344, 3414, 3864, 4144, 4282, 4741, 4795,
    #                                4928,5018,5022, 5078, 5086, 5102, 5186, 5196, 5237, 5596, 5939, 6018, 6027, 6324, 6734,
    #                                6821,6877,6912, 6984, 7021, 7268, 7306, 7341, 8048, 8268, 8311, 8408, 8600, 8693, 8734,
    #                                8741,8971,9043, 9085, 9319, 9324, 9510, 9637, 9920, 9939, 10019, 10264, 10314, 10512,
    #                                10830,10875,10911, 11079, 11410, 11543, 11721, 11864, 11867, 12146, 12392, 12424, 12471,
    #                                12490,12630, 12718, 12804, 12871, 12920, 13252, 13293, 13494, 13566, 13654, 13790, 14014,
    #                                14026, 14189, 14217, 14225, 14620, 14646, 14681, 14985, 7305, 9376, 14229, 149, 193,
    #                                215,994, 1187, 1515, 1631, 1918, 2313, 3941, 4031, 4707, 4888, 5064, 5287, 5463,
    #                                6045,6225,6887, 7340, 7635, 8439, 8444, 9053, 9110, 9224, 10003, 11153, 11307, 11846,
    #                                12433,12573,14514, 14703]
    # images_indices = interesting_indices_to_look

    images_indices = list(chain(*images_idx_by_auc_diff_base_opt_model.values()))
    print(len(images_indices))
    n_samples = len(images_indices)
    imagenet_ds = ImageNetDataset(root_dir=IMAGENET_VALIDATION_PATH, n_samples=n_samples,
                                  list_of_images_names=images_indices, transform=transform)
    sample_loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    hila_auc_by_img_idx = {}
    outputs = compute_saliency_and_save(dataloader=sample_loader)
    for idx, output in enumerate(outputs):
        auc = eval_perturbation_test(experiment_dir=Path(""), model=vit_for_classification_image, outputs=[output])
        output["auc"] = auc
        output["image_idx"] = images_indices[idx]
        # ic(images_indices[idx])
        hila_auc_by_img_idx[images_indices[idx]] = auc
    save_obj_to_disk(path=HILA_AUC_BY_IMG_IDX_PATH, obj=hila_auc_by_img_idx)

    # print(f'AUC: {round(auc, 4)}% for {len(outputs)} records')
    # for key in images_idx_by_auc_diff_base_opt_model.keys():
    #     if image_idx in images_idx_by_auc_diff_base_opt_model[key]:
    #         plot_subfolder_name = f"auc_diff_{key}"
    # load_obj(HILA_AUC_BY_IMG_IDX_PATH)
    # visualize_outputs(outputs=outputs)

    # ADP & PIC
    # images_and_masks_generator = compute_saliency_generator(dataloader=sample_loader)
    # run_adp_pic_tests_hila(vit_for_image_classification=vit_for_classification_image, images_and_masks=images_and_masks_generator)
