from datetime import datetime as dt
from icecream import ic
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification
from evaluation.perturbation_tests.seg_cls_perturbation_tests import run_perturbation_test, eval_perturbation_test
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

seed_everything(config["general"]["seed"])

HILA_VISUAILZATION_PATH = "/home/yuvalas/explainability/research/plots/hila"


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
        # target = target.to(device)
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
        outputs.append({'image_resized': resized_image, 'image_mask': Res, 'patches_mask': Res_patches})
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
    print(len(outputs))
    for batch_idx, output in enumerate(outputs):
        for idx, (image, mask) in enumerate(
                zip(output["image_resized"].detach().cpu(), output["patches_mask"].detach().cpu())):
            visu(
                original_image=image,
                transformer_attribution=mask,
                file_name=Path(HILA_VISUAILZATION_PATH, f"hila_{batch_idx}").resolve(),
            )


def run_adp_pic_tests_hila(vit_for_image_classification: ViTForImageClassification, images_and_masks, ADP_PIC_config):
    gt_classes_list = get_gt_classes(GT_VALIDATION_PATH_LABELS)
    start_time = dt.now()
    print(
        f'ADP_PIC tests for {ADP_PIC_config["IS_COMPARED_BY_TARGET"]}; data: Hila')
    print(
        f'Evaluation Params: IS_COMPARED_BY_TARGET: {ADP_PIC_config["IS_COMPARED_BY_TARGET"]}, IS_CLAMP_BETWEEN_0_TO_1: {ADP_PIC_config["IS_CLAMP_BETWEEN_0_TO_1"]}')

    evaluation_metrics = infer_adp_pic_acp(vit_for_image_classification=vit_for_image_classification,
                                           images_and_masks=images_and_masks,
                                           gt_classes_list=gt_classes_list, ADP_PIC_config=ADP_PIC_config)
    print(
        f'ADP_PIC tests for IS_COMPARED_BY_TARGET: {ADP_PIC_config["IS_COMPARED_BY_TARGET"]}; data: Hila')
    print(
        f'PIC (% Increase in Confidence - Higher is better): {round(evaluation_metrics["percentage_increase_in_confidence"], 4)}%; ADP (Average Drop % - Lower is better): {round(evaluation_metrics["averaged_drop_percentage"], 4)}%; ACP (% Average Change Percentage - Higher is better): {round(evaluation_metrics["averaged_change_percentage"], 4)}%;')
    print(f"timing: {(dt.now() - start_time).total_seconds()}")


IMAGENET_VALIDATION_PATH = "/home/amiteshel1/Projects/explainablity-transformer/vit_data/"
GT_VALIDATION_PATH_LABELS = "/home/yuvalas/explainability/data/val ground truth 2012.txt"
if __name__ == '__main__':
    BATCH_SIZE = 1
    MODEL_NAME = 'google/vit-base-patch16-224'
    model_LRP = vit_LRP(pretrained=True).to(device)
    model_LRP.eval()
    lrp = LRP(model_LRP)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    vit_for_classification_image, _ = load_vit_pretrained(model_name=MODEL_NAME)
    vit_for_classification_image = vit_for_classification_image.to(device)
    n_samples = 5000
    imagenet_ds = ImageNetDataset(root_dir=IMAGENET_VALIDATION_PATH, n_samples=n_samples, transform=transform)
    sample_loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    outputs = compute_saliency_and_save(dataloader=sample_loader)
    visualize_outputs(outputs=outputs)
    # remove_old_results_dfs(experiment_path=experiment_path)
    ADP_PIC_config = {'IS_CLAMP_BETWEEN_0_TO_1': True, 'IS_COMPARED_BY_TARGET': False}
    images_and_masks = compute_saliency_generator(dataloader=sample_loader)
    run_adp_pic_tests_hila(vit_for_image_classification=vit_for_classification_image, images_and_masks=images_and_masks,
                           ADP_PIC_config=ADP_PIC_config)
