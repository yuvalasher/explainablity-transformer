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
from main.seg_classification.images_with_pic import infer_pic
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


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
        Res_patches = lrp.generate_LRP(data, start_layer=1, method="grad", index=target).reshape(data.shape[0], 1, 14,
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
        yield dict(image_resized=resized_image.to(device), image_mask=Res.to(device), target_class=target.to(device))
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


def run_pic_hila(vit_for_image_classification, images_and_masks):
    gt_classes_list = get_gt_classes(GT_VALIDATION_PATH_LABELS)
    evaluation_metrics = infer_pic(vit_for_image_classification=vit_for_image_classification,
                                   images_and_masks=images_and_masks,
                                   gt_classes_list=gt_classes_list,
                                   is_hila=True
                                   )


def run_perturbation_for_specific_indices(image_indices: List[int]):
    n_samples = len(image_indices)
    imagenet_ds = ImageNetDataset(root_dir=IMAGENET_VALIDATION_PATH, n_samples=n_samples,
                                  list_of_images_names=image_indices, transform=transform)
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

        print(
            f"hila, image: {idx}, auc_perturbation:{auc}, gt_class: {get_gt_classes(path=GT_VALIDATION_PATH_LABELS)[image_indices[idx]]}")


def run_perturbation_for_specific_indices_and_plot_by_condition(image_indices: List[int]):
    hila_worst_than_us_indices = []
    HILA_AUC_WORST_THAN_IMAGE_INDICES_PATH = "/home/yuvalas/explainability/pickles/hila_image_indices_worst_than_us.pkl"
    n_samples = len(image_indices)
    image_indices = sorted(image_indices)
    imagenet_ds = ImageNetDataset(root_dir=IMAGENET_VALIDATION_PATH, n_samples=n_samples,
                                  list_of_images_names=image_indices, transform=transform)
    sample_loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    outputs = compute_saliency_generator(dataloader=sample_loader)
    for idx, output in tqdm(enumerate(outputs)):
        auc = eval_perturbation_test(experiment_dir=Path(""), model=vit_for_classification_image, outputs=[output])
        if auc > 14:
            visualized_image = plot_visualization(original_image=output["image_resized"],
                                                  transformer_attribution=output["image_mask"])
            hila_worst_than_us_indices.append(image_indices[idx])
            save_image_try(image=visualized_image, image_idx=image_indices[idx])
    print(hila_worst_than_us_indices)
    save_obj_to_disk(path=HILA_AUC_WORST_THAN_IMAGE_INDICES_PATH, obj=hila_worst_than_us_indices)
    # print(
    #     f"hila, image: {idx}, auc_perturbation:{auc}, gt_class: {get_gt_classes(path=GT_VALIDATION_PATH_LABELS)[image_indices[idx]]}")


def plot_visualization(original_image, transformer_attribution):
    """
    original_image.shape:
    vis.shape: (224,224)
    """
    # transformer_attribution = vis.reshape(1, 1, 14, 14)
    # transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16,
    #                                                           mode='bilinear')
    # transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
    #         transformer_attribution.max() - transformer_attribution.min())
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    image_transformer_attribution = original_image.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
            image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def save_image_try(image, image_idx: int) -> None:  # [1,3,224,224] or [3,224,224]
    image = image if len(image.shape) == 3 else image.squeeze(0)
    image = transforms.ToPILImage()(image)
    image = image.resize((224, 224))
    plt.imshow(transforms.ToTensor()(image).permute(1, 2, 0))
    plt.axis('off');
    # plt.show();
    path = f"/home/yuvalas/explainability/research/plots/additional/hila1_worst_than_us/{image_idx}_hila_1.png"
    plt.margins(x=0, y=0)
    plt.savefig(path, dpi=300,
                bbox_inches='tight', pad_inches=0, transparent=True)


if __name__ == '__main__':
    BATCH_SIZE = 1
    MODEL_NAME = config["vit"]["model_name"]
    model_LRP = vit_LRP(pretrained=True).to(device)
    model_LRP.eval()
    lrp = LRP(model_LRP)

    vit_for_classification_image, _ = load_vit_pretrained(model_name=MODEL_NAME)
    vit_for_classification_image = vit_for_classification_image.to(device)
    """
    images_indices = sorted([17721, 16167, 7436, 11616])
    imagenet_ds = ImageNetDataset(root_dir=IMAGENET_VALIDATION_PATH, n_samples=len(images_indices),
                                  list_of_images_names=images_indices, transform=transform)
    sample_loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    images_and_masks_generator = compute_saliency_generator(dataloader=sample_loader)
    
    # run_pic_hila(vit_for_image_classification=vit_for_classification_image, images_and_masks=images_and_masks_generator)
    """
    IMAGES_IDX_BY_AUC_DIFF_BASE_OPT_MODEL_PATH = "/home/yuvalas/explainability/pickles/images_idx_by_auc_diff_base_opt_model.pkl"
    HILA_AUC_BY_IMG_IDX_PATH = "/home/yuvalas/explainability/pickles/hila_auc_by_img_idx.pkl"
    # PLOTS_OUTPUT_PATH = "/home/yuvalas/explainability/pickles/comparison_base_opt_models"
    IMAGES_IDX_BY_AUC_TARGET_OPT_MODEL_PATH = "/home/yuvalas/explainability/pickles/images_idx_by_auc_target_opt.pkl"
    images_idx_by_auc_target_opt = load_obj(path=IMAGES_IDX_BY_AUC_TARGET_OPT_MODEL_PATH)

    image_indices = []
    for key in images_idx_by_auc_target_opt.keys():
        if key in [5]:
            image_indices.extend(images_idx_by_auc_target_opt[key])
    image_indices = image_indices
    print(len(image_indices))
    n_samples = len(image_indices)
    run_perturbation_for_specific_indices_and_plot_by_condition(image_indices=sorted(image_indices))

    """
    imagenet_ds = ImageNetDataset(root_dir=IMAGENET_VALIDATION_PATH, n_samples=n_samples,
                                  list_of_images_names=sorted(image_indices), transform=transform)
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
        output["image_idx"] = image_indices[idx]
        # ic(images_indices[idx])
        hila_auc_by_img_idx[image_indices[idx]] = auc
    save_obj_to_disk(path=HILA_AUC_BY_IMG_IDX_PATH, obj=hila_auc_by_img_idx)
    """
    # print(f'AUC: {round(auc, 4)}% for {len(outputs)} records')
    # for key in images_idx_by_auc_diff_base_opt_model.keys():
    #     if image_idx in images_idx_by_auc_diff_base_opt_model[key]:
    #         plot_subfolder_name = f"auc_diff_{key}"
    # load_obj(HILA_AUC_BY_IMG_IDX_PATH)
    # visualize_outputs(outputs=outputs)

    # ADP & PIC
    # images_and_masks_generator = compute_saliency_generator(dataloader=sample_loader)
    # run_adp_pic_tests_hila(vit_for_image_classification=vit_for_classification_image, images_and_masks=images_and_masks_generator)
