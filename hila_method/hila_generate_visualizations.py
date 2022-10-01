from icecream import ic
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluation.perturbation_tests.seg_cls_perturbation_tests import run_perturbation_test
from hila_method.utils.ViT_LRP import vit_base_patch16_224 as vit_LRP
from hila_method.utils.ViT_explanation_generator import LRP
from hila_method.utils.imagenet_dataset import ImageNetDataset
import torch
from vit_loader.load_vit import load_vit_pretrained

device = torch.device(type='cuda', index=0)
cuda = torch.cuda.is_available()
torch.cuda.empty_cache()
from torchvision.transforms import transforms
from pytorch_lightning import seed_everything
from config import config

seed_everything(config["general"]["seed"])
import numpy as np
# import torch
from torch import Tensor
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List
from pathlib import Path

HILA_VISUAILZATION_PATH = "/home/yuvalas/explainability/research/plots/hila"


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
        original_image = data.clone()
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
        outputs.append({'image_resized': original_image, 'image_mask': Res, 'patches_mask': Res_patches})
    return outputs


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


if __name__ == '__main__':
    IMAGENET_VALIDATION_PATH = '/home/yuvalas/explainability/data/ILSVRC2012_test_earlystopping'
    BATCH_SIZE = 1
    MODEL_NAME = 'google/vit-base-patch16-224'
    model_LRP = vit_LRP(pretrained=True).to(device)  # .cuda()
    model_LRP.eval()
    lrp = LRP(model_LRP)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    vit_for_classification_image, _ = load_vit_pretrained(model_name=MODEL_NAME)
    # n_samples = config["vit"]["seg_cls"]["val_n_samples"]
    n_samples = 208
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

    auc = run_perturbation_test(
        model=vit_for_classification_image,
        outputs=outputs,
        stage="hila",
        epoch_idx=0,
    )
