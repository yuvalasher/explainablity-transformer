import os
from typing import List, Union

from torch.utils.data import DataLoader

from cnn_baselines.imagenet_dataset_cnn_baselines import ImageNetDataset
from main.seg_classification.cnns.cnn_utils import convnet_preprocess

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from pathlib import Path
import torch
from torch import Tensor
from tqdm import tqdm
from matplotlib import pyplot as plt
from cnn_baselines.grad_methods_utils import run_by_class_grad
from cnn_baselines.saliency_models import GradModel, ReLU, lift_cam, ig_captum, generic_torchcam
from utils import get_gt_classes, get_preprocessed_image, show_image
from utils.consts import GT_VALIDATION_PATH_LABELS, IMAGENET_VAL_IMAGES_FOLDER_PATH
from cnn_baselines.torchgc.pytorch_grad_cam.fullgrad_cam import FullGrad
from cnn_baselines.torchgc.pytorch_grad_cam.layer_cam import LayerCAM
from cnn_baselines.torchgc.pytorch_grad_cam.score_cam import ScoreCAM
from cnn_baselines.torchgc.pytorch_grad_cam.ablation_cam import AblationCAM
import h5py
import numpy as np

device = torch.device('cuda')
USE_MASK = True

backbones = ['densenet', 'resnet101']
FEATURE_LAYER_NUMBER_BY_BACKBONE = {'resnet101': 8, 'densenet': 12}


def compute_saliency_and_save(dir: Path,
                              method: str,
                              dataloader: DataLoader,
                              vis_class: str,
                              ):
    first = True
    with h5py.File(os.path.join(dir, 'results.hdf5'), 'a') as f:
        data_cam = f.create_dataset('vis',
                                    (1, 1, 224, 224),
                                    maxshape=(None, 1, 224, 224),
                                    dtype=np.float32,
                                    compression="gzip")
        data_image = f.create_dataset('image',
                                      (1, 3, 224, 224),
                                      maxshape=(None, 3, 224, 224),
                                      dtype=np.float32,
                                      compression="gzip")
        data_target = f.create_dataset('target',
                                       (1,),
                                       maxshape=(None,),
                                       dtype=np.int32,
                                       compression="gzip")

        for batch_idx, (data, target, image_without_transform) in enumerate(tqdm(dataloader)):
            if first:
                first = False
                data_cam.resize(data_cam.shape[0] + data.shape[0] - 1, axis=0)
                data_image.resize(data_image.shape[0] + data.shape[0] - 1, axis=0)
                data_target.resize(data_target.shape[0] + data.shape[0] - 1, axis=0)
            else:
                data_cam.resize(data_cam.shape[0] + data.shape[0], axis=0)
                data_image.resize(data_image.shape[0] + data.shape[0], axis=0)
                data_target.resize(data_target.shape[0] + data.shape[0], axis=0)

            # Add data
            data_image[-data.shape[0]:] = data.data.cpu().numpy()
            data_target[-data.shape[0]:] = target.data.cpu().numpy()

            data = data.to(device)
            data.requires_grad_()

            input_predictions = model(data.to(device), hook=False).detach()
            predicted_label = torch.max(input_predictions, 1).indices[0].item()

            index = predicted_label
            if vis_class == 'target':
                index = target
            index = index.to(device)

            if method == 'lift-cam':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = lift_cam(
                    model=model,
                    inputs=data,
                    label=index,
                    device=device,
                    use_mask=USE_MASK,
                )

            elif method == 'score-cam':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = generic_torchcam(
                    modelCAM=ScoreCAM,
                    backbone_name=backbone_name,
                    inputs=data,
                    label=index,
                    device=device,
                    use_mask=USE_MASK,
                )

            elif method == 'ablation-cam':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = generic_torchcam(
                    modelCAM=AblationCAM,
                    backbone_name=backbone_name,
                    inputs=data,
                    label=index,
                    device=device,
                    use_mask=USE_MASK,
                )

            elif method == 'ig':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = ig_captum(
                    model=model,
                    inputs=data,
                    label=index,
                    device=device,
                    use_mask=USE_MASK,
                )

            elif method == 'layercam':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = generic_torchcam(
                    modelCAM=LayerCAM,
                    backbone_name=backbone_name,
                    inputs=data,
                    label=index,
                    device=device,
                    use_mask=USE_MASK,
                )

            elif method == 'fullgrad':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = generic_torchcam(
                    modelCAM=FullGrad,
                    backbone_name=backbone_name,
                    inputs=data,
                    label=index,
                    device=device,
                    use_mask=USE_MASK,
                )

            elif method in ['gradcam', 'gradcampp']:
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = run_by_class_grad(model=model,
                                                                                                       image_preprocessed=data.squeeze(
                                                                                                           0),
                                                                                                       label=index,
                                                                                                       backbone_name=backbone_name,
                                                                                                       device=device,
                                                                                                       features_layer=FEATURE_LAYER_NUMBER,
                                                                                                       method=method,
                                                                                                       use_mask=USE_MASK,
                                                                                                       )
            else:
                raise NotImplementedError
            show_image(blended_im, title=method)
            data_cam[-data.shape[0]:] = heatmap


if __name__ == '__main__':
    BY_MAX_CLASS = False  # predicted / TARGET
    vis_class = "top" if BY_MAX_CLASS else "target"
    # methods = ['lift-cam', 'layercam', 'ig', 'ablation-cam', 'fullgrad', 'gradcam', 'gradcampp']
    method = 'gradcam'

    backbone_name = backbones[1]
    FEATURE_LAYER_NUMBER = FEATURE_LAYER_NUMBER_BY_BACKBONE[backbone_name]
    PREV_LAYER = FEATURE_LAYER_NUMBER - 1
    num_layers_options = [1]

    torch.nn.modules.activation.ReLU.forward = ReLU.forward

    model = GradModel(backbone_name, feature_layer=FEATURE_LAYER_NUMBER)
    model.to(device)
    model.eval()
    model.zero_grad()

    BASELINE_RESULTS_PATH = '/raid/yuvalas/baselines_results'
    os.makedirs(Path(BASELINE_RESULTS_PATH, 'visualizations'), exist_ok=True)
    dir_path = Path(BASELINE_RESULTS_PATH, f'visualizations/{method}/{vis_class}')
    os.makedirs(dir_path, exist_ok=True)
    if os.path.exists(Path(dir_path, 'results.hdf5')):
        os.remove(Path(dir_path, 'results.hdf5'))
    print(dir_path)

    imagenet_ds = ImageNetDataset(root_dir=IMAGENET_VAL_IMAGES_FOLDER_PATH,
                                  transform=convnet_preprocess,
                                  list_of_images_names=list(range(10)),
                                  )

    sample_loader = DataLoader(
        imagenet_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    compute_saliency_and_save(dir=dir_path,
                              method=method,
                              dataloader=sample_loader,
                              vis_class=vis_class,
                              )
