import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.functional import F
from transformers import ViTForImageClassification
from feature_extractor import ViTFeatureExtractor
from models.modeling_vit import ViTBasicForForImageClassification
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple, Union, NewType, List, Optional
from pathlib import Path, WindowsPath
from utils.consts import PLOTS_PATH
from utils import save_obj_to_disk
from config import config
from torch import optim
from utils.consts import IMAGES_FOLDER_PATH
from utils.transformation import image_transformations, wolf_image_transformations
from utils.utils_functions import get_image_from_path

cuda = torch.cuda.is_available()
ce_loss = nn.CrossEntropyLoss(reduction="mean")

vit_config = config["vit"]

VitModelForClassification = NewType("VitModelForClassification", ViTForImageClassification)

vit_model_types = {
    "vit-basic": ViTBasicForForImageClassification,
}


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
    transformer_attribution = transformer_attribution.reshape(1, int(vit_config["img_size"] / vit_config["patch_size"]),
                                                              int(vit_config["img_size"] / vit_config["patch_size"]))
    transformer_attribution = torch.nn.functional.interpolate(
        transformer_attribution.unsqueeze(0), scale_factor=vit_config["patch_size"], mode="bilinear"
    )
    transformer_attribution = transformer_attribution.reshape(vit_config["img_size"],
                                                              vit_config["img_size"]).data.cpu().numpy()
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
    plt.imsave(fname=Path(f"{file_name}.png"), dpi=600, arr=vis, format="png")


def freeze_all_model_params(model: VitModelForClassification) -> VitModelForClassification:
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_x_attention_params(model: VitModelForClassification) -> VitModelForClassification:
    for param in model.named_parameters():
        if param[0] == "vit.encoder.x_attention":
            param[1].requires_grad = True
    return model


def handle_model_freezing(model: VitModelForClassification) -> VitModelForClassification:
    model = freeze_all_model_params(model=model)
    model = unfreeze_x_attention_params(model=model)
    return model


def setup_model_config(model: VitModelForClassification) -> VitModelForClassification:
    model.config.output_scores = True
    model.config.output_attentions = True
    return model


def get_logits_for_image(
        model: VitModelForClassification, feature_extractor: ViTFeatureExtractor, image: Image
) -> Tensor:
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(
        **inputs
    )  # inputs['pixel_values].shape: [batch_Size, n_channels, height, width]
    logits = outputs.logits
    return logits


def get_pred_idx_from_logits(logits: Tensor) -> int:
    predicted_class_idx = logits.argmax(-1).item()
    return predicted_class_idx


def calculate_num_of_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_num_of_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def calculate_percentage_of_trainable_params(model) -> str:
    return f"{round(calculate_num_of_trainable_params(model) / calculate_num_of_params(model), 2) * 100}%"


def print_number_of_trainable_and_not_trainable_params(model) -> None:
    print(
        f"Number of params: {calculate_num_of_params(model)}, Number of trainable params: {calculate_num_of_trainable_params(model)}"
    )


def load_feature_extractor(vit_config: Dict, is_competitive_method_transforms: bool) -> ViTFeatureExtractor:
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        vit_config["model_name"], is_competitive_method_transforms=is_competitive_method_transforms
    )
    return feature_extractor


def load_ViTModel(vit_config: Dict, model_type: str) -> VitModelForClassification:
    model = vit_model_types[model_type].from_pretrained(
        vit_config["model_name"], output_hidden_states=True
    )
    return model


def load_feature_extractor_and_vit_model(
        vit_config: Dict, model_type: str, is_competitive_method_transforms: bool = False
) -> Tuple[ViTFeatureExtractor, ViTForImageClassification]:
    feature_extractor = load_feature_extractor(
        vit_config=vit_config, is_competitive_method_transforms=is_competitive_method_transforms
    )
    vit_model = load_vit_model_by_type(vit_config=vit_config, model_type=model_type)
    return feature_extractor, vit_model


def load_vit_model_by_type(vit_config: Dict, model_type: str):
    vit_model = handle_model_config_and_freezing_for_task(
        model=load_ViTModel(vit_config, model_type=model_type)
    )
    return vit_model


def handle_model_config_and_freezing_for_task(
        model: VitModelForClassification, freezing_transformer: bool = True
) -> VitModelForClassification:
    model = setup_model_config(model=model)
    if freezing_transformer:
        model = handle_model_freezing(model=model)
    return model


def freeze_multitask_model(model, freezing_classification_transformer: bool = True,
                           segmentation_transformer_n_first_layers_to_freeze: int = 0):
    if freezing_classification_transformer:
        for param in model.vit_for_classification_image.parameters():
            param.requires_grad = False

    modules = [model.vit_for_patch_classification.vit.embeddings, model.vit_for_patch_classification.vit.encoder.layer[
                                                                  :segmentation_transformer_n_first_layers_to_freeze]]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False
    return model


def create_folder(path: Path) -> Path:
    os.makedirs(name=path, exist_ok=True)
    return path


def get_image_and_inputs_and_transformed_image(
        feature_extractor: ViTFeatureExtractor,
        image_name: str = None,
        image=None,
        is_competitive_method_transforms: bool = False,
):
    if image is None and image_name is not None:
        image = get_image_from_path(Path(IMAGES_FOLDER_PATH, image_name))
    inputs = feature_extractor(images=image, return_tensors="pt")
    transformed_image = (
        wolf_image_transformations(image) if is_competitive_method_transforms else image_transformations(image)
    )
    return inputs, transformed_image


def setup_model_and_optimizer(model_name: str):
    vit_ours_model = handle_model_config_and_freezing_for_task(
        model=load_ViTModel(vit_config, model_type=model_name),
        freezing_transformer=vit_config["freezing_transformer"],
    )
    optimizer = optim.Adam([vit_ours_model.vit.encoder.x_attention], lr=vit_config["lr"])
    return vit_ours_model, optimizer


def get_warmup_steps_and_total_training_steps(
        n_epochs: int, train_samples_length: int, batch_size: int
) -> Tuple[int, int]:
    steps_per_epoch = train_samples_length // batch_size
    total_training_steps = steps_per_epoch * n_epochs
    warmup_steps = total_training_steps // 5
    return warmup_steps, total_training_steps


def normalize_losses(mask_loss_mul: float, prediction_loss_mul: float) -> Tuple[float, float]:
    s = mask_loss_mul + prediction_loss_mul
    mask_loss_mul_norm = mask_loss_mul / s
    pred_loss_mul_norm = prediction_loss_mul / s
    return mask_loss_mul_norm, pred_loss_mul_norm


def get_loss_multipliers(loss_config) -> Dict[str, float]:
    if loss_config["normalize"]:
        mask_loss_mul, prediction_loss_mul = normalize_losses(mask_loss_mul=loss_config["mask_loss_mul"],
                                                              prediction_loss_mul=loss_config["prediction_loss_mul"])
    else:
        prediction_loss_mul = loss_config["prediction_loss_mul"]
        mask_loss_mul = loss_config["mask_loss_mul"]
    return dict(prediction_loss_mul=prediction_loss_mul, mask_loss_mul=mask_loss_mul)


def get_checkpoint_idx(ckpt_path: str) -> int:
    return int(str(ckpt_path).split("epoch=")[-1].split("_val")[0]) + 1


def get_ckpt_model_auc(ckpt_path: str) -> float:
    return float(str(ckpt_path).split("epoch_auc=")[-1].split(".ckpt")[0])
