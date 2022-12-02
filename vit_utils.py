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


def plot_vis_on_image(original_image, mask, file_name: str):
    """
    :param original_image.shape: [3, 224, 224]
    :param mask.shape: [1,1, 224, 224]:
    """
    mask = mask.data.squeeze(0).squeeze(0).cpu().numpy()  # [1,1,224,224]
    # mask = torch.tensor(mask.data).squeeze(0).squeeze(0).numpy()  # [1,1,224,224]
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    original_image = original_image.squeeze(0) if len(original_image.shape) == 4 else original_image
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
            image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(img=image_transformer_attribution, mask=mask)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    # plt.axis('off')
    plt.imsave(fname=Path(f"{file_name}.png"), dpi=300, arr=vis, format="png")


def visu(original_image, transformer_attribution, file_name: str, img_size: int, patch_size: int):
    """
    :param original_image: shape: [3, 224, 224]
    :param transformer_attribution: shape: [n_patches, n_patches] = [14, 14]
    :param file_name:
    :return:
    """
    if type(transformer_attribution) == np.ndarray:
        transformer_attribution = torch.tensor(transformer_attribution)
    transformer_attribution = transformer_attribution.reshape(1, int(img_size / patch_size),
                                                              int(img_size / patch_size))
    transformer_attribution = torch.nn.functional.interpolate(
        transformer_attribution.unsqueeze(0), scale_factor=patch_size, mode="bilinear"
    )
    transformer_attribution = transformer_attribution.reshape(img_size, img_size).data.cpu().numpy()
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


def freeze_multitask_model(model,
                           freezing_classification_transformer: bool = True,
                           segmentation_transformer_n_first_layers_to_freeze: int = 0,
                           is_explainer_convnet: bool = False):
    if freezing_classification_transformer:
        for param in model.vit_for_classification_image.parameters():
            param.requires_grad = False
    if not is_explainer_convnet:
        modules = [model.vit_for_patch_classification.vit.embeddings,
                   model.vit_for_patch_classification.vit.encoder.layer[
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


def get_loss_multipliers(normalize: bool, mask_loss_mul: int, prediction_loss_mul: int) -> Dict[str, float]:
    if normalize:
        mask_loss_mul, prediction_loss_mul = normalize_losses(mask_loss_mul=mask_loss_mul,
                                                              prediction_loss_mul=prediction_loss_mul)
    else:
        prediction_loss_mul = prediction_loss_mul
        mask_loss_mul = mask_loss_mul
    return dict(prediction_loss_mul=prediction_loss_mul, mask_loss_mul=mask_loss_mul)


def get_checkpoint_idx(ckpt_path: str) -> int:
    return int(str(ckpt_path).split("epoch=")[-1].split("_val")[0]) + 1


def get_ckpt_model_auc(ckpt_path: str) -> float:
    return float(str(ckpt_path).split("epoch_auc=")[-1].split(".ckpt")[0])


def get_params_from_vit_config(vit_config: Dict):
    loss_config = vit_config["seg_cls"]["loss"]
    batch_size = vit_config["batch_size"]
    n_epochs = vit_config["n_epochs"]
    is_sampled_train_data_uniformly = vit_config["is_sampled_train_data_uniformly"]
    is_sampled_val_data_uniformly = vit_config["is_sampled_val_data_uniformly"]
    train_model_by_target_gt_class = vit_config["train_model_by_target_gt_class"]
    freezing_classification_transformer = vit_config["freezing_classification_transformer"]
    segmentation_transformer_n_first_layers_to_freeze = vit_config["segmentation_transformer_n_first_layers_to_freeze"]
    is_clamp_between_0_to_1 = vit_config["is_clamp_between_0_to_1"]
    enable_checkpointing = vit_config["enable_checkpointing"]
    is_competitive_method_transforms = vit_config["is_competitive_method_transforms"]
    explainer_model_name = vit_config["explainer_model_name"]
    explainee_model_name = vit_config["explainee_model_name"]
    plot_path = vit_config["plot_path"]
    default_root_dir = vit_config["default_root_dir"]
    train_n_samples = vit_config["seg_cls"]["train_n_label_sample"]
    mask_loss = loss_config["mask_loss"]
    mask_loss_mul = loss_config["mask_loss_mul"]
    prediction_loss_mul = loss_config["prediction_loss_mul"]
    lr = vit_config['lr']
    start_epoch_to_evaluate = vit_config["start_epoch_to_evaluate"]
    n_batches_to_visualize = vit_config["n_batches_to_visualize"]
    is_ce_neg = loss_config["is_ce_neg"]
    activation_function = vit_config["activation_function"]
    n_epochs_to_optimize_stage_b = vit_config["n_epochs_to_optimize_stage_b"]
    RUN_BASE_MODEL = vit_config["run_base_model"]
    use_logits_only = loss_config["use_logits_only"]
    VERBOSE = vit_config["verbose"]
    IMG_SIZE = vit_config["img_size"]
    PATCH_SIZE = vit_config["patch_size"]
    return batch_size, n_epochs, is_sampled_train_data_uniformly, is_sampled_val_data_uniformly, \
           train_model_by_target_gt_class, freezing_classification_transformer, \
           segmentation_transformer_n_first_layers_to_freeze, is_clamp_between_0_to_1, enable_checkpointing, \
           is_competitive_method_transforms, explainer_model_name, explainee_model_name, plot_path, default_root_dir, \
           train_n_samples, mask_loss, mask_loss_mul, prediction_loss_mul, lr, start_epoch_to_evaluate, n_batches_to_visualize, \
           is_ce_neg, activation_function, n_epochs_to_optimize_stage_b, RUN_BASE_MODEL, use_logits_only, VERBOSE, IMG_SIZE, PATCH_SIZE
