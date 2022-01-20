import torch
from torch import Tensor
from torch import nn
from transformers import ViTFeatureExtractor, ViTForImageClassification
from modeling_vit_sigmoid import ViTSigmoidForImageClassification
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Union, NewType
from pathlib import Path
import numpy as np
import cv2
from config import config

VitModelForClassification = NewType('VitModelForClassification',
                                    Union[ViTSigmoidForImageClassification, ViTForImageClassification])
vit_model_types = {'vit': ViTForImageClassification, 'vit-sigmoid': ViTSigmoidForImageClassification}


def get_scores(scores: torch.Tensor, image_size: int = config['vit']['img_size'],
               patch_size: int = config['vit']['patch_size']) -> None:
    num_patches = (image_size // patch_size) * (image_size // patch_size)

    if len(scores.shape) == 1:
        scores = scores.unsqueeze(0)
    if scores.shape[-1] == num_patches + 1:
        scores = scores[:, 1:]

    w_featmap, h_featmap = image_size // patch_size, image_size // patch_size
    scores = scores.reshape(1, w_featmap, h_featmap)
    scores = nn.functional.interpolate(scores.unsqueeze(0), scale_factor=patch_size, mode="nearest")[
        0].cpu().detach().numpy()
    scores_image = scores[0]
    #     plt.imsave(fname=Path(image_plot_folder_path, f'{file_name.replace(" ", "")[:45]}_iter_{iteration_idx}.png'), arr=scores_image,
    #                format='png')
    #     plt.imshow(scores_image, interpolation='nearest')
    #     plt.show()
    return scores_image


def save_saliency_map(image: Tensor, saliency_map: Tensor, filename: Union[str, Path], verbose: True) -> None:
    """
    Save saliency map on image.
    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W)
        filename: string with complete path and file extension
    """
    image = image.data.numpy()
    saliency_map = saliency_map

    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map = saliency_map.clip(0, 1)

    saliency_map = np.uint8(saliency_map * 255).transpose(1, 2, 0)
    saliency_map = cv2.resize(saliency_map, (224, 224))

    image = np.uint8(image * 255).transpose(1, 2, 0)
    image = cv2.resize(image, (224, 224))

    # Apply JET colormap
    color_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)

    # Combine image with heatmap
    img_with_heatmap = np.float32(color_heatmap) + np.float32(image)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    if verbose:
        plt.imshow(img_with_heatmap, interpolation='nearest')
        plt.show()
        cv2.imwrite(f'{filename}.png', np.uint8(255 * img_with_heatmap))


def freeze_all_model_params(model: VitModelForClassification) -> VitModelForClassification:
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_x_attention_params(model: VitModelForClassification) -> VitModelForClassification:
    for param in model.named_parameters():
        if param[0] == 'vit.encoder.x_attention':
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


def get_logits_for_image(model: VitModelForClassification, feature_extractor: ViTFeatureExtractor,
                         image: Image) -> Tensor:
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)  # inputs['pixel_values].shape: [batch_Size, n_channels, height, width]
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
    return f'{round(calculate_num_of_trainable_params(model) / calculate_num_of_params(model), 2) * 100}%'


def print_number_of_trainable_and_not_trainable_params(model: VitModelForClassification, model_name: str) -> None:
    print(
        f'{model_name} - Number of params: {calculate_num_of_params(model)}, Number of trainable params: {calculate_num_of_trainable_params(model)}')


def load_feature_extractor(vit_config: Dict) -> ViTFeatureExtractor:
    feature_extractor = ViTFeatureExtractor.from_pretrained(vit_config['model_name'])
    return feature_extractor


def load_ViTModel(vit_config: Dict, model_type: str) -> VitModelForClassification:
    model = vit_model_types[model_type].from_pretrained(vit_config['model_name'])
    return model


def load_feature_extractor_and_vit_models(vit_config: Dict) -> Tuple[
    ViTFeatureExtractor, ViTForImageClassification, ViTSigmoidForImageClassification]:
    feature_extractor = load_feature_extractor(vit_config=vit_config)
    vit_model, vit_sigmoid_model = load_handled_models_for_task(vit_config=vit_config)
    return feature_extractor, vit_model, vit_sigmoid_model


def handle_model_for_task(model: VitModelForClassification) -> VitModelForClassification:
    model = handle_model_freezing(model=setup_model_config(model=model))
    return model


def load_handled_models_for_task(vit_config: Dict) -> Tuple[
    VitModelForClassification, ViTSigmoidForImageClassification]:
    vit_model = handle_model_for_task(model=load_ViTModel(vit_config, model_type='vit'))
    vit_sigmoid_model = handle_model_for_task(model=load_ViTModel(vit_config, model_type='vit-sigmoid'))
    return vit_model, vit_sigmoid_model


def verify_transformer_params_not_changed(vit_model: ViTForImageClassification,
                                          vit_sigmoid_model: ViTSigmoidForImageClassification,
                                          x_attention_param_idx: int = 5):
    for idx, (tensor_a, tensor_b) in enumerate(zip(list(vit_sigmoid_model.parameters())[:x_attention_param_idx - 1],
                                                   list(vit_model.parameters())[:x_attention_param_idx - 1])):
        if not torch.equal(tensor_a, tensor_b):
            print(f'Not Equal at idx {idx}')
            return False

    for idx, (tensor_a, tensor_b) in enumerate(zip(list(vit_sigmoid_model.parameters())[x_attention_param_idx:],
                                                   list(vit_model.parameters())[x_attention_param_idx - 1:])):
        if not torch.equal(tensor_a, tensor_b):
            print(f'Not Equal at idx {idx}')
            return False
        return True


def plot_scores(scores: torch.Tensor, file_name: str, iteration_idx: int, image_plot_folder_path: Union[str, Path],
                image_size: int = 384,
                patch_size: int = 16) -> None:
    num_patches = (image_size // patch_size) * (image_size // patch_size)

    if len(scores.shape) == 1:
        scores = scores.unsqueeze(0)
    if scores.shape[-1] == num_patches + 1:
        scores = scores[:, 1:]

    w_featmap, h_featmap = image_size // patch_size, image_size // patch_size
    scores = scores.reshape(1, w_featmap, h_featmap)
    scores = nn.functional.interpolate(scores.unsqueeze(0), scale_factor=patch_size, mode="nearest")[
        0].cpu().detach().numpy()
    scores_image = scores[0]
    plt.imsave(fname=Path(image_plot_folder_path, f'{file_name.replace(" ", "")[:45]}_iter_{iteration_idx}.png'),
               arr=scores_image,
               format='png')
    plt.imshow(scores_image, interpolation='nearest')
    plt.show()
