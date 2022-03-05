import re
from config import config
import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.functional import F
from transformers import ViTForImageClassification
from feature_extractor import ViTFeatureExtractor
from models.modeling_vit_sigmoid import ViTSigmoidForImageClassification
from models.modeling_dino_vit import ViTBasicForDinoForImageClassification
from models.modeling_infer_vit import ViTInferForImageClassification
from models.vit_sigmoid_mask_head_layer import ViTSigmoidPerLayerHeadForImageClassification
from models.modeling_temp_softmax_vit import ViTTempSoftmaxForImageClassification
from models.modeling_temp_bias_softmax_vit import ViTTempBiasSoftmaxForImageClassification
from models.modeling_vit_gumble_resolutions import ViTGumbleResolutionsForImageClassification
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple, Union, NewType, List
from pathlib import Path, WindowsPath
from utils.consts import PLOTS_PATH
from utils import save_obj_to_disk

VitModelForClassification = NewType('VitModelForClassification',
                                    Union[ViTSigmoidForImageClassification, ViTForImageClassification])
vit_model_types = {'vit': ViTForImageClassification,
                   'vit-sigmoid': ViTSigmoidForImageClassification,
                   'vit-for-dino': ViTBasicForDinoForImageClassification,
                   'infer': ViTInferForImageClassification,
                   'per-layer-head': ViTSigmoidPerLayerHeadForImageClassification,
                   'softmax_temp': ViTTempSoftmaxForImageClassification,
                   'softmax_bias_temp': ViTTempBiasSoftmaxForImageClassification,
                   'gumble_resolutions': ViTGumbleResolutionsForImageClassification,
                   }


def get_head_num_heads(model) -> int:
    return model.vit.encoder.layer[-1].attention.attention.attention_probs.shape[1]


def get_attention_probs(model) -> List[Tensor]:
    """

    :return: Tensor of size (num_heads, num_tokens)
    """
    num_heads = get_head_num_heads(model=model)
    attentions = [model.vit.encoder.layer[head].attention.attention.attention_probs for head in range(num_heads)]
    return attentions


def get_attention_probs_by_layer_of_the_CLS(model, layer: int = -1) -> Tensor:
    """

    :return: Tensor of size (num_heads, num_tokens)
    """
    num_heads = get_head_num_heads(model=model)
    attentions = model.vit.encoder.layer[layer].attention.attention.attention_probs[0, :, 0, 1:].reshape(
        num_heads, -1)
    return attentions


def dino_method_attention_probs_cls_on_tokens_last_layer(vit_sigmoid_model: ViTSigmoidForImageClassification,
                                                         path: Union[str, WindowsPath, Path],
                                                         image_size: int = config['vit']['img_size'],
                                                         patch_size: int = config['vit']['patch_size']) -> None:
    image_dino_plots_folder = Path(path, 'dino')
    os.makedirs(image_dino_plots_folder, exist_ok=True)
    num_heads = get_head_num_heads(model=vit_sigmoid_model)
    attentions = get_attention_probs_by_layer_of_the_CLS(model=vit_sigmoid_model)
    save_obj_to_disk(path=Path(image_dino_plots_folder, 'attentions.pkl'), obj=attentions)
    plot_attn_probs(attentions=attentions, image_size=image_size, patch_size=patch_size, num_heads=num_heads,
                    path=image_dino_plots_folder, only_fusion=False)


def plot_attention_rollout(attention_probs, path, patch_size: int, iteration_idx: int,
                           head_fusion: str = 'max') -> None:
    image_rollout_plots_folder = Path(path, 'rollout')
    os.makedirs(image_rollout_plots_folder, exist_ok=True)
    mask_rollout = rollout(attentions=attention_probs, head_fusion=head_fusion)
    attention_rollout_original_size = \
        nn.functional.interpolate(torch.tensor(mask_rollout).unsqueeze(0).unsqueeze(0), scale_factor=patch_size,
                                  mode="nearest")[0].cpu().detach().numpy()
    plt.imsave(fname=Path(image_rollout_plots_folder, f'{head_fusion}_rollout_iter_{iteration_idx}.png'),
               arr=attention_rollout_original_size[0],
               format='png')


def plot_attn_probs(attentions: Tensor, image_size: int, patch_size: int, num_heads: int, path: Path,
                    iteration_idx: int = None, only_fusion: bool = True) -> None:
    w_featmap, h_featmap = image_size // patch_size, image_size // patch_size
    attentions = attentions.reshape(num_heads, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[
        0].cpu().detach().numpy()

    # plt.imsave(fname=Path(path, f'iter_{iteration_idx}_max_fusion.png'), arr=attentions.max(axis=0), format='png')
    # plt.imsave(fname=Path(path, f'iter_{iteration_idx}_min_fusion.png'), arr=attentions.min(axis=0), format='png')
    plt.imsave(fname=Path(path, f'iter_{iteration_idx}_mean_fusion.png'), arr=attentions.mean(axis=0), format='png')

    if not only_fusion:
        for head_idx in range(num_heads):
            filename = f'attn-head{head_idx}.png' if iteration_idx is None else f'iter_{iteration_idx}_attn-head{head_idx}.png'
            plt.imsave(fname=Path(path, filename), arr=attentions[head_idx], format='png')


def get_scores(scores: torch.Tensor, image_size: int = config['vit']['img_size'],
               patch_size: int = config['vit']['patch_size']) -> None:
    num_patches = (image_size // patch_size) * (image_size // patch_size)

    if len(scores.shape) == 1:
        scores = scores.unsqueeze(0)
    if scores.shape[-1] == num_patches + 1:
        scores = scores[:, 1:]  # not include the cls token

    w_featmap, h_featmap = image_size // patch_size, image_size // patch_size
    scores = scores.reshape(1, w_featmap, h_featmap)
    scores = nn.functional.interpolate(scores.unsqueeze(0), scale_factor=patch_size, mode="bilinear")[
        0].cpu().detach().numpy()
    scores_image = scores[0]
    #     plt.imsave(fname=Path(image_plot_folder_path, f'{file_name.replace(" ", "")[:45]}_iter_{iteration_idx}.png'), arr=scores_image,
    #                format='png')
    #     plt.imshow(scores_image, interpolation='nearest')
    #     plt.show()
    return scores_image


def save_saliency_map(image: Tensor, saliency_map: Tensor, filename: Path, verbose: bool = True,
                      image_size: int = 224) -> None:
    """
    Save saliency map on image.
    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W)
        filename: string with complete path and file extension
    """
    image = image.data.numpy()
    image = np.uint8(image * 255).transpose(1, 2, 0)
    image = cv2.resize(image, (image_size, image_size))

    heatmap = saliency_map

    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / heatmap.max()
    heatmap = heatmap.clip(0, 1)

    heatmap = np.uint8(heatmap * 255).transpose(1, 2, 0)
    heatmap = cv2.resize(heatmap, (image_size, image_size))

    # Apply JET colormap
    color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # / 255

    # Combine image with heatmap
    img_with_heatmap = np.float32(color_heatmap) + np.float32(image)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    if verbose:
        plt.imshow(img_with_heatmap, interpolation='nearest')
        plt.show()
    cv2.imwrite(f'{filename.resolve()}.png', np.uint8(255 * img_with_heatmap))


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


def load_feature_extractor_and_vit_model(vit_config: Dict) -> Tuple[
    ViTFeatureExtractor, ViTForImageClassification]:
    feature_extractor = load_feature_extractor(vit_config=vit_config)
    # vit_model, vit_sigmoid_model = load_handled_models_for_task(vit_config=vit_config)
    vit_model = load_vit_model_by_type(vit_config=vit_config, model_type='vit')
    return feature_extractor, vit_model


def load_vit_model_by_type(vit_config: Dict, model_type: str):
    vit_model = handle_model_config_and_freezing_for_task(model=load_ViTModel(vit_config, model_type=model_type))
    return vit_model


def handle_model_config_and_freezing_for_task(model: VitModelForClassification,
                                              freezing_transformer: bool = True) -> VitModelForClassification:
    model = setup_model_config(model=model)
    if freezing_transformer:
        model = handle_model_freezing(model=model)
    return model


# def load_handled_models_for_task(vit_config: Dict, freezing_transformer: bool=True) -> ViTSigmoidForImageClassification:
# vit_sigmoid_model = load_ViTModel(vit_config, model_type='vit-sigmoid')
# if freezing_transformer:
#     vit_sigmoid_model = handle_model_config_and_freezing_for_task(model=vit_sigmoid_model)
# return vit_sigmoid_model


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
                image_size: int = 224, patch_size: int = 16) -> None:
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


def save_resized_original_picture(image_size, picture_path, dst_path) -> None:
    Image.open(picture_path).resize((image_size, image_size)).save(
        Path(dst_path, f"{image_size}x{image_size}.JPEG"), "JPEG")


def check_stop_criterion(x_attention: Tensor) -> bool:
    if len(torch.where(F.sigmoid(x_attention) >= float(config['vit']['stop_prob_criterion']))[0]) == 0:
        return True
    return False


def get_and_create_image_plot_folder_path(images_folder_path: Path, experiment_name: str, image_name: str,
                                          is_contrastive_run: bool = False) -> Path:
    """
    Also saving the original picture in the models' resolution (img_size, img_size)
    """
    print(image_name)
    image_plot_folder_path = Path(PLOTS_PATH, experiment_name, f'{image_name.replace(".JPEG", "")}')
    if is_contrastive_run:
        image_plot_folder_path = Path(image_plot_folder_path, 'contrastive')
    os.makedirs(name=image_plot_folder_path, exist_ok=True)
    save_resized_original_picture(image_size=config['vit']['img_size'],
                                  picture_path=Path(images_folder_path, image_name),
                                  dst_path=Path(PLOTS_PATH, experiment_name, f'{image_name.replace(".JPEG", "")}'))
    return image_plot_folder_path


def get_vector_to_print(model, vit_config: Dict):
    return model.vit.encoder.sampled_binary_patches if vit_config['objective'] in vit_config[
        'gumble_objectives'] else F.relu(model.vit.encoder.x_attention)


def get_top_k_mimimum_values_indices(array: List[float], k: int = 5):
    return torch.topk(torch.tensor(array), k=min(len(array), k), largest=False)[1]


def get_patches_by_discard_ratio(array: Tensor, discard_ratio: float, top: bool = True) -> Tensor:
    k = int(array.shape[-1] * discard_ratio)
    _, indices = torch.topk(array, k, largest=bool(1 - top))
    array[indices] = 0
    return array


def rollout(attentions, discard_ratio: float = 0.9, head_fusion: str = 'max'):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise ("Attention head fusion type Not supported")

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0, 1:]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask


def get_minimum_predictions_string(image_name: str, total_losses, prediction_losses, k: int = 10) -> str:
    return f'{image_name} - Minimum prediction_loss at iteration: {get_top_k_mimimum_values_indices(array=prediction_losses, k=k)}\n {image_name} - Minimum total loss at iteration: {get_top_k_mimimum_values_indices(array=total_losses, k=k)}'


def js_kl(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def kl_div(p: Tensor, q: Tensor, eps: float = 1e-7):
    q += eps
    q /= torch.sum(q, axis=1, keepdims=True)
    mask = p > 0
    return torch.sum(p[mask] * torch.log(p[mask] / q[mask])) / len(p)


def convert_probability_vector_to_bernoulli_kl(p) -> Tensor:
    bernoulli_p = torch.stack((p, 1 - p)).T
    return bernoulli_p


def compare_between_predicted_classes(vit_logits: Tensor, vit_s_logits: Tensor, contrastive_class_idx: Tensor = None) -> \
        Tuple[bool, float]:
    target_class_idx = contrastive_class_idx.item() if contrastive_class_idx is not None else torch.argmax(
        vit_logits[0]).item()
    original_idx_logits_diff = (abs(max(vit_logits[0]).item() - vit_s_logits[0][target_class_idx].item()))
    is_predicted_same_class = target_class_idx == torch.argmax(vit_s_logits[0]).item()
    return is_predicted_same_class, original_idx_logits_diff


def compare_results_each_n_steps(iteration_idx: int, target: Tensor, output: Tensor, prev_x_attention: Tensor,
                                 sampled_binary_patches: Tensor = None, contrastive_class_idx: Tensor = None):
    is_predicted_same_class, original_idx_logits_diff = compare_between_predicted_classes(
        vit_logits=target, vit_s_logits=output, contrastive_class_idx=contrastive_class_idx)
    print(
        f'Is predicted same class: {is_predicted_same_class}, Correct Class Prob: {F.softmax(output, dim=-1)[0][contrastive_class_idx.item()]}')
    if is_predicted_same_class is False:
        print(f'Predicted class change at {iteration_idx} iteration !!!!!!')
    # if is_iteration_to_action(iteration_idx=iteration_idx, action='print'):
    # print('temp')
    # print(prev_x_attention)


def save_model(model: nn.Module, path: str) -> None:
    path = Path(f'{path}.pt')
    torch.save(model.state_dict(), path)
    print(f'Model Saved at {path}')


# def load_model(path: str) -> nn.Module:
#     if path[-3:] == '.pt':
#         path = Path(f'{path}')
#     else:
#         path = Path(f'{path}.pt')
#     c = ViTConfig()
#     c.image_size = vit_config['img_size']
#     c.num_labels = vit_config['num_labels']
#     model = ViTSigmoidForImageClassification(config=c)
#     model.load_state_dict(torch.load(path))
#     return model


def save_objects(path: Path, objects_dict: Dict) -> None:
    for obj_name, obj in objects_dict.items():
        save_obj_to_disk(path=Path(path, obj_name), obj=obj)
