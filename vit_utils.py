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
from models.modeling_temp_softmax_vit_only_head import ViTTempSoftmaxForHeadForImageClassification
from models.modeling_vit_gumble_resolutions import ViTGumbleResolutionsForImageClassification
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
from utils.transformation import image_transformations
from utils.utils_functions import get_image_from_path

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
ce_loss = nn.CrossEntropyLoss(reduction='mean')

vit_config = config['vit']
loss_config = vit_config['loss']

VitModelForClassification = NewType('VitModelForClassification',
                                    Union[ViTSigmoidForImageClassification, ViTForImageClassification])

vit_model_types = {'vit': ViTForImageClassification,
                   'vit-sigmoid': ViTSigmoidForImageClassification,
                   'vit-for-dino': ViTBasicForDinoForImageClassification,
                   'infer': ViTInferForImageClassification,
                   'per-layer-head': ViTSigmoidPerLayerHeadForImageClassification,
                   'softmax_temp': ViTTempSoftmaxForImageClassification,
                   'softmax_bias_temp': ViTTempBiasSoftmaxForImageClassification,
                   'softmax_for_head_temp': ViTTempSoftmaxForHeadForImageClassification,
                   'gumble_resolutions': ViTGumbleResolutionsForImageClassification,
                   }


def get_head_num_heads(model) -> int:
    return model.vit.encoder.layer[-1].attention.attention.attention_probs.shape[1]


def get_num_layers(model) -> int:
    return len(model.vit.encoder.layer)


def get_attention_probs(model) -> List[Tensor]:
    """

    :return: Tensor of size (num_heads, num_tokens)
    """
    num_layers = get_num_layers(model=model)
    attentions = [model.vit.encoder.layer[layer_idx].attention.attention.attention_probs for layer_idx in
                  range(num_layers)]
    return attentions


def get_attention_probs_by_layer_of_the_CLS(model, layer: int = -1) -> Tensor:
    """

    :return: Tensor of size (num_heads, num_tokens)
    """
    num_heads = get_head_num_heads(model=model)
    attentions = model.vit.encoder.layer[layer].attention.attention.attention_probs[0, :, 0, 1:].reshape(
        num_heads, -1)
    return attentions


def generate_grad_cam_vector(vit_sigmoid_model, cls_attentions_probs):
    grads = vit_sigmoid_model.vit.encoder.layer[-1].attention.attention.attention_probs.grad[0, :, 0, 1:].reshape(12,
                                                                                                                  -1)
    grads_relu = F.relu(grads)
    # v = cls_attentions_probs * grads_relu
    return grads_relu


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


def get_rollout_mask(fusions: List[str], gradients: List[Tensor] = None,
                     attention_probs=None) -> List[Tensor]:
    """
    Each mask is a [n_tokens] mask (after head aggregation)
    """
    if gradients is not None:
        attn = []
        for attn_score, grad in zip(attention_probs, gradients):
            attn.append(attn_score * grad)
    else:
        attn = attention_probs

    masks = []
    if 'mean' in fusions:
        mask_rollout_mean = rollout(attentions=attn, head_fusion='mean', return_resized=False)
        masks.append(mask_rollout_mean)
    if 'median' in fusions:
        mask_rollout_median = rollout(attentions=attn, head_fusion='median', return_resized=False)
        masks.append(mask_rollout_median)
    if 'min' in fusions:
        mask_rollout_min = rollout(attentions=attn, head_fusion='min', return_resized=False)
        masks.append(mask_rollout_min)
    if 'max' in fusions:
        mask_rollout_max = rollout(attentions=attn, head_fusion='max', return_resized=False)
        masks.append(mask_rollout_max)
    return masks


def plot_attention_rollout(attention_probs, path, patch_size: int, iteration_idx: int,
                           head_fusion: str = 'max', original_image=None) -> None:
    image_rollout_plots_folder = create_folder(Path(path, 'rollout'))
    mask_rollout = rollout(attentions=attention_probs, head_fusion=head_fusion)
    file_path = Path(image_rollout_plots_folder, f'{head_fusion}_rollout_iter_{iteration_idx}')
    if original_image is not None:
        visu(original_image=original_image, transformer_attribution=mask_rollout, file_name=file_path)
    else:
        attention_rollout_original_size = \
            nn.functional.interpolate(torch.tensor(mask_rollout).unsqueeze(0).unsqueeze(0), scale_factor=patch_size,
                                      mode="bilinear")[0].cpu().detach().numpy()
        plt.imsave(fname=f'{file_path}.png',
                   arr=attention_rollout_original_size[0],
                   format='png')


def plot_attn_probs(attentions: Tensor, image_size: int, patch_size: int, num_heads: int, path: Path,
                    iteration_idx: int = None, only_fusion: bool = True) -> None:
    w_featmap, h_featmap = image_size // patch_size, image_size // patch_size
    attentions = attentions.reshape(num_heads, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="bilinear")[
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
    return scores_image


def save_saliency_map(image: Tensor, saliency_map: Tensor, filename: Path, verbose: bool = False,
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
    color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # / 255

    # Combine image with heatmap
    img_with_heatmap = np.float32(color_heatmap) + np.float32(image)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    if verbose:
        plt.imshow(img_with_heatmap, interpolation='bilinear')
        plt.show()
    cv2.imwrite(f'{filename.resolve()}.png', np.uint8(255 * img_with_heatmap))


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
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
            transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
            image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    plt.imsave(fname=Path(f'{file_name}.png'),
               arr=vis,
               format='png')


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


def print_number_of_trainable_and_not_trainable_params(model: VitModelForClassification) -> None:
    print(
        f'Number of params: {calculate_num_of_params(model)}, Number of trainable params: {calculate_num_of_trainable_params(model)}')


def load_feature_extractor(vit_config: Dict) -> ViTFeatureExtractor:
    feature_extractor = ViTFeatureExtractor.from_pretrained(vit_config['model_name'])
    return feature_extractor


def load_ViTModel(vit_config: Dict, model_type: str) -> VitModelForClassification:
    model = vit_model_types[model_type].from_pretrained(vit_config['model_name'])
    return model


def load_feature_extractor_and_vit_model(vit_config: Dict, model_type: str = 'vit') -> Tuple[
    ViTFeatureExtractor, ViTForImageClassification]:
    feature_extractor = load_feature_extractor(vit_config=vit_config)
    vit_model = load_vit_model_by_type(vit_config=vit_config, model_type=model_type)
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


def get_top_k_mimimum_values_indices(array: List[float], k: int = 5, is_largest: bool = False):
    return torch.topk(torch.tensor(array), k=min(len(array), k), largest=is_largest)[1]


def get_patches_by_discard_ratio(array: Tensor, discard_ratio: float, top: bool = True) -> Tensor:
    k = int(array.shape[-1] * discard_ratio)
    _, indices = torch.topk(array, k, largest=bool(1 - top))
    array[indices] = 0
    return array


def rollout(attentions, discard_ratio: float = 0.9, head_fusion: str = 'max', return_resized: bool = True):
    result = torch.eye(attentions[0].size(-1)).to(device)
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            elif head_fusion == "median":
                attention_heads_fused = attention.median(axis=1)[0]
            else:
                raise ("Attention head fusion type Not supported")
            I = torch.eye(attention_heads_fused.size(-1)).to(device)
            a = ((attention_heads_fused + 1.0 * I) / 2).to(device)
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0, 1:]  # result.shape: [1, n_tokens + 1, n_tokens + 1]
    # In case of 224x224 image, this brings us from 196 to 14
    if return_resized:
        width = int(mask.size(-1) ** 0.5)
        mask = mask.reshape(width, width).numpy()
    mask = mask / torch.max(torch.tensor(mask))
    return mask


def get_minimum_predictions_string(image_name: str, total_losses: List[float], prediction_losses: List[float],
                                   logits: List[float], correct_class_probs: List[float], k: int = 25) -> str:
    min_pred_loss_iter, min_total_loss_iter, max_prob_iter, max_logits_iter = get_best_k_values_iterations(
        prediction_losses=prediction_losses, total_losses=total_losses,
        correct_class_probs=correct_class_probs, logits=logits, k=k)
    return f'Minimum prediction_loss at iteration: {min_pred_loss_iter}\nMinimum total loss at iteration: {min_total_loss_iter}\nMaximum logits at iteration: {max_logits_iter}\nMaximum probs at iteration: {max_prob_iter}'


def get_best_k_values_iterations(prediction_losses: List[float], total_losses: List[float],
                                 correct_class_probs: List[float], logits: List[float], k: int = 3):
    min_pred_loss_iter = get_top_k_mimimum_values_indices(array=prediction_losses, k=k, is_largest=False)
    min_total_loss_iter = get_top_k_mimimum_values_indices(array=total_losses, k=k, is_largest=False)
    max_logits_iter = get_top_k_mimimum_values_indices(array=logits, k=k, is_largest=True)
    max_prob_iter = get_top_k_mimimum_values_indices(array=correct_class_probs, k=k, is_largest=True)

    return min_pred_loss_iter, min_total_loss_iter, max_prob_iter, max_logits_iter


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


def compare_between_predicted_classes(vit_logits: Tensor, vit_s_logits: Tensor,
                                      contrastive_class_idx: Optional[Tensor] = None) -> \
        Tuple[bool, float]:
    target_class_idx = contrastive_class_idx.item() if contrastive_class_idx is not None else torch.argmax(
        vit_logits[0]).item()
    original_idx_logits_diff = (abs(max(vit_logits[0]).item() - vit_s_logits[0][target_class_idx].item()))
    is_predicted_same_class = target_class_idx == torch.argmax(vit_s_logits[0]).item()
    return is_predicted_same_class, original_idx_logits_diff


def compare_results_each_n_steps(iteration_idx: int, target: Tensor, output: Tensor, prev_x_attention: Tensor,
                                 sampled_binary_patches: Tensor = None, contrastive_class_idx: Tensor = None):
    target_class_idx = contrastive_class_idx.item() if contrastive_class_idx is not None else torch.argmax(
        target[0]).item()
    is_predicted_same_class, original_idx_logits_diff = compare_between_predicted_classes(
        vit_logits=target, vit_s_logits=output, contrastive_class_idx=contrastive_class_idx)
    print(
        f'Is predicted same class: {is_predicted_same_class}, Correct Class Prob: {F.softmax(output, dim=-1)[0][target_class_idx]}')
    if is_predicted_same_class is False:
        print(f'Predicted class change at {iteration_idx} iteration !!!!!!')
    # print(prev_x_attention)
    # if is_iteration_to_action(iteration_idx=iteration_idx, action='print'):
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


def plot_different_visualization_methods(path: Path, inputs, patch_size: int, vit_config: Dict,
                                         original_image=None) -> None:
    """
    Plotting Dino supervise, & rollout methods
    """
    vit_basic_for_dino = handle_model_config_and_freezing_for_task(
        model=load_ViTModel(vit_config, model_type='vit-for-dino'))
    _ = vit_basic_for_dino(**inputs)  # run forward to save attention_probs
    dino_method_attention_probs_cls_on_tokens_last_layer(vit_sigmoid_model=vit_basic_for_dino,
                                                         path=path)
    attention_probs = get_attention_probs(model=vit_basic_for_dino)
    plot_attention_rollout(attention_probs=attention_probs, path=path,
                           patch_size=patch_size, iteration_idx=0, head_fusion='max', original_image=original_image)
    plot_attention_rollout(attention_probs=attention_probs, path=path,
                           patch_size=patch_size, iteration_idx=0, head_fusion='mean', original_image=original_image)
    # plot_attention_rollout(attention_probs=attention_probs, path=path,
    #                        patch_size=patch_size, iteration_idx=0, head_fusion='median', original_image=original_image)
    # plot_attention_rollout(attention_probs=attention_probs, path=path,
    #                        patch_size=patch_size, iteration_idx=0, head_fusion='min', original_image=original_image)


def get_temp_to_visualize(temp):
    if len(temp.shape) == 1:  # [n_tokens]
        return temp[1:]
    elif len(temp.shape) == 2:  # [n_heads, n_tokens]
        return temp[:, 1:].reshape(12, -1)
    else:  # [n_layers, n_heads, n_tokens]
        return temp[-1, :, 1:].reshape(12, -1)


def save_text_to_file(path: Path, file_name: str, text: str):
    print(text)
    text = str(text) if text is None else text
    with open(Path(path, f'{file_name}.txt'), 'w') as f:
        f.write(text)


def read_file(path: Path) -> str:
    with open(Path(path), 'r') as f:
        data = f.read()
    return data


def get_minimum_prediction_string_and_write_to_disk(image_plot_folder_path, image_name, total_losses, prediction_losses,
                                                    correct_class_logits, correct_class_probs):
    minimum_predictions = get_minimum_predictions_string(image_name=image_name, total_losses=total_losses,
                                                         prediction_losses=prediction_losses,
                                                         logits=correct_class_logits,
                                                         correct_class_probs=correct_class_probs)
    save_text_to_file(path=image_plot_folder_path, file_name='minimum_predictions', text=minimum_predictions)


def visualize_attentions_and_temps(cls_attentions_probs, iteration_idx, mean_folder, median_folder, max_folder,
                                   min_folder, original_transformed_image, temp,
                                   temp_tokens_mean_folder, temp_tokens_median_folder, temp_tokens_max_folder,
                                   temp_tokens_min_folder, temp_tokens_folder=None):
    visualize_attention_scores(cls_attentions_probs=cls_attentions_probs, iteration_idx=iteration_idx,
                               max_folder=max_folder, mean_folder=mean_folder, median_folder=median_folder,
                               min_folder=min_folder, original_transformed_image=original_transformed_image)

    visualize_temp(iteration_idx=iteration_idx, original_transformed_image=original_transformed_image, temp=temp,
                   temp_tokens_folder=temp_tokens_folder, temp_tokens_max_folder=temp_tokens_max_folder,
                   temp_tokens_mean_folder=temp_tokens_mean_folder, temp_tokens_median_folder=temp_tokens_median_folder,
                   temp_tokens_min_folder=temp_tokens_min_folder)


def visualize_temp(iteration_idx, original_transformed_image, temp, temp_tokens_folder, temp_tokens_max_folder,
                   temp_tokens_mean_folder, temp_tokens_median_folder, temp_tokens_min_folder):
    if len(temp.shape) > 1:
        visu(original_image=original_transformed_image,
             transformer_attribution=temp.mean(dim=0),
             file_name=Path(temp_tokens_mean_folder, f'plot_{iteration_idx}'))
        visu(original_image=original_transformed_image,
             transformer_attribution=temp.median(dim=0)[0],
             file_name=Path(temp_tokens_median_folder, f'plot_{iteration_idx}'))
        visu(original_image=original_transformed_image,
             transformer_attribution=temp.max(dim=0)[0],
             file_name=Path(temp_tokens_max_folder, f'plot_{iteration_idx}'))
        visu(original_image=original_transformed_image,
             transformer_attribution=temp.min(dim=0)[0],
             file_name=Path(temp_tokens_min_folder, f'plot_{iteration_idx}'))
    else:
        visu(original_image=original_transformed_image,
             transformer_attribution=temp,
             file_name=Path(temp_tokens_folder, f'plot_{iteration_idx}'))


def visualize_attention_scores_with_rollout(cls_attentions_probs, rollout_vector, iteration_idx, max_folder,
                                            mean_folder, median_folder, min_folder,
                                            original_transformed_image, rollout_folder=None):
    visu(original_image=original_transformed_image,
         transformer_attribution=cls_attentions_probs.mean(dim=0) * rollout_vector,
         file_name=Path(mean_folder, f'plot_{iteration_idx}'))
    visu(original_image=original_transformed_image,
         transformer_attribution=cls_attentions_probs.median(dim=0)[0] * rollout_vector,
         file_name=Path(median_folder, f'plot_{iteration_idx}'))
    visu(original_image=original_transformed_image,
         transformer_attribution=cls_attentions_probs.max(dim=0)[0] * rollout_vector,
         file_name=Path(max_folder, f'plot_{iteration_idx}'))
    visu(original_image=original_transformed_image,
         transformer_attribution=cls_attentions_probs.min(dim=0)[0] * rollout_vector,
         file_name=Path(min_folder, f'plot_{iteration_idx}'))

    if rollout_folder is not None:
        visu(original_image=original_transformed_image,
             transformer_attribution=rollout_vector,
             file_name=Path(rollout_folder, f'plot_{iteration_idx}'))


def visualize_attention_scores(cls_attentions_probs, iteration_idx, max_folder, mean_folder, median_folder, min_folder,
                               original_transformed_image):
    visu(original_image=original_transformed_image,
         transformer_attribution=cls_attentions_probs.mean(dim=0),
         file_name=Path(mean_folder, f'plot_{iteration_idx}'))
    visu(original_image=original_transformed_image,
         transformer_attribution=cls_attentions_probs.median(dim=0)[0],
         file_name=Path(median_folder, f'plot_{iteration_idx}'))
    visu(original_image=original_transformed_image,
         transformer_attribution=cls_attentions_probs.max(dim=0)[0],
         file_name=Path(max_folder, f'plot_{iteration_idx}'))
    visu(original_image=original_transformed_image,
         transformer_attribution=cls_attentions_probs.min(dim=0)[0],
         file_name=Path(min_folder, f'plot_{iteration_idx}'))


def create_folder(path: Path) -> Path:
    os.makedirs(name=path, exist_ok=True)
    return path


def create_attention_probs_folders(image_plot_folder_path: Path):
    mean_folder = create_folder(Path(image_plot_folder_path, 'mean'))
    median_folder = create_folder(Path(image_plot_folder_path, 'median'))
    max_folder = create_folder(Path(image_plot_folder_path, 'max'))
    min_folder = create_folder(Path(image_plot_folder_path, 'min'))
    return mean_folder, median_folder, max_folder, min_folder


def create_temp_tokens_folders(image_plot_folder_path):
    temp_tokens_folder = create_folder(Path(image_plot_folder_path, 'temp_tokens'))
    temp_tokens_mean_folder = create_folder(Path(temp_tokens_folder, 'mean'))
    temp_tokens_median_folder = create_folder(Path(temp_tokens_folder, 'median'))
    temp_tokens_max_folder = create_folder(Path(temp_tokens_folder, 'max'))
    temp_tokens_min_folder = create_folder(Path(temp_tokens_folder, 'min'))
    return temp_tokens_folder, temp_tokens_max_folder, temp_tokens_mean_folder, temp_tokens_median_folder, temp_tokens_min_folder


def create_folders(image_plot_folder_path: Path):
    mean_folder, median_folder, max_folder, min_folder = create_attention_probs_folders(image_plot_folder_path)
    temp_tokens_folder, temp_tokens_max_folder, temp_tokens_mean_folder, temp_tokens_median_folder, temp_tokens_min_folder = create_temp_tokens_folders(
        image_plot_folder_path)

    return mean_folder, median_folder, max_folder, min_folder, temp_tokens_folder, temp_tokens_mean_folder, temp_tokens_median_folder, temp_tokens_max_folder, temp_tokens_min_folder


def visualize_attention_scores_by_layer_idx(model, image_plot_folder_path, original_image, iteration_idx):
    attention_scores_folder = create_folder(Path(image_plot_folder_path, 'attention_scores'))
    for layer_idx in range(get_num_layers(model=model)):
        mean_folder, median_folder, max_folder, min_folder = create_attention_probs_folders(
            image_plot_folder_path=Path(attention_scores_folder, f'layer_{layer_idx}'))
        attention_probs = get_attention_probs_by_layer_of_the_CLS(model=model, layer=layer_idx)
        visualize_attention_scores(cls_attentions_probs=attention_probs, iteration_idx=iteration_idx,
                                   max_folder=max_folder, mean_folder=mean_folder, median_folder=median_folder,
                                   min_folder=min_folder, original_transformed_image=original_image)


def visualize_temp_tokens_and_attention_scores(iteration_idx, max_folder, mean_folder, median_folder, min_folder,
                                               original_transformed_image, temp_tokens_max_folder,
                                               temp_tokens_mean_folder, temp_tokens_median_folder,
                                               temp_tokens_min_folder, vit_sigmoid_model, cls_attentions_probs):
    # visualize_attention_scores_by_layer_idx(model=vit_sigmoid_model, image_plot_folder_path=mean_folder.parent,
    #                                         original_image=original_transformed_image, iteration_idx=iteration_idx)
    temp = vit_sigmoid_model.vit.encoder.x_attention.clone()
    temp = get_temp_to_visualize(temp)
    visualize_attentions_and_temps(cls_attentions_probs=cls_attentions_probs, iteration_idx=iteration_idx,
                                   mean_folder=mean_folder, median_folder=median_folder,
                                   max_folder=max_folder, min_folder=min_folder,
                                   original_transformed_image=original_transformed_image,
                                   temp=temp, temp_tokens_mean_folder=temp_tokens_mean_folder,
                                   temp_tokens_median_folder=temp_tokens_median_folder,
                                   temp_tokens_max_folder=temp_tokens_max_folder,
                                   temp_tokens_min_folder=temp_tokens_min_folder)
    return temp


def visualize_attention_scores_only(iteration_idx, max_folder, mean_folder, median_folder, min_folder,
                                    original_transformed_image, vit_sigmoid_model):
    cls_attentions_probs = get_attention_probs_by_layer_of_the_CLS(model=vit_sigmoid_model)
    visualize_attention_scores(cls_attentions_probs=cls_attentions_probs, iteration_idx=iteration_idx,
                               max_folder=max_folder, mean_folder=mean_folder, median_folder=median_folder,
                               min_folder=min_folder, original_transformed_image=original_transformed_image)

    return cls_attentions_probs


def start_run_save_files_plot_visualizations_create_folders(model: nn.Module, image_plot_folder_path: Path, inputs, run,
                                                            original_image=None):
    print_number_of_trainable_and_not_trainable_params(model=model)
    if run is not None:
        save_text_to_file(path=image_plot_folder_path, file_name='metrics_url', text=run.url)
    if vit_config['plot_visualizations']:
        plot_different_visualization_methods(path=image_plot_folder_path, inputs=inputs,
                                             patch_size=vit_config['patch_size'], vit_config=vit_config,
                                             original_image=original_image)
    mean_folder, median_folder, max_folder, min_folder, temp_tokens_folder, temp_tokens_mean_folder, \
    temp_tokens_median_folder, temp_tokens_max_folder, temp_tokens_min_folder = create_folders(
        image_plot_folder_path)
    objects_path = create_folder(Path(image_plot_folder_path, 'objects'))
    return max_folder, mean_folder, median_folder, min_folder, objects_path, temp_tokens_max_folder, \
           temp_tokens_mean_folder, temp_tokens_median_folder, temp_tokens_min_folder


def end_iteration(correct_class_logits, correct_class_probs, image_name, image_plot_folder_path, iteration_idx,
                  objects_path, prediction_losses, tokens_mask, total_losses, temps, vit_sigmoid_model):
    if is_iteration_to_action(iteration_idx=iteration_idx, action='save'):
        objects_dict = {'losses': prediction_losses, 'total_losses': total_losses,
                        'tokens_mask': tokens_mask, 'temps': temps}
        save_objects(path=objects_path, objects_dict=objects_dict)

        get_minimum_prediction_string_and_write_to_disk(image_plot_folder_path=image_plot_folder_path,
                                                        image_name=image_name, total_losses=total_losses,
                                                        prediction_losses=prediction_losses,
                                                        correct_class_logits=correct_class_logits,
                                                        correct_class_probs=correct_class_probs)


def get_image_spec(image_dict: Dict) -> Tuple[str, Tensor, Optional[Tensor]]:
    image_name, correct_class_idx, contrastive_class_idx = image_dict['image_name'], image_dict['correct_class'], \
                                                           image_dict['contrastive_class']
    return image_name, correct_class_idx, contrastive_class_idx


def get_iteration_target_class_stats(output, target_class_idx: Tensor):
    correct_class_logit = output.logits[0][target_class_idx].item()
    correct_class_prob = F.softmax(output.logits[0], dim=-1)[target_class_idx].item()
    prediction_loss = ce_loss(output.logits, target_class_idx.unsqueeze(0)) * loss_config['pred_loss_multiplier']
    return correct_class_logit, correct_class_prob, prediction_loss


def is_iteration_to_action(iteration_idx: int, action: str = 'print') -> bool:
    """
    :param action: 'print' / 'save'
    """
    is_iter_to_action = iteration_idx > 0 and iteration_idx % vit_config[f'{action}_every'] == 0 or iteration_idx == \
                        vit_config['num_steps'] - 1
    if action == 'print':
        return vit_config['verbose'] and is_iter_to_action
    return is_iter_to_action


def get_image_and_inputs_and_transformed_image(feature_extractor: ViTFeatureExtractor, image_name: str = None,
                                               image=None):
    if image is None and image_name is not None:
        image = get_image_from_path(Path(IMAGES_FOLDER_PATH, image_name))
    inputs = feature_extractor(images=image, return_tensors="pt")
    original_transformed_image = image_transformations(image)
    return inputs, original_transformed_image


def setup_model_and_optimizer(model_name: str):
    vit_ours_model = handle_model_config_and_freezing_for_task(
        model=load_ViTModel(vit_config, model_type=model_name),
        freezing_transformer=vit_config['freezing_transformer'])
    optimizer = optim.Adam([vit_ours_model.vit.encoder.x_attention], lr=vit_config['lr'])
    return vit_ours_model, optimizer
