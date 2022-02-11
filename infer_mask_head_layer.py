from transformers import ViTConfig
from tqdm import tqdm
import torch
from torch import nn
from torch.functional import F
from torch.nn.functional import relu
from torch import optim
from config import config
from utils import *
from vit_utils import *
from loss_utils import *
from log_utils import configure_log
from consts import *
from pytorch_lightning import seed_everything
from typing import Callable

vit_config = config['vit']
loss_config = vit_config['loss']

seed_everything(config['general']['seed'])
feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config)

def compare_results_each_n_steps(iteration_idx: int, target: Tensor, output: Tensor, prev_x_attention: Tensor,
                                 sampled_binary_patches: Tensor = None):
    is_predicted_same_class, original_idx_logits_diff = compare_between_predicted_classes(
        vit_logits=target, vit_s_logits=output)
    print(
        f'Is predicted same class: {is_predicted_same_class}, Correct Class Prob: {F.softmax(output)[0][torch.argmax(F.softmax(target)).item()]}')
    if is_predicted_same_class is False:
        print(f'Predicted class change at {iteration_idx} iteration !!!!!!')
    if is_iteration_to_action(iteration_idx=iteration_idx, action='print'):
        # print(nn.functional.softmax(prev_x_attention))
        print(F.sigmoid(prev_x_attention))
        if sampled_binary_patches is not None and vit_config['objective'] != 'objective_gumble_minimize_softmax':
            print(sampled_binary_patches)
        # print(prev_x_attention.grad)

def compare_between_predicted_classes(vit_logits: Tensor, vit_s_logits: Tensor) -> Tuple[bool, float]:
    original_predicted_idx = torch.argmax(vit_logits[0]).item()
    original_idx_logits_diff = (abs(max(vit_logits[0]).item() - vit_s_logits[0][original_predicted_idx].item()))
    is_predicted_same_class = original_predicted_idx == torch.argmax(vit_s_logits[0]).item()
    return is_predicted_same_class, original_idx_logits_diff

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


def objective_gumble_softmax(output: Tensor, target: Tensor, x_attention: Tensor,
                             sampled_binary_patches: Tensor = None) -> Tensor:
    prediction_loss = ce_loss(output, torch.argmax(target).unsqueeze(0)) * loss_config['pred_loss_multiplier']
    kl = kl_div(p=convert_probability_vector_to_bernoulli_kl(F.sigmoid(x_attention)),
                q=convert_probability_vector_to_bernoulli_kl(torch.zeros_like(x_attention))) * loss_config[
             'kl_loss_multiplier']
    print(f'kl_loss: {kl}, prediction_loss: {prediction_loss}')
    loss = kl + prediction_loss
    log(loss=loss, kl_loss=kl, prediction_loss=prediction_loss, x_attention=x_attention,
        sampled_binary_patches=sampled_binary_patches, output=output, target=target)
    return loss


def optimize_params(vit_model: ViTForImageClassification, criterion: Callable, log_run):
    for idx, image_name in enumerate(os.listdir(images_folder_path)):
        if image_name in vit_config['sample_images']:

            vit_sigmoid_model = handle_model_config_and_freezing_for_task(
                model=load_ViTModel(vit_config, model_type='per-layer-head'),
                freezing_transformer=vit_config['freezing_transformer'])
            optimizer = optim.Adam([vit_sigmoid_model.vit.encoder.x_attention], lr=vit_config['lr'])
            image = get_image_from_path(Path(images_folder_path, image_name))
            inputs = feature_extractor(images=image, return_tensors="pt")
            target = vit_model(**inputs)

            for iteration_idx in tqdm(range(vit_config['num_steps'])):
                optimizer.zero_grad()
                output = vit_sigmoid_model(**inputs)
                loss = criterion(output=output.logits, target=target.logits,
                                 x_attention=vit_sigmoid_model.vit.encoder.x_attention,
                                 sampled_binary_patches=vit_sigmoid_model.vit.encoder.sampled_binary_patches)
                loss.backward()
                compare_results_each_n_steps(iteration_idx=iteration_idx, target=target.logits, output=output.logits,
                                             prev_x_attention=vit_sigmoid_model.vit.encoder.x_attention,
                                             sampled_binary_patches=vit_sigmoid_model.vit.encoder.sampled_binary_patches.clone() if
                                             vit_config['objective'] in vit_config['gumble_objectives'] else None)

                # if vit_config['verbose'] and iteration_idx % 100 == 0 and iteration_idx > 0:
                #     printed_vector = vit_sigmoid_model.vit.encoder.sampled_binary_patches if vit_config[
                #                                                                                  'objective'] in \
                #                                                                              vit_config[
                #                                                                                  'gumble_objectives'] else relu(
                #         vit_sigmoid_model.vit.encoder.x_attention)
                #     save_saliency_map(image=original_transformed_image,
                #                       saliency_map=torch.tensor(
                #                           get_scores(printed_vector)).unsqueeze(0),
                #                       filename=Path(image_plot_folder_path, f'relu_x_iter_idx_{iteration_idx}'),
                #                       verbose=is_iteration_to_action(iteration_idx=iteration_idx, action='print'))
                optimizer.step()

OBJECTIVES = {'objective_gumble_softmax': objective_gumble_softmax,  # x_attention as rand
              }
experiment_name = f"{vit_config['objective']}_lr{str(vit_config['lr']).replace('.', '_')}_temp_{vit_config['temperature']}+l1_{loss_config['l1_loss_multiplier']}+kl_loss_{loss_config['kl_loss_multiplier']}+entropy_loss_{loss_config['entropy_loss_multiplier']}+pred_loss_{loss_config['pred_loss_multiplier']}"

if __name__ == '__main__':
    log_run = configure_log(vit_config=vit_config, experiment_name=experiment_name)
    os.makedirs(name=Path(PLOTS_PATH, experiment_name), exist_ok=True)
    optimize_params(vit_model=vit_model, criterion=OBJECTIVES[vit_config['objective']], log_run=log_run)
