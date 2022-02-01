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
from transformation import pil_to_resized_tensor_transform
from typing import Callable
from vit_for_dino import ViTBasicForDinoForImageClassification
vit_config = config['vit']
loss_config = vit_config['loss']

seed_everything(config['general']['seed'])
feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config)


def save_model(model: nn.Module, path: str) -> None:
    path = Path(f'{path}.pt')
    torch.save(model.state_dict(), path)
    print(f'Model Saved at {path}')


def load_model(path: str) -> nn.Module:
    # path = Path(f'{PICKLES_FOLDER_PATH}', f'{model_name}.pt')
    if path[-3:] == '.pt':
        path = Path(f'{path}')
    else:
        path = Path(f'{path}.pt')
    c = ViTConfig()
    c.image_size = vit_config['img_size']
    c.num_labels = vit_config['num_labels']
    model = ViTSigmoidForImageClassification(config=c)
    model.load_state_dict(torch.load(path))
    return model


def compare_results_each_n_steps(iteration_idx: int, target: Tensor, output: Tensor, prev_x_attention: Tensor,
                                 sampled_binary_patches: Tensor = None):
    is_predicted_same_class, original_idx_logits_diff = compare_between_predicted_classes(
        vit_logits=target, vit_s_logits=output)
    print(
        f'Is predicted same class: {is_predicted_same_class}, Correct Class Prob: {F.softmax(output)[0][torch.argmax(F.softmax(target)).item()]}')
    if is_predicted_same_class is False:
        print(f'Predicted class change at {iteration_idx} iteration !!!!!!')

    if is_iteration_to_action(iteration_idx=iteration_idx, action='print'):
        print(torch.sigmoid(prev_x_attention))
        if sampled_binary_patches is not None:
            print(sampled_binary_patches)
        # print(prev_x_attention.grad)


def compare_between_predicted_classes(vit_logits: Tensor, vit_s_logits: Tensor) -> Tuple[bool, float]:
    original_predicted_idx = torch.argmax(vit_logits[0]).item()
    original_idx_logits_diff = (abs(max(vit_logits[0]).item() - vit_s_logits[0][original_predicted_idx].item()))
    is_predicted_same_class = original_predicted_idx == torch.argmax(vit_s_logits[0]).item()
    return is_predicted_same_class, original_idx_logits_diff


def objective_2(output: Tensor, target: Tensor, x_attention: Tensor) -> Tensor:
    # prediction_loss = ce_loss(output, torch.argmax(target).unsqueeze(0)) * loss_config['pred_loss_multiplier']
    prediction_loss = output[0][torch.argmax(F.softmax(target)).item()] * loss_config['pred_loss_multiplier']
    l1_loss = (x_attention - 1).abs().sum() / len(x_attention) * loss_config['l1_loss_multiplier']
    x_attention_chopped = torch.clamp(x_attention, min=0, max=1)
    x_attention_chopped = -(x_attention_chopped - x_attention_chopped.max())
    x_attention_chopped_normalized = x_attention_chopped / x_attention_chopped.sum()
    entropy_loss = entropy(x_attention_chopped_normalized) * loss_config['entropy_loss_multiplier']

    loss = l1_loss + entropy_loss + prediction_loss
    log(loss=loss, l1_loss=l1_loss, entropy_loss=entropy_loss, prediction_loss=prediction_loss, x_attention=x_attention,
        output=output, target=target)
    return loss


def objective_loss_relu_entropy(output, target, x_attention: Tensor) -> Tensor:
    prediction_loss = ce_loss(output, torch.argmax(target).unsqueeze(0)) * vit_config['loss']['pred_loss_multiplier']
    x_attention_relu = relu(x_attention)
    x_attention_relu_normalized = x_attention_relu / x_attention_relu.sum()
    l1_loss = x_attention.abs().sum() / len(x_attention) * vit_config['loss']['l1_loss_multiplier']
    entropy_loss = entropy(x_attention_relu_normalized) * vit_config['loss']['entropy_loss_multiplier']
    # l1_loss = x_attention_relu.sum() / len(x_attention_relu)
    loss = l1_loss + entropy_loss + prediction_loss
    log(loss=loss, l1_loss=l1_loss, entropy_loss=entropy_loss, prediction_loss=prediction_loss, x_attention=x_attention,
        output=output, target=target)
    return loss


def objective_1(output: Tensor, target: Tensor, x_attention: Tensor) -> Tensor:
    prediction_loss = ce_loss(output, torch.argmax(target).unsqueeze(0)) * loss_config['pred_loss_multiplier']
    l1_loss = x_attention.abs().sum() / len(x_attention) * vit_config['loss']['l1_loss_multiplier']
    x_attention_chopped = relu(x_attention)
    x_attention_chopped_normalized = x_attention_chopped / x_attention_chopped.sum()
    entropy_loss = entropy(x_attention_chopped_normalized) * loss_config['entropy_loss_multiplier']

    loss = l1_loss + entropy_loss + prediction_loss
    log(loss=loss, l1_loss=l1_loss, entropy_loss=entropy_loss, prediction_loss=prediction_loss, x_attention=x_attention,
        output=output, target=target)
    return loss


def js_kl(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


# def kl_div(p, q):
#     return torch.sum(
#         torch.where(p != 0, p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q)), torch.tensor(0.0))) / len(p)

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


def objective_gumble_minimize_softmax(output: Tensor, target: Tensor, x_attention: Tensor,
                                      sampled_binary_patches: Tensor = None) -> Tensor:
    prediction_loss = -output[0][torch.argmax(F.softmax(target)).item()]
    kl = kl_div(p=convert_probability_vector_to_bernoulli_kl(F.sigmoid(x_attention)),
                q=convert_probability_vector_to_bernoulli_kl(torch.ones_like(x_attention))) * loss_config[
             'kl_loss_multiplier']
    print(f'kl_loss: {kl}, prediction_loss: {prediction_loss}')
    loss = kl + prediction_loss
    log(loss=loss, kl_loss=kl, prediction_loss=prediction_loss, x_attention=x_attention,
        sampled_binary_patches=sampled_binary_patches, output=output, target=target)
    return loss


def optimize_params(vit_model: ViTForImageClassification, criterion: Callable, log_run):
    for idx, image_name in enumerate(os.listdir(images_folder_path)):
        if image_name in vit_config['sample_images']:
            vit_sigmoid_model = handle_model_for_task(model=load_ViTModel(vit_config, model_type='vit-sigmoid'))
            optimizer = optim.Adam([vit_sigmoid_model.vit.encoder.x_attention], lr=vit_config['lr'])

            image_plot_folder_path = get_and_create_image_plot_folder_path(images_folder_path=images_folder_path,
                                                                           image_name=image_name,
                                                                           experiment_name=experiment_name)
            image = get_image_from_path(Path(images_folder_path, image_name))
            inputs = feature_extractor(images=image, return_tensors="pt")
            original_transformed_image = pil_to_resized_tensor_transform(image)
            target = vit_model(**inputs)
            x_gradients = []
            sampled_binary_patches = []
            # vit_basic_for_dino = handle_model_for_task(model=load_ViTModel(vit_config, model_type='vit-for-dino'))
            # _ = vit_basic_for_dino(**inputs)
            # dino_method_attention_probs_cls_on_tokens_last_layer(vit_sigmoid_model=vit_basic_for_dino, image_name=image_name)
            for iteration_idx in tqdm(range(vit_config['num_steps'])):
                optimizer.zero_grad()
                output = vit_sigmoid_model(**inputs)

                if vit_config['objective'] in ['objective_gumble_softmax', 'objective_gumble_minimize_softmax','objective_opposite_gumble_softmax']:
                    loss = criterion(output=output.logits, target=target.logits,
                                     x_attention=vit_sigmoid_model.vit.encoder.x_attention,
                                     sampled_binary_patches=vit_sigmoid_model.vit.encoder.sampled_binary_patches)
                else:
                    loss = criterion(output=output.logits, target=target.logits,
                                     x_attention=vit_sigmoid_model.vit.encoder.x_attention)
                loss.backward()
                compare_results_each_n_steps(iteration_idx=iteration_idx, target=target.logits, output=output.logits,
                                             prev_x_attention=vit_sigmoid_model.vit.encoder.x_attention,
                                             sampled_binary_patches=vit_sigmoid_model.vit.encoder.sampled_binary_patches.clone() if
                                             vit_config['objective'] in ['objective_gumble_softmax',
                                                                         'objective_gumble_minimize_softmax',
                                                                         'objective_opposite_gumble_softmax'] else None)
                if vit_config['verbose']:
                    printed_vector = vit_sigmoid_model.vit.encoder.sampled_binary_patches if vit_config['objective'] in \
                                                                                             ['objective_gumble_minimize_softmax',
                                                                                              'objective_gumble_softmax',
                                                                                              'objective_opposite_gumble_softmax'] else relu(
                        vit_sigmoid_model.vit.encoder.x_attention)
                    save_saliency_map(image=original_transformed_image,
                                      saliency_map=torch.tensor(
                                          get_scores(printed_vector)).unsqueeze(0),
                                      filename=Path(image_plot_folder_path, f'relu_x_iter_idx_{iteration_idx}'),
                                      verbose=is_iteration_to_action(iteration_idx=iteration_idx, action='print'))

                    # save_saliency_map(image=original_transformed_image,
                    #                   saliency_map=torch.tensor(
                    #                       get_scores(
                    #                           torch.sigmoid(vit_sigmoid_model.vit.encoder.x_attention))).unsqueeze(0),
                    #                   filename=Path(image_plot_folder_path, f'sigmoid_x_iter_idx_{iteration_idx}'),
                    #                   verbose=is_iteration_to_action(iteration_idx=iteration_idx, action='print'))

                    # save_saliency_map(image=original_transformed_image,
                    #                   saliency_map=torch.tensor(
                    #                       get_scores(vit_sigmoid_model.vit.encoder.x_attention.grad)).unsqueeze(0),
                    #                   filename=Path(image_plot_folder_path, f'grad_x_iter_idx_{iteration_idx}'),
                    #                   verbose=is_iteration_to_action(iteration_idx=iteration_idx, action='print'))
                    #
                    # save_saliency_map(image=original_transformed_image,
                    #                   saliency_map=torch.tensor(
                    #                       get_scores(
                    #                           vit_sigmoid_model.vit.encoder.x_attention.grad * vit_sigmoid_model.vit.encoder.x_attention)).unsqueeze(
                    #                       0),
                    #                   filename=Path(image_plot_folder_path, f'grad_x_mul_x_iter_idx_{iteration_idx}'),
                    #                   verbose=is_iteration_to_action(iteration_idx=iteration_idx, action='print'))
                    #
                    # save_saliency_map(image=original_transformed_image,
                    #                   saliency_map=torch.tensor(
                    #                       get_scores(relu(
                    #                           vit_sigmoid_model.vit.encoder.x_attention.grad))).unsqueeze(0),
                    #                   filename=Path(image_plot_folder_path, f'relu_grad_x_iter_idx_{iteration_idx}'),
                    #                   verbose=is_iteration_to_action(iteration_idx=iteration_idx, action='print'))
                    #
                    # save_saliency_map(image=original_transformed_image,
                    #                   saliency_map=torch.tensor(
                    #                       get_scores(relu(
                    #                           vit_sigmoid_model.vit.encoder.x_attention.grad) * vit_sigmoid_model.vit.encoder.x_attention)).unsqueeze(
                    #                       0),
                    #                   filename=Path(image_plot_folder_path, f'relu_grad_x_mul_x_iter_idx_{iteration_idx}'),
                    #                   verbose=is_iteration_to_action(iteration_idx=iteration_idx, action='print'))
                optimizer.step()
                x_gradients.append(vit_sigmoid_model.vit.encoder.x_attention.grad.clone())
                sampled_binary_patches.append(vit_sigmoid_model.vit.encoder.sampled_binary_patches.clone())
                if is_iteration_to_action(iteration_idx=iteration_idx, action='save'):
                    save_obj_to_disk(f'{image_plot_folder_path}_x_gradients', x_gradients)
                    save_obj_to_disk(f'{image_plot_folder_path}_s_b_p', sampled_binary_patches)
                    save_model(model=vit_sigmoid_model, path=Path(f'{image_plot_folder_path}', 'vit_sigmoid_model'))
            if vit_config['log']:
                log_run.finish()
                vit_config['log'] = False
            save_model(model=vit_sigmoid_model, path=Path(f'{image_plot_folder_path}', 'vit_sigmoid_model'))
            print(1)


OBJECTIVES = {'objective_gumble_softmax': objective_gumble_softmax,  # x_attention as rand
              'objective_opposite_gumble_softmax': objective_gumble_softmax,  # x_attention as rand
              'objective_1': objective_1,  # Require to initiate x_attention as ones & relu
              'objective_2': objective_2,  # x_attention as rand & clamp
              'objective_loss_relu_entropy': objective_loss_relu_entropy,  # x_attention as rand & clamp
              'objective_gumble_minimize_softmax': objective_gumble_minimize_softmax
              }
experiment_name = f"{vit_config['objective']}_lr{str(vit_config['lr']).replace('.', '_')}_temp_{vit_config['temperature']}+l1_{loss_config['l1_loss_multiplier']}+kl_loss_{loss_config['kl_loss_multiplier']}+entropy_loss_{loss_config['entropy_loss_multiplier']}+pred_loss_{loss_config['pred_loss_multiplier']}"

if __name__ == '__main__':
    log_run = configure_log(vit_config=vit_config, experiment_name=experiment_name)
    os.makedirs(name=Path(PLOTS_PATH, experiment_name), exist_ok=True)
    optimize_params(vit_model=vit_model, criterion=OBJECTIVES[vit_config['objective']], log_run=log_run)
    # infer_prediction(path=r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\D_objective_opposite_gumble_softmax_lr0_3_temp_1+l1_0+kl_loss_1+entropy_loss_0+pred_loss_20\ILSVRC2012_val_00000001\vit_sigmoid_model")
