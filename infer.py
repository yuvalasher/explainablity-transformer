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

vit_config = config['vit']
loss_config = vit_config['loss']

seed_everything(config['general']['seed'])
experiment_name = f"clipped instead of relu l1_abs_x_mul_{loss_config['l1_loss_multiplier']} + entropy_loss_mul_{loss_config['entropy_loss_multiplier']} + prediction_loss_mul_{loss_config['prediction_loss_multiplier']}"
run = configure_log(vit_config=vit_config, experiment_name=experiment_name)
feature_extractor, vit_model = load_feature_extractor_and_vit_models(vit_config=vit_config)


def save_model(model: nn.Module, model_name: str) -> None:
    path = Path(f'{PICKLES_FOLDER_PATH}', f'{model_name}.pt')
    torch.save(model.state_dict(), path)


def load_model(model_name: str) -> nn.Module:
    path = Path(f'{PICKLES_FOLDER_PATH}', f'{model_name}.pt')
    c = ViTConfig()
    c.image_size = vit_config['img_size']
    c.num_labels = vit_config['num_labels']
    model = ViTSigmoidForImageClassification(config=c)
    model.load_state_dict(torch.load(path))
    return model


def compare_results_each_n_steps(iteration_idx: int, target: Tensor, output: Tensor, prev_x_attention: Tensor):
    is_predicted_same_class, original_idx_logits_diff = compare_between_predicted_classes(
        vit_logits=target, vit_s_logits=output)
    print(
        f'Is predicted same class: {is_predicted_same_class}, Correct Class Prob: {F.softmax(output)[0][torch.argmax(F.softmax(target)).item()]}')
    if is_predicted_same_class is False:
        print(f'Predicted class change at {iteration_idx} iteration !!!!!!')
    if is_iteration_to_print(iteration_idx=iteration_idx):
        print(prev_x_attention)


def compare_between_predicted_classes(vit_logits: Tensor, vit_s_logits: Tensor) -> Tuple[bool, float]:
    original_predicted_idx = torch.argmax(vit_logits[0]).item()
    original_idx_logits_diff = (abs(max(vit_logits[0]).item() - vit_s_logits[0][original_predicted_idx].item()))
    is_predicted_same_class = original_predicted_idx == torch.argmax(vit_s_logits[0]).item()
    return is_predicted_same_class, original_idx_logits_diff


def objective_2(output: Tensor, target: Tensor, x_attention: Tensor) -> Tensor:
    # prediction_loss = ce_loss(output, torch.argmax(target).unsqueeze(0)) * loss_config['pred_loss_multiplier']
    prediction_loss = output[0][torch.argmax(F.softmax(target)).item()] * loss_config['pred_loss_multiplier'] # logits in correct class
    l1_loss = (x_attention - 1).abs().sum() / len(x_attention) * loss_config['l1_loss_multiplier']
    # x_attention_chopped = relu(x_attention)
    x_attention_chopped = torch.clamp(x_attention, min=0, max=1)
    x_attention_chopped = -(x_attention_chopped - x_attention_chopped.max())
    x_attention_chopped_normalized = x_attention_chopped / x_attention_chopped.sum()
    entropy_loss = entropy(x_attention_chopped_normalized) * loss_config['entropy_loss_multiplier']

    loss = l1_loss + entropy_loss + prediction_loss
    log(loss=loss, l1_loss=l1_loss, entropy_loss=entropy_loss, prediction_loss=prediction_loss, x_attention=x_attention,
        output=output, target=target)
    return loss


def objective_loss_relu_entropy(output: Tensor, target: Tensor, x_attention: Tensor) -> Tensor:
    prediction_loss = ce_loss(output, torch.argmax(target).unsqueeze(0)) * loss_config['pred_loss_multiplier']
    # prediction_loss = -output[0][torch.argmax(F.softmax(target)).item()]  # logits in correct class
    x_attention_chopped = relu(x_attention)
    # x_attention_chopped = torch.clamp(x_attention, min=0, max=1)
    # x_attention_chopped = -(x_attention_chopped - x_attention_chopped.max())
    x_attention_chopped_normalized = x_attention_chopped / x_attention_chopped.sum()
    # l1_loss = (x_attention - 1).abs().sum() / len(x_attention) * loss_config['l1_loss_multiplier']
    l1_loss = x_attention.abs().sum() / len(x_attention) * loss_config['l1_loss_multiplier']
    entropy_loss = entropy(x_attention_chopped_normalized) * loss_config['entropy_loss_multiplier']

    loss = l1_loss + entropy_loss + prediction_loss
    log(loss=loss, l1_loss=l1_loss, entropy_loss=entropy_loss, prediction_loss=prediction_loss, x_attention=x_attention,
        output=output, target=target)
    return loss


def optimize_params(vit_model: ViTForImageClassification, criterion: Callable):
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
            for iteration_idx in tqdm(range(vit_config['num_steps'])):
                optimizer.zero_grad()
                output = vit_sigmoid_model(**inputs)
                # loss = criterion(output=output.logits, target=target.logits,
                #                  x_attention=vit_sigmoid_model.vit.encoder.x_attention, iteration_idx=iteration_idx)
                # dino_method_attention_probs_cls_on_tokens_last_layer(vit_sigmoid_model=vit_sigmoid_model,
                #                                                      image_name=image_name)
                loss = criterion(output=output.logits, target=target.logits,
                                 x_attention=vit_sigmoid_model.vit.encoder.x_attention)
                loss.backward()
                prev_x_attention = vit_sigmoid_model.vit.encoder.x_attention.clone()
                optimizer.step()
                compare_results_each_n_steps(iteration_idx=iteration_idx, target=target.logits, output=output.logits,
                                             prev_x_attention=prev_x_attention)
                if vit_config['verbose']:
                    save_saliency_map(image=original_transformed_image,
                                      saliency_map=torch.tensor(
                                          get_scores(vit_sigmoid_model.vit.encoder.x_attention)).unsqueeze(0),
                                      filename=Path(image_plot_folder_path, f'iter_idx_{iteration_idx}'),
                                      verbose=is_iteration_to_print(iteration_idx=iteration_idx))
            if vit_config['log']:
                run.finish()
                vit_config['log'] = False


def infer(experiment_name: str):
    """
    Load saved model and run forward
    """
    vit_sigmoid_model = load_model(model_name=f'{experiment_name}_vit_sigmoid_model')
    image = get_image_from_path(os.path.join(images_folder_path, vit_config['sample_picture_name']))
    inputs = feature_extractor(images=image, return_tensors="pt")
    output = vit_sigmoid_model(**inputs)
    print(F.softmax(output.logits))
    print(F.softmax(output.logits)[0][65])  # 65 refer to correct class: torch.argmax(F.softmax(target)).item()


if __name__ == '__main__':
    os.makedirs(name=Path(PLOTS_PATH, experiment_name), exist_ok=True)
    optimize_params(vit_model=vit_model, criterion=objective_2)
    # optimize_params(vit_model=vit_model, criterion=objective_loss_relu_entropy)
    # save_model(model=vit_sigmoid_model, model_name=f'{experiment_name}_vit_sigmoid_model')
