from transformers import ViTConfig
from tqdm import tqdm
import torch
from torch import nn
from torch.functional import F
from torch import optim
from config import config
from utils import *
from vit_utils import *
from loss_utils import *
from log_utils import configure_log
from consts import *
from pytorch_lightning import seed_everything

vit_config = config['vit']
seed_everything(config['general']['seed'])
experiment_name = f"l1_abs_x + entropy_loss + prediction_loss_multiplied_{vit_config['loss']['prediction_loss_multiplier']}"
configure_log(vit_config=vit_config, experiment_name=experiment_name)
feature_extractor, vit_model, vit_sigmoid_model = load_feature_extractor_and_vit_models(vit_config=vit_config)


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


def check_stop_criterion(x_attention: Tensor) -> bool:
    if len(torch.where(sigmoid(x_attention) >= float(vit_config['stop_prob_criterion']))[0]) == 0:
        return True
    return False


def compare_results_each_n_steps(iteration_idx: int, target: Tensor, output: Tensor, prev_x_attention: Tensor):
    is_predicted_same_class, original_idx_logits_diff = compare_between_predicted_classes(
        vit_logits=target, vit_s_logits=output)
    print(
        f'Is predicted same class: {is_predicted_same_class}, Correct Class Prob: {F.softmax(output)[0][torch.argmax(F.softmax(target)).item()]}')
    if is_predicted_same_class is False:
        print(f'Predicted class change at {iteration_idx} iteration !!!!!!')
        # print(sigmoid(prev_x_attention))


def compare_between_predicted_classes(vit_logits: Tensor, vit_s_logits: Tensor) -> Tuple[bool, float]:
    original_predicted_idx = torch.argmax(vit_logits[0]).item()
    original_idx_logits_diff = (abs(max(vit_logits[0]).item() - vit_s_logits[0][original_predicted_idx].item()))
    is_predicted_same_class = original_predicted_idx == torch.argmax(vit_s_logits[0]).item()
    return is_predicted_same_class, original_idx_logits_diff


def objective_loss_relu_entropy(output, target, x_attention: Tensor) -> Tensor:
    prediction_loss = ce_loss(F.softmax(output), torch.argmax(target).unsqueeze(0))
    x_attention_relu = nn.functional.relu(x_attention)
    x_attention_relu_normalized = x_attention_relu / x_attention_relu.sum()
    l1_loss = x_attention.abs().sum() / len(x_attention)
    entropy_loss = entropy(x_attention_relu_normalized + vit_config['small_number_for_stability'])
    # l1_loss = x_attention_relu.sum() / len(x_attention_relu)
    loss = l1_loss + entropy_loss + vit_config['loss']['prediction_loss_multiplier'] * prediction_loss
    log(loss=loss, l1_loss=l1_loss, entropy_loss=entropy_loss, prediction_loss=prediction_loss, x_attention=x_attention,
        output=output, target=target)
    return loss


def optimize_params(vit_model: ViTForImageClassification, vit_sigmoid_model: ViTSigmoidForImageClassification,
                    criterion, optimizer):
    for idx, image_name in enumerate(os.listdir(images_folder_path)):
        if image_name == vit_config['sample_picture_name']:
            print(image_name)
            image_plot_folder_path = Path(PLOTS_PATH, experiment_name)
            os.makedirs(name=image_plot_folder_path, exist_ok=True)
            image = get_image_from_path(os.path.join(images_folder_path, image_name))
            inputs = feature_extractor(images=image, return_tensors="pt")
            prev_loss = None
            losses = []
            x_attention = [vit_sigmoid_model.vit.encoder.x_attention]
            for iteration_idx in tqdm(range(vit_config['num_steps'])):
                optimizer.zero_grad()
                target = vit_model(**inputs)
                output = vit_sigmoid_model(**inputs)
                # loss = criterion(output=output.logits, target=target.logits,
                #                  x_attention=vit_sigmoid_model.vit.encoder.x_attention, iteration_idx=iteration_idx)
                loss = criterion(output=output.logits, target=target.logits,
                                 x_attention=vit_sigmoid_model.vit.encoder.x_attention)
                losses.append(loss)
                loss.backward()
                prev_x_attention = vit_sigmoid_model.vit.encoder.x_attention.clone()
                optimizer.step()
                x_attention.append(prev_x_attention)
                compare_results_each_n_steps(iteration_idx=iteration_idx, target=target.logits, output=output.logits,
                                             prev_x_attention=prev_x_attention)
                if prev_loss and is_iteration_to_print:
                    print(nn.functional.relu(vit_sigmoid_model.vit.encoder.x_attention))
                    if vit_config['verbose']:
                        plot_scores(scores=vit_sigmoid_model.vit.encoder.x_attention, file_name=experiment_name,
                                    iteration_idx=iteration_idx, image_plot_folder_path=image_plot_folder_path)
                    # if check_stop_criterion(x_attention=prev_x_attention):
                    #     print(f"stop_at_iteration_idx: {iteration_idx}")
                    #     break
                prev_loss = loss
            save_obj_to_disk(f'{experiment_name}_{image_name.replace(".JPEG", "")}_x_attention', x_attention)
            save_obj_to_disk(f'{experiment_name}_{image_name.replace(".JPEG", "")}_losses', losses)
            return vit_model, vit_sigmoid_model


def infer(experiment_name: str):
    """
    Load saved model and run forward
    :return:
    """
    vit_sigmoid_model = load_model(model_name=f'{experiment_name}_vit_sigmoid_model')
    image = get_image_from_path(os.path.join(images_folder_path, vit_config['sample_picture_name']))
    inputs = feature_extractor(images=image, return_tensors="pt")
    output = vit_sigmoid_model(**inputs)
    print(F.softmax(output.logits))
    print(F.softmax(output.logits)[0][65])  # 65 refer to correct class: torch.argmax(F.softmax(target)).item()


if __name__ == '__main__':
    # print_number_of_trainable_and_not_trainable_params(model=vit_model, model_name='ViT')
    # print_number_of_trainable_and_not_trainable_params(model=vit_sigmoid_model, model_name='ViT-Sigmoid')
    optimizer = optim.Adam([vit_sigmoid_model.vit.encoder.x_attention], lr=vit_config['lr'])
    vit_model, vit_sigmoid_model = optimize_params(vit_model=vit_model, vit_sigmoid_model=vit_sigmoid_model,
                                                   criterion=objective_loss_relu_entropy,
                                                   optimizer=optimizer)
    # save_model(model=vit_sigmoid_model, model_name=f'{experiment_name}_vit_sigmoid_model')
