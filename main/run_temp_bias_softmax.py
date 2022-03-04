from transformers import ViTConfig
from tqdm import tqdm
from torch import optim
from utils import *
from loss_utils import *
from log_utils import configure_log
from utils.consts import *
from pytorch_lightning import seed_everything
from utils.transformation import pil_to_resized_tensor_transform
from typing import Callable

vit_config = config['vit']
loss_config = vit_config['loss']

seed_everything(config['general']['seed'])
feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config)
vit_infer = handle_model_config_and_freezing_for_task(model=load_ViTModel(vit_config, model_type='infer'))


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
        f'Is predicted same class: {is_predicted_same_class}, Correct Class Prob: {F.softmax(output, dim=-1)[0][torch.argmax(F.softmax(target, dim=-1)).item()]}')
    if is_predicted_same_class is False:
        print(f'Predicted class change at {iteration_idx} iteration !!!!!!')
    # if is_iteration_to_action(iteration_idx=iteration_idx, action='print'):
    #     print('temp')
    #     print(prev_x_attention)


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


def objective_temp_bias_softmax(output: Tensor, target: Tensor, temp: Tensor) -> Tensor:
    prediction_loss = ce_loss(output, torch.argmax(target).unsqueeze(0)) * loss_config['pred_loss_multiplier']
    entropy_loss = entropy(F.softmax(temp, dim=-1)) * loss_config['entropy_loss_multiplier']
    other_loss = torch.mean(temp) * loss_config['other_loss_multiplier']
    # print(f'prediction_loss: {prediction_loss}')
    loss = prediction_loss + other_loss + entropy_loss
    print(f'loss: {loss}, max_temp: {torch.max(temp)}, min_temp: {torch.min(temp)}')

    log(loss=loss, prediction_loss=prediction_loss, entropy_loss=entropy_loss, other_loss=other_loss, x_attention=temp, output=output,
        target=target)
    return loss


def optimize_params(vit_model: ViTForImageClassification, criterion: Callable, log_run):
    for idx, image_dict in enumerate(vit_config['images']):
        image_name, correct_class_idx, contrastive_class_idx = image_dict['image_name'], image_dict['correct_class'], image_dict['contrastive_class']
        vit_sigmoid_model = handle_model_config_and_freezing_for_task(
            model=load_ViTModel(vit_config, model_type='softmax_bias_temp'),
            freezing_transformer=vit_config['freezing_transformer'])
        print_number_of_trainable_and_not_trainable_params(model=vit_sigmoid_model, model_name='soft_temp')
        optimizer = optim.Adam([vit_sigmoid_model.vit.encoder.x_attention], lr=vit_config['lr'])

        image_plot_folder_path = get_and_create_image_plot_folder_path(images_folder_path=IMAGES_FOLDER_PATH,
                                                                       experiment_name=experiment_name,
                                                                       image_name=image_name)
        save_url_to_text_file(path=image_plot_folder_path, log_run=log_run) if log_run is not None else []
        image = get_image_from_path(Path(IMAGES_FOLDER_PATH, image_name))
        inputs = feature_extractor(images=image, return_tensors="pt")
        original_transformed_image = pil_to_resized_tensor_transform(image)
        target = vit_model(**inputs)
        total_losses = []
        prediction_losses = []
        tokens_mask = []
        x_attention = []
        for iteration_idx in tqdm(range(vit_config['num_steps'])):
            optimizer.zero_grad()
            output = vit_sigmoid_model(**inputs)
            prediction_losses.append(ce_loss(output.logits, torch.argmax(target.logits).unsqueeze(0)) * loss_config[
                'pred_loss_multiplier'])
            x_attention.append(vit_sigmoid_model.vit.encoder.x_attention.clone())
            loss = criterion(output=output.logits, target=target.logits,
                             temp=vit_sigmoid_model.vit.encoder.x_attention)
            loss.backward()
            total_losses.append(loss.item())
            cls_attentions_probs = get_attention_probs_by_layer_of_the_CLS(model=vit_sigmoid_model)
            tokens_mask.append(cls_attentions_probs.clone())
            compare_results_each_n_steps(iteration_idx=iteration_idx, target=target.logits, output=output.logits,
                                         prev_x_attention=vit_sigmoid_model.vit.encoder.x_attention,
                                         sampled_binary_patches=None)

            save_saliency_map(image=original_transformed_image,
                              saliency_map=torch.tensor(
                                  get_scores(cls_attentions_probs.mean(dim=0))).unsqueeze(0),
                              filename=Path(image_plot_folder_path, f'plot_{iteration_idx}'),
                              verbose=False)
            optimizer.step()
            if is_iteration_to_action(iteration_idx=iteration_idx, action='save'):
                save_obj_to_disk(path=Path(image_plot_folder_path, 'losses'), obj=prediction_losses)
                save_obj_to_disk(path=Path(image_plot_folder_path, 'total_losses'), obj=total_losses)
            #     save_obj_to_disk(path=Path(image_plot_folder_path, 'tokens_mask'), obj=tokens_mask)
        minimum_predictions = get_minimum_predictions_string(image_name=image_name, total_losses=total_losses, prediction_losses=prediction_losses)
        save_text_to_file(path=image_plot_folder_path, file_name='minimum_predictions', text=minimum_predictions)
        print(minimum_predictions)

        # save_obj_to_disk(path=Path(image_plot_folder_path, 'temp'),
        #                  obj=x_attention[get_top_k_mimimum_values_indices(array=prediction_losses, k=1)])


experiment_name = f"bias_mean_t_h_l_lr_{str(vit_config['lr']).replace('.', '_')}+l1_{loss_config['l1_loss_multiplier']}+kl_loss_{loss_config['kl_loss_multiplier']}+entropy_loss_{loss_config['entropy_loss_multiplier']}+pred_loss_{loss_config['pred_loss_multiplier']}+other_loss_mul_{loss_config['other_loss_multiplier']}"

if __name__ == '__main__':
    log_run = configure_log(vit_config=vit_config, experiment_name=experiment_name)
    os.makedirs(name=Path(PLOTS_PATH, experiment_name), exist_ok=True)
    optimize_params(vit_model=vit_model, criterion=objective_temp_bias_softmax, log_run=log_run)
