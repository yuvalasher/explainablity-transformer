from tqdm import tqdm
from torch import optim
from utils import *
from loss_utils import *
import wandb
from log_utils import get_wandb_config
from utils.consts import *
from pytorch_lightning import seed_everything
from utils.transformation import image_transformations
from typing import Callable

vit_config = config['vit']
loss_config = vit_config['loss']

seed_everything(config['general']['seed'])
feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config)
vit_infer = handle_model_config_and_freezing_for_task(model=load_ViTModel(vit_config, model_type='infer'))


def objective_temp_softmax(output: Tensor, target: Tensor, temp: Tensor) -> Tensor:
    prediction_loss = ce_loss(output, torch.argmax(target).unsqueeze(0)) * loss_config['pred_loss_multiplier']
    entropy_loss = entropy(F.softmax(temp, dim=-1)) * loss_config['entropy_loss_multiplier']
    print(f'entropy: {entropy_loss}, prediction_loss: {prediction_loss}')
    loss = entropy_loss + prediction_loss
    print(f'loss: {loss}, max_temp: {torch.max(temp)}, min_temp: {torch.min(temp)}')

    log(loss=loss, entropy_loss=entropy_loss, prediction_loss=prediction_loss, x_attention=temp, output=output,
        target=target)
    return loss


def optimize_params(vit_model: ViTForImageClassification, criterion: Callable):
    for idx, image_dict in enumerate(vit_config['images']):
        image_name, correct_class_idx, contrastive_class_idx = image_dict['image_name'], image_dict['correct_class'], \
                                                               image_dict['contrastive_class']
        wandb_config = get_wandb_config(vit_config=vit_config, experiment_name=experiment_name, image_name=image_name)
        with wandb.init(project=config['general']['wandb_project'], entity=config['general']['wandb_entity'], config=wandb_config) as run:
            vit_sigmoid_model = handle_model_config_and_freezing_for_task(
                model=load_ViTModel(vit_config, model_type='softmax_temp'),
                freezing_transformer=vit_config['freezing_transformer'])
            optimizer = optim.Adam([vit_sigmoid_model.vit.encoder.x_attention], lr=vit_config['lr'])

            image_plot_folder_path = get_and_create_image_plot_folder_path(images_folder_path=IMAGES_FOLDER_PATH,
                                                                           experiment_name=experiment_name,
                                                                           image_name=image_name)
            save_text_to_file(path=image_plot_folder_path, file_name='metrics_url',
                              text=run.url) if run is not None else ''
            image = get_image_from_path(Path(IMAGES_FOLDER_PATH, image_name))
            inputs = feature_extractor(images=image, return_tensors="pt")
            original_transformed_image = image_transformations(image)
            target = vit_model(**inputs)

            plot_different_visualization_methods(path=image_plot_folder_path, inputs=inputs,
                                                 patch_size=vit_config['patch_size'], vit_config=vit_config)

            total_losses, prediction_losses, correct_class_logits, correct_class_probs, tokens_mask, x_attention = [], [], [], [], [], []
            target_class_idx = torch.argmax(target.logits[0]).item()
            mean_folder, max_folder, min_folder, median_folder, temp_tokens_folder, temp_tokens_mean_folder, temp_tokens_max_folder, temp_tokens_min_folder, temp_tokens_median_folder = create_folders(
                image_plot_folder_path)
            for iteration_idx in tqdm(range(vit_config['num_steps'])):
                optimizer.zero_grad()
                output = vit_sigmoid_model(**inputs)
                correct_class_logits.append(output.logits[0][target_class_idx].item())
                correct_class_probs.append(F.softmax(output.logits[0], dim=-1)[target_class_idx].item())
                prediction_losses.append(ce_loss(output.logits, torch.argmax(target.logits).unsqueeze(0)) * loss_config[
                    'pred_loss_multiplier'])
                x_attention.append(vit_sigmoid_model.vit.encoder.x_attention.clone())
                loss = criterion(output=output.logits, target=target.logits,
                                 temp=vit_sigmoid_model.vit.encoder.x_attention)
                loss.backward()

                compare_results_each_n_steps(iteration_idx=iteration_idx, target=target.logits, output=output.logits,
                                             prev_x_attention=vit_sigmoid_model.vit.encoder.x_attention,
                                             sampled_binary_patches=None)

                cls_attentions_probs = get_attention_probs_by_layer_of_the_CLS(model=vit_sigmoid_model)

                temp = vit_sigmoid_model.vit.encoder.x_attention.clone()
                temp = get_temp_to_visualize(temp)
                if len(temp.shape) == 1:
                    temp_tokens_1d_folder = Path(temp_tokens_folder, '1d_tokens')
                    os.makedirs(name=temp_tokens_1d_folder, exist_ok=True)
                    save_saliency_map(image=original_transformed_image,
                                      saliency_map=torch.tensor(get_scores(temp)).unsqueeze(0),
                                      filename=Path(temp_tokens_1d_folder, f'plot_{iteration_idx}'))
                else:
                    visu(original_image=original_transformed_image,
                         transformer_attribution=cls_attentions_probs.mean(dim=0),
                         file_name=Path(mean_folder, f'plot_{iteration_idx}'))
                    visu(original_image=original_transformed_image,
                         transformer_attribution=cls_attentions_probs.max(dim=0)[0],
                         file_name=Path(max_folder, f'plot_{iteration_idx}'))
                    visu(original_image=original_transformed_image,
                         transformer_attribution=cls_attentions_probs.min(dim=0)[0],
                         file_name=Path(min_folder, f'plot_{iteration_idx}'))
                    visu(original_image=original_transformed_image,
                         transformer_attribution=cls_attentions_probs.median(dim=0)[0],
                         file_name=Path(median_folder, f'plot_{iteration_idx}'))

                    visu(original_image=original_transformed_image,
                         transformer_attribution=temp.mean(dim=0),
                         file_name=Path(temp_tokens_mean_folder, f'plot_{iteration_idx}'))
                    visu(original_image=original_transformed_image,
                         transformer_attribution=temp.max(dim=0)[0],
                         file_name=Path(temp_tokens_max_folder, f'plot_{iteration_idx}'))
                    visu(original_image=original_transformed_image,
                         transformer_attribution=temp.min(dim=0)[0],
                         file_name=Path(temp_tokens_min_folder, f'plot_{iteration_idx}'))
                    visu(original_image=original_transformed_image,
                         transformer_attribution=temp.median(dim=0)[0],
                         file_name=Path(temp_tokens_median_folder, f'plot_{iteration_idx}'))

                total_losses.append(loss.item())
                tokens_mask.append(cls_attentions_probs.clone())
                optimizer.step()

                if is_iteration_to_action(iteration_idx=iteration_idx, action='save'):
                    objects_dict = {'losses': prediction_losses, 'total_losses': total_losses,
                                    'tokens_mask': tokens_mask}
                    save_objects(path=image_plot_folder_path, objects_dict=objects_dict)

            minimum_predictions = get_minimum_predictions_string(image_name=image_name, total_losses=total_losses,
                                                                 prediction_losses=prediction_losses,
                                                                 logits=correct_class_logits,
                                                                 correct_class_probs=correct_class_probs)
            save_text_to_file(path=image_plot_folder_path, file_name='minimum_predictions', text=minimum_predictions)


if __name__ == '__main__':
    experiment_name = f"exp_fusion_per_token_mul_temp_entropy_loss_{loss_config['entropy_loss_multiplier']}+pred_loss_{loss_config['pred_loss_multiplier']}"
    print(experiment_name)
    os.makedirs(name=Path(PLOTS_PATH, experiment_name), exist_ok=True)
    optimize_params(vit_model=vit_model, criterion=objective_temp_softmax)
