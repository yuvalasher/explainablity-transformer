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
    # l1_loss = torch.mean(torch.abs(temp)) * loss_config['l1_loss_multiplier']
    # print(f'entropy: {entropy_loss}, prediction_loss: {prediction_loss}, l1_loss: {l1_loss}')
    print(f'entropy: {entropy_loss}, prediction_loss: {prediction_loss}')
    loss = entropy_loss + prediction_loss
    print(f'loss: {loss}, max_temp: {torch.max(temp)}, min_temp: {torch.min(temp)}')

    log(loss=loss, entropy_loss=entropy_loss, prediction_loss=prediction_loss, x_attention=temp, output=output,
        target=target)
    return loss


# def get_target_class_idx(correct_class_idx: int, contrastive_class_idx: int = None) -> int:
#     target_class_idx = contrastive_class_idx if contrastive_class_idx is not None else correct_class_idx
#     return target_class_idx


def optimize_params(vit_model: ViTForImageClassification, criterion: Callable):
    for idx, image_dict in enumerate(vit_config['images']):
        image_name, correct_class_idx, contrastive_class_idx = get_image_spec(image_dict)
        wandb_config = get_wandb_config(vit_config=vit_config, experiment_name=experiment_name, image_name=image_name)
        with wandb.init(project=config['general']['wandb_project'], entity=config['general']['wandb_entity'],
                        config=wandb_config) as run:
            vit_sigmoid_model = handle_model_config_and_freezing_for_task(
                model=load_ViTModel(vit_config, model_type='softmax_temp'),
                freezing_transformer=vit_config['freezing_transformer'])
            optimizer = optim.Adam([vit_sigmoid_model.vit.encoder.x_attention], lr=vit_config['lr'])

            image_plot_folder_path = get_and_create_image_plot_folder_path(images_folder_path=IMAGES_FOLDER_PATH,
                                                                           experiment_name=experiment_name,
                                                                           image_name=image_name)

            image = get_image_from_path(Path(IMAGES_FOLDER_PATH, image_name))
            inputs = feature_extractor(images=image, return_tensors="pt")
            original_transformed_image = image_transformations(image)
            target = vit_model(**inputs)
            target_class_idx = torch.argmax(target.logits[0])

            total_losses, prediction_losses, correct_class_logits, correct_class_probs, tokens_mask, x_attention = [], [], [], [], [], []

            max_folder, mean_folder, median_folder, min_folder, objects_path, temp_tokens_max_folder, \
            temp_tokens_mean_folder, temp_tokens_median_folder, temp_tokens_min_folder = start_run(
                model=vit_sigmoid_model, image_plot_folder_path=image_plot_folder_path, inputs=inputs, run=run)
            for iteration_idx in tqdm(range(vit_config['num_steps'])):
                optimizer.zero_grad()
                output = vit_sigmoid_model(**inputs)

                correct_class_logit, correct_class_prob, prediction_loss = get_iteration_target_class_stats(
                    output=output, target_class_idx=target_class_idx)
                x_attention.append(vit_sigmoid_model.vit.encoder.x_attention.clone())
                loss = criterion(output=output.logits, target=target.logits,
                                 temp=vit_sigmoid_model.vit.encoder.x_attention)
                loss.backward()

                compare_results_each_n_steps(iteration_idx=iteration_idx, target=target.logits, output=output.logits,
                                             prev_x_attention=vit_sigmoid_model.vit.encoder.x_attention,
                                             sampled_binary_patches=None)

                cls_attentions_probs, temp = visualize_temp_tokens_and_attention_scores(iteration_idx=iteration_idx,
                                                                                        max_folder=max_folder,
                                                                                        mean_folder=mean_folder,
                                                                                        median_folder=median_folder,
                                                                                        min_folder=min_folder,
                                                                                        original_transformed_image=original_transformed_image,
                                                                                        temp_tokens_max_folder=temp_tokens_max_folder,
                                                                                        temp_tokens_mean_folder=temp_tokens_mean_folder,
                                                                                        temp_tokens_median_folder=temp_tokens_median_folder,
                                                                                        temp_tokens_min_folder=temp_tokens_min_folder,
                                                                                        vit_sigmoid_model=vit_sigmoid_model)

                correct_class_logits.append(correct_class_logit)
                correct_class_probs.append(correct_class_logit)
                prediction_losses.append(prediction_loss)
                total_losses.append(loss.item())
                tokens_mask.append(cls_attentions_probs.clone())
                optimizer.step()

                end_iteration(correct_class_logits=correct_class_logits, correct_class_probs=correct_class_probs,
                              image_name=image_name, image_plot_folder_path=image_plot_folder_path,
                              iteration_idx=iteration_idx, objects_path=objects_path,
                              prediction_losses=prediction_losses, tokens_mask=tokens_mask, total_losses=total_losses,
                              vit_sigmoid_model=vit_sigmoid_model)


if __name__ == '__main__':
    # prediction_loss_multipliers = [10, 100, 1000]
    # entropy_loss_multipliers = [10, 100, 1000]
    # for pred_loss_multiplier in prediction_loss_multipliers:
    #     for entropy_loss_multiplier in entropy_loss_multipliers:
    #         loss_config['pred_loss_multiplier'] = pred_loss_multiplier
    #         loss_config['entropy_loss_multiplier'] = entropy_loss_multiplier
    experiment_name = f"vis_mul_{vit_config['objective']}_lr{str(vit_config['lr']).replace('.', '_')}+l1_{loss_config['l1_loss_multiplier']}+kl_loss_{loss_config['kl_loss_multiplier']}+entropy_loss_{loss_config['entropy_loss_multiplier']}+pred_loss_{loss_config['pred_loss_multiplier']}"
    print(experiment_name)
    os.makedirs(name=Path(PLOTS_PATH, experiment_name), exist_ok=True)
    optimize_params(vit_model=vit_model, criterion=objective_temp_softmax)
