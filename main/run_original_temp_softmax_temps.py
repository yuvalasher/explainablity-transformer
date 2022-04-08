from tqdm import tqdm
# from utils.utils import *
from loss_utils import *
import wandb
from log_utils import get_wandb_config
from main.rollout_grad import get_rollout_grad
from utils.consts import *
from pytorch_lightning import seed_everything
from typing import Callable
from objectives import objective_temp_softmax_test
from time import time

vit_config = config['vit']
loss_config = vit_config['loss']

seed_everything(config['general']['seed'])
feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config, model_type='vit-for-dino',
                                                                    is_wolf_transforms=vit_config['is_wolf_transforms'])


def optimize_params(vit_model: ViTForImageClassification, criterion: Callable):
    for idx, image_dict in enumerate(vit_config['images']):
        image_name, correct_class_idx, contrastive_class_idx = get_image_spec(image_dict)
        wandb_config = get_wandb_config(vit_config=vit_config, experiment_name=experiment_name, image_name=image_name)
        with wandb.init(project=config['general']['wandb_project'], entity=config['general']['wandb_entity'],
                        config=wandb_config) as run:
            vit_ours_model, optimizer = setup_model_and_optimizer(model_name='softmax_temp')

            image_plot_folder_path = get_and_create_image_plot_folder_path(images_folder_path=IMAGES_FOLDER_PATH,
                                                                           experiment_name=experiment_name,
                                                                           image_name=image_name, save_image=False)
            # mean_temp_rollout_max_grad_folder = create_folder(Path(image_plot_folder_path, 'mean_temp_rollout_max_grad'))
            # median_temp_rollout_max_grad_folder = create_folder(Path(image_plot_folder_path, 'median_temp_rollout_max_grad'))
            # mean_temp_rollout_relu_grad_mean_folder = create_folder(Path(image_plot_folder_path, 'mean_temp_rollout_relu_grad_mean'))
            # median_temp_rollout_relu_grad_mean_folder = create_folder(Path(image_plot_folder_path, 'median_temp_rollout_relu_grad_mean'))
            inputs, original_transformed_image = get_image_and_inputs_and_transformed_image(image_name=image_name,
                                                                                            feature_extractor=feature_extractor)
            target = vit_model(**inputs)
            target_class_idx = torch.argmax(target.logits[0])

            total_losses, prediction_losses, correct_class_logits, correct_class_probs, tokens_mask, temps = [], [], [], [], [], []

            # max_folder, mean_folder, median_folder, min_folder, objects_path, temp_tokens_max_folder, \
            # temp_tokens_mean_folder, temp_tokens_median_folder, temp_tokens_min_folder = start_run_save_files_plot_visualizations_create_folders(
            #     model=vit_ours_model, image_plot_folder_path=image_plot_folder_path, inputs=inputs, run=None,
            #     original_image=original_transformed_image)

            # _ = vit_ours_model(**inputs)
            # attention_probs = get_attention_probs(model=vit_ours_model)
            # mask_rollout_max = rollout(attentions=attention_probs, head_fusion='max', discard_ratio=0)
            d_masks = get_rollout_grad(vit_ours_model=vit_ours_model,
                                       feature_extractor=feature_extractor,
                                       inputs=inputs,
                                       discard_ratio=0.9, return_resized=False)
            for iteration_idx in tqdm(range(vit_config['num_steps'])):
                optimizer.zero_grad()
                output = vit_ours_model(**inputs)
                # correct_class_logit, correct_class_prob, prediction_loss = get_iteration_target_class_stats(
                #     output=output, target_class_idx=target_class_idx)
                # temps.append(vit_ours_model.vit.encoder.x_attention.clone())
                loss = criterion(output=output.logits, target=target.logits,
                                 temp=vit_ours_model.vit.encoder.x_attention)
                loss.backward()

                if vit_config['verbose']:
                    compare_results_each_n_steps(iteration_idx=iteration_idx, target=target.logits,
                                                 output=output.logits,
                                                 prev_x_attention=vit_ours_model.vit.encoder.x_attention,
                                                 sampled_binary_patches=None)
                temp = vit_ours_model.vit.encoder.x_attention.clone()
                temp = get_temp_to_visualize(temp)
                cls_attentions_probs = get_attention_probs_by_layer_of_the_CLS(model=vit_ours_model)



                if vit_config['plot_visualizations'] and iteration_idx > 100:
                    visu(original_image=original_transformed_image,
                         transformer_attribution=d_masks['rollout_max_grad'],
                         file_name=Path(image_plot_folder_path, f'max_grad'))

                    visu(original_image=original_transformed_image,
                         transformer_attribution=d_masks['rollout_max_grad'],
                         file_name=Path(image_plot_folder_path, f'relu_grad_mean'))

                    visu(original_image=original_transformed_image,
                         transformer_attribution=temp.mean(dim=0) * d_masks['rollout_mean_relu_grad'],
                         file_name=Path(image_plot_folder_path, f'max_grad_mean_temp_rollout'))

                    visu(original_image=original_transformed_image,
                         transformer_attribution=temp.median(dim=0)[0] * d_masks['rollout_mean_relu_grad'],
                         file_name=Path(image_plot_folder_path, f'max_grad_median_temp_rollout'))
                    #
                    # visu(original_image=original_transformed_image,
                    #      transformer_attribution=temp.mean(dim=0) * d_masks['rollout_max_grad'],
                    #      file_name=Path(image_plot_folder_path, f'relu_grad_mean_mean_temp_rollout'))
                    #
                    # visu(original_image=original_transformed_image,
                    #      transformer_attribution=temp.median(dim=0)[0] * d_masks['rollout_max_grad'],
                    #      file_name=Path(image_plot_folder_path, f'relu_grad_mean_median_temp_rollout'))
                    #
                    visu(original_image=original_transformed_image,
                         transformer_attribution=F.softmax(temp, dim=1).median(dim=0)[0],
                         file_name=Path(image_plot_folder_path, f'temp_softmax_median_{iteration_idx}'))
                    visu(original_image=original_transformed_image,
                         transformer_attribution=temp.median(dim=0)[0],
                         file_name=Path(image_plot_folder_path, f'temp_median_{iteration_idx}'))
                    visu(original_image=original_transformed_image,
                         transformer_attribution=temp.mean(dim=0),
                         file_name=Path(image_plot_folder_path, f'temp_mean_{iteration_idx}'))
                    for head_idx in range(12):
                        visu(original_image=original_transformed_image,
                             transformer_attribution=temp[head_idx],
                             file_name=Path(image_plot_folder_path, f'temp_head_{head_idx}'))
                    print(1)
                    # visualize_attention_scores(cls_attentions_probs=cls_attentions_probs, iteration_idx=iteration_idx,
                    #                            max_folder=max_folder, mean_folder=mean_folder,
                    #                            median_folder=median_folder,
                    #                            min_folder=min_folder,
                    #                            original_transformed_image=original_transformed_image)
                    # visualize_attention_scores_with_rollout(original_transformed_image=original_transformed_image,
                    #                                         rollout_vector=mask_rollout_max,
                    #                                         cls_attentions_probs=cls_attentions_probs,
                    #                                         iteration_idx=iteration_idx, max_folder=max_folder,
                    #                                         min_folder=min_folder, median_folder=median_folder,
                    #                                         mean_folder=mean_folder)
                # temp = visualize_temp_tokens_and_attention_scores(iteration_idx=iteration_idx,
                #                                                   original_transformed_image=original_transformed_image,
                #                                                   vit_sigmoid_model=vit_ours_model,
                #                                                   cls_attentions_probs=cls_attentions_probs,
                #                                                   temp_tokens_mean_folder=temp_tokens_mean_folder,
                #                                                   temp_tokens_median_folder=temp_tokens_median_folder)
                # temp = visualize_temp_tokens_and_attention_scores(iteration_idx=iteration_idx,
                #                                                   max_folder=max_folder,
                #                                                   mean_folder=mean_folder,
                #                                                   median_folder=median_folder,
                #                                                   min_folder=min_folder,
                #                                                   original_transformed_image=original_transformed_image,
                #                                                   temp_tokens_mean_folder=temp_tokens_mean_folder,
                #                                                   temp_tokens_median_folder=temp_tokens_median_folder,
                #                                                   vit_sigmoid_model=vit_ours_model,
                #                                                   cls_attentions_probs=cls_attentions_probs)
                #
                # correct_class_logits.append(correct_class_logit)
                # correct_class_probs.append(correct_class_prob)
                # prediction_losses.append(prediction_loss)
                # total_losses.append(loss.item())
                # tokens_mask.append(cls_attentions_probs.clone())

                optimizer.step()

                # end_iteration(correct_class_logits=correct_class_logits, correct_class_probs=correct_class_probs,
                #               image_name=image_name, image_plot_folder_path=image_plot_folder_path,
                #               iteration_idx=iteration_idx, objects_path=objects_path,
                #               prediction_losses=prediction_losses, tokens_mask=tokens_mask, total_losses=total_losses,
                #               temps=temps, vit_sigmoid_model=vit_ours_model)


if __name__ == '__main__':
    experiment_name = 'temp_softmax_rollout_grad_with_logits_loss_test'
    # experiment_name = f"rollout_temp_mul_{vit_config['objective']}_lr{str(vit_config['lr']).replace('.', '_')}+l1_{loss_config['l1_loss_multiplier']}+kl_loss_{loss_config['kl_loss_multiplier']}+entropy_loss_{loss_config['entropy_loss_multiplier']}+pred_loss_{loss_config['pred_loss_multiplier']}"
    print(experiment_name)
    _ = create_folder(Path(PLOTS_PATH, experiment_name))
    optimize_params(vit_model=vit_model, criterion=objective_temp_softmax_test)