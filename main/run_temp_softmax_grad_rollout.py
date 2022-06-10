from tqdm import tqdm
# from utils.utils import *
from loss_utils import *
from utils.consts import *
from pytorch_lightning import seed_everything
from typing import Callable
from objectives import objective_temp_softmax, objective_grad_rollout

vit_config = config['vit']
loss_config = vit_config['loss']

seed_everything(config['general']['seed'])
feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config, model_type='vit-for-dino',
                                                                    is_wolf_transforms=vit_config['is_wolf_transforms'])
vit_unfreezed = handle_model_config_and_freezing_for_task(
    model=load_ViTModel(vit_config, model_type='vit-for-dino-grad'), freezing_transformer=False)


def optimize_params(vit_model: ViTForImageClassification, vit_ours_model, feature_extractor, criterion: Callable):
    run = None
    for idx, image_dict in enumerate(vit_config['images']):
        image_name, correct_class_idx, contrastive_class_idx = get_image_spec(image_dict)
        image_plot_folder_path = get_and_create_image_plot_folder_path(images_folder_path=IMAGES_FOLDER_PATH,
                                                                       experiment_name=experiment_name,
                                                                       image_name=image_name)

        inputs, original_transformed_image = get_image_and_inputs_and_transformed_image(image_name=image_name,
                                                                                        feature_extractor=feature_extractor)
        # target = vit_model(**inputs)
        target = vit_unfreezed(**inputs)
        target_class_idx = torch.argmax(target.logits[0])
        loss = objective_grad_rollout(output=target.logits, target_idx=target_class_idx)
        loss.backward()
        attention_probs = get_attention_probs(model=vit_unfreezed)
        gradients = get_attention_grads_probs(model=vit_unfreezed, apply_relu=False)
        rollout_max_grad = rollout(attentions=attention_probs, head_fusion='max', gradients=gradients).reshape(-1)
        priors = torch.cat((torch.ones(1), rollout_max_grad)).expand(vit_model.config.num_hidden_layers,
                                                                     vit_model.config.num_attention_heads,
                                                                     len(rollout_max_grad) + 1)
        vit_ours_model.vit.encoder.x_attention.data = nn.Parameter(priors.clone())
        optimizer = optim.Adam([vit_ours_model.vit.encoder.x_attention], lr=vit_config['lr'])
        visu(original_image=original_transformed_image,
             transformer_attribution=rollout_max_grad,
             file_name=Path(image_plot_folder_path, f'grad_rollout_max'))
        visu(original_image=original_transformed_image,
             transformer_attribution=rollout_max_grad.clip(min=0),
             file_name=Path(image_plot_folder_path, f'grad_rollout_max_clipped'))
        total_losses, prediction_losses, correct_class_logits, correct_class_probs, tokens_mask, temps = [], [], [], [], [], []

        max_folder, mean_folder, median_folder, min_folder, objects_path, temp_tokens_max_folder, \
        temp_tokens_mean_folder, temp_tokens_median_folder, temp_tokens_min_folder = start_run_save_files_plot_visualizations_create_folders(
            model=vit_ours_model, image_plot_folder_path=image_plot_folder_path, inputs=inputs, run=run,
            original_image=original_transformed_image)
        temps.append(vit_ours_model.vit.encoder.x_attention.clone())
        for iteration_idx in tqdm(range(vit_config['num_steps'])):
            optimizer.zero_grad()
            output = vit_ours_model(**inputs)

            correct_class_logit, correct_class_prob, prediction_loss = get_iteration_target_class_stats(
                output=output, target_class_idx=target_class_idx)
            loss = criterion(output=output.logits, target=target.logits,
                             temp=vit_ours_model.vit.encoder.x_attention)
            loss.backward()

            if vit_config['verbose']:
                compare_results_each_n_steps(iteration_idx=iteration_idx, target=target.logits,
                                             output=output.logits,
                                             prev_x_attention=vit_ours_model.vit.encoder.x_attention,
                                             sampled_binary_patches=None)
            cls_attentions_probs = get_attention_probs_by_layer_of_the_CLS(model=vit_ours_model)
            if vit_config['plot_visualizations']:
                visualize_attention_scores_with_rollout(original_transformed_image=original_transformed_image,
                                                        rollout_vector=rollout_max_grad,
                                                        cls_attentions_probs=cls_attentions_probs,
                                                        iteration_idx=iteration_idx, median_folder=median_folder)
            # temp = visualize_temp_tokens_and_attention_scores(iteration_idx=iteration_idx,
            #                                                   max_folder=max_folder,
            #                                                   mean_folder=mean_folder,
            #                                                   median_folder=median_folder,
            #                                                   min_folder=min_folder,
            #                                                   original_transformed_image=original_transformed_image,
            #                                                   temp_tokens_max_folder=temp_tokens_max_folder,
            #                                                   temp_tokens_mean_folder=temp_tokens_mean_folder,
            #                                                   temp_tokens_median_folder=temp_tokens_median_folder,
            #                                                   temp_tokens_min_folder=temp_tokens_min_folder,
            #                                                   vit_sigmoid_model=vit_ours_model,
            #                                                   cls_attentions_probs=cls_attentions_probs)

            correct_class_logits.append(correct_class_logit)
            correct_class_probs.append(correct_class_prob)
            prediction_losses.append(prediction_loss)
            total_losses.append(loss.item())
            tokens_mask.append(cls_attentions_probs.clone())

            optimizer.step()

            end_iteration(correct_class_logits=correct_class_logits, correct_class_probs=correct_class_probs,
                          image_name=image_name, image_plot_folder_path=image_plot_folder_path,
                          iteration_idx=iteration_idx, objects_path=objects_path,
                          prediction_losses=prediction_losses, tokens_mask=tokens_mask, total_losses=total_losses,
                          temps=temps, vit_sigmoid_model=vit_ours_model)


if __name__ == '__main__':
    experiment_name = f"temp_softmax_from_grad_rollout_priors_mul"
    # experiment_name = f"rollout_temp_mul_{vit_config['objective']}_lr{str(vit_config['lr']).replace('.', '_')}+l1_{loss_config['l1_loss_multiplier']}+kl_loss_{loss_config['kl_loss_multiplier']}+entropy_loss_{loss_config['entropy_loss_multiplier']}+pred_loss_{loss_config['pred_loss_multiplier']}"
    print(experiment_name)
    _ = create_folder(Path(PLOTS_PATH, experiment_name))
    vit_ours_model, optimizer = setup_model_and_optimizer(model_name='softmax_temp')
    optimize_params(vit_model=vit_model, vit_ours_model=vit_ours_model, feature_extractor=feature_extractor,
                    criterion=objective_temp_softmax)
