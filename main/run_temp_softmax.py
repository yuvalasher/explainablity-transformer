from torchvision.transforms import transforms
from tqdm import tqdm
# from utils.utils import *
from loss_utils import *
import wandb
from log_utils import get_wandb_config
from main.rollout_grad import get_rollout_grad
from utils.consts import *
from pytorch_lightning import seed_everything
from typing import Callable
from objectives import objective_temp_softmax
from time import time

vit_config = config['vit']
loss_config = vit_config['loss']

seed_everything(config['general']['seed'])
feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config, model_type='vit-for-dino',
                                                                    is_wolf_transforms=vit_config['is_wolf_transforms'])


def optimize_params(vit_model: ViTForImageClassification, criterion: Callable):
    vit_ours_model = handle_model_config_and_freezing_for_task(
        model=load_ViTModel(vit_config, model_type='softmax_temp'),
        freezing_transformer=vit_config['freezing_transformer'])

    for idx, image_dict in enumerate(vit_config['images']):
        image_name, correct_class_idx, contrastive_class_idx = get_image_spec(image_dict)
        # wandb_config = get_wandb_config(vit_config=vit_config, experiment_name=experiment_name, image_name=image_name)
        # start_time = time()
        # with wandb.init(project=config['general']['wandb_project'], entity=config['general']['wandb_entity'],
        #                 config=wandb_config) as run:
        # vit_ours_model, optimizer = setup_model_and_optimizer(model_name='softmax_temp')
        vit_ours_model.vit.encoder.x_attention.data = nn.Parameter(
            torch.ones_like(vit_ours_model.vit.encoder.x_attention))
        optimizer = optim.Adam([vit_ours_model.vit.encoder.x_attention], lr=vit_config['lr'])
        image_plot_folder_path = get_and_create_image_plot_folder_path(images_folder_path=IMAGES_FOLDER_PATH,
                                                                       experiment_name=experiment_name,
                                                                       image_name=image_name, save_image=False)

        inputs, original_transformed_image = get_image_and_inputs_and_transformed_image(image_name=image_name,
                                                                                        feature_extractor=feature_extractor,
                                                                                        is_wolf_transforms=vit_config[
                                                                                            'is_wolf_transforms'])
        target = vit_model(**inputs)
        target_class_idx = torch.argmax(target.logits[0])

        # total_losses, prediction_losses, correct_class_logits, correct_class_probs, tokens_mask, temps = [], [], [], [], [], []
        # gradients_list = []

        # max_folder, mean_folder, median_folder, min_folder, objects_path, temp_tokens_max_folder, \
        # temp_tokens_mean_folder, temp_tokens_median_folder, temp_tokens_min_folder = start_run_save_files_plot_visualizations_create_folders(
        #     model=vit_ours_model, image_plot_folder_path=image_plot_folder_path, inputs=inputs, run=run,
        #     original_image=original_transformed_image)
        # rollout_folder = create_folder(Path(image_plot_folder_path, 'rollout'))
        # objects_path = create_folder(Path(image_plot_folder_path, 'objects'))
        # rollout_min_folder_mean_relu_grad = create_folder(
        #     Path(image_plot_folder_path, 'grad_rollout', 'min', 'rollout_mean_relu_grad'))
        rollout_median_folder_mean_relu_grad = create_folder(
            Path(image_plot_folder_path, 'grad_rollout', 'median', 'rollout_mean_relu_grad'))
        # rollout_max_folder_mean_relu_grad = create_folder(
        #     Path(image_plot_folder_path, 'grad_rollout', 'max', 'rollout_mean_relu_grad'))
        # rollout_mean_folder_mean_relu_grad = create_folder(
        #     Path(image_plot_folder_path, 'grad_rollout', 'mean', 'rollout_mean_relu_grad'))
        #
        # rollout_min_folder_max_grad = create_folder(
        #     Path(image_plot_folder_path, 'grad_rollout', 'min', 'rollout_max_grad'))
        rollout_median_folder_max_grad = create_folder(
            Path(image_plot_folder_path, 'grad_rollout', 'median', 'rollout_max_grad'))
        # rollout_max_folder_max_grad = create_folder(
        #     Path(image_plot_folder_path, 'grad_rollout', 'max', 'rollout_max_grad'))
        # rollout_mean_folder_max_grad = create_folder(
        #     Path(image_plot_folder_path, 'grad_rollout', 'mean', 'rollout_max_grad'))
        print_number_of_trainable_and_not_trainable_params(model=vit_ours_model)
        _ = vit_ours_model(**inputs)
        d_masks = get_rollout_grad(vit_ours_model=vit_ours_model,
                                   feature_extractor=feature_extractor,
                                   inputs=inputs,
                                   discard_ratio=0.9, return_resized=False)
        for iteration_idx in tqdm(range(vit_config['num_steps'])):
            optimizer.zero_grad()
            output = vit_ours_model(**inputs)
            # attention_probs = get_attention_probs(model=vit_ours_model)
            # temps.append(vit_ours_model.vit.encoder.x_attention.clone())
            # correct_class_logit, correct_class_prob, prediction_loss = get_iteration_target_class_stats(
            #     output=output, target_class_idx=target_class_idx)
            loss = criterion(output=output.logits, target=target.logits,
                             temp=vit_ours_model.vit.encoder.x_attention)
            loss.backward()

            cls_attentions_probs = get_attention_probs_by_layer_of_the_CLS(model=vit_ours_model)

            if vit_config['plot_visualizations']:
                visu(original_image=original_transformed_image,
                     transformer_attribution=cls_attentions_probs.median(dim=0)[0] * d_masks['rollout_mean_relu_grad'],
                     file_name=Path(rollout_median_folder_mean_relu_grad, f'plot_{iteration_idx}'))
                visu(original_image=original_transformed_image,
                     transformer_attribution=cls_attentions_probs.median(dim=0)[0] * d_masks['rollout_max_grad'],
                     file_name=Path(rollout_median_folder_max_grad, f'plot_{iteration_idx}'))
                # visualize_attention_scores_with_rollout(cls_attentions_probs=cls_attentions_probs,
                #                                         rollout_vector=d_masks['rollout_mean_relu_grad'],
                #                                         iteration_idx=iteration_idx,
                #                                         max_folder=rollout_max_folder_mean_relu_grad,
                #                                         mean_folder=rollout_mean_folder_mean_relu_grad,
                #                                         median_folder=rollout_median_folder_mean_relu_grad,
                #                                         min_folder=rollout_min_folder_mean_relu_grad,
                #                                         original_transformed_image=original_transformed_image)
                #
                # visualize_attention_scores_with_rollout(cls_attentions_probs=cls_attentions_probs,
                #                                         rollout_vector=d_masks['rollout_max_grad'],
                #                                         iteration_idx=iteration_idx,
                #                                         max_folder=rollout_max_folder_max_grad,
                #                                         mean_folder=rollout_mean_folder_max_grad,
                #                                         median_folder=rollout_median_folder_max_grad,
                #                                         min_folder=rollout_min_folder_max_grad,
                #                                         original_transformed_image=original_transformed_image)
            optimizer.step()

            # end_iteration(correct_class_logits=correct_class_logits, correct_class_probs=correct_class_probs,
            #               image_name=image_name, image_plot_folder_path=image_plot_folder_path,
            #               iteration_idx=iteration_idx, objects_path=objects_path,
            #               prediction_losses=prediction_losses, tokens_mask=tokens_mask, total_losses=total_losses,
            #               temps=temps, vit_sigmoid_model=vit_ours_model, gradients=gradients_list)
        # print(f'*********************** Image Run Time: {(time() - start_time) / 60} minutes ***********************')


if __name__ == '__main__':
    experiment_name = f"grad_rollout_temp_softmax"
    print(experiment_name)
    _ = create_folder(Path(PLOTS_PATH, experiment_name))
    optimize_params(vit_model=vit_model, criterion=objective_temp_softmax)
