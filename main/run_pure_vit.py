from tqdm import tqdm
# from utils.utils import *
from loss_utils import *
from utils.consts import *
from pytorch_lightning import seed_everything
from typing import Callable
from objectives import objective_grad_rollout

vit_config = config['vit']
loss_config = vit_config['loss']

seed_everything(config['general']['seed'])
feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config, model_type='vit-for-dino',
                                                                    is_wolf_transforms=vit_config['is_wolf_transforms'])


def optimize_params(vit_model: ViTForImageClassification, criterion: Callable):
    vit_unfreezed = handle_model_config_and_freezing_for_task(
        model=load_ViTModel(vit_config, model_type='vit-for-dino-grad'),
        freezing_transformer=False)

    for idx, image_dict in enumerate(vit_config['images']):
        image_name, correct_class_idx, contrastive_class_idx = get_image_spec(image_dict)
        optimizer = optim.Adam(vit_unfreezed.parameters(), lr=vit_config['lr'])
        image_plot_folder_path = get_and_create_image_plot_folder_path(images_folder_path=IMAGES_FOLDER_PATH,
                                                                       experiment_name=experiment_name,
                                                                       image_name=image_name, save_image=False)

        inputs, original_transformed_image = get_image_and_inputs_and_transformed_image(image_name=image_name,
                                                                                        feature_extractor=feature_extractor)
        target = vit_model(**inputs)
        optimizer.zero_grad()
        # print_number_of_trainable_and_not_trainable_params(model=vit_unfreezed)
        output = vit_unfreezed(**inputs)
        correct_idx = target.logits.argmax().item()
        # contrastive_idx = 88
        target_idx = correct_idx
        loss = criterion(output=output.logits, target=target.logits, target_idx=target_idx)
        loss.backward()
        # rollout_folder = create_folder(Path(image_plot_folder_path, 'rollout'))
        attention_probs = get_attention_probs(model=vit_model)
        gradients = get_attention_grads_probs(model=vit_unfreezed, apply_relu=False)
        relu_gradients = get_attention_grads_probs(model=vit_unfreezed, apply_relu=True)
        # rollout_max = rollout(attnetions=attention_probs, head_fusion='max')
        rollout_mean_discard = rollout(attnetions=attention_probs, head_fusion='mean', discard_ratio=0.9)
        rollout_mean = rollout(attnetions=attention_probs, head_fusion='mean', discard_ratio=0)
        # rollout_min = rollout(attnetions=attention_probs, head_fusion='min')
        # rollout_median = rollout(attnetions=attention_probs, head_fusion='median')
        # rollout_min_relu_grad = rollout(attnetions=attention_probs, head_fusion='min', gradients=relu_gradients)
        # rollout_median_relu_grad = rollout(attnetions=attention_probs, head_fusion='median', gradients=relu_gradients)
        rollout_max_relu_grad = rollout(attnetions=attention_probs, head_fusion='max', gradients=relu_gradients,
                                        discard_ratio=0.9)
        rollout_mean_relu_grad = rollout(attnetions=attention_probs, head_fusion='mean', gradients=relu_gradients,
                                         discard_ratio=0.9)
        rollout_max_grad = rollout(attnetions=attention_probs, head_fusion='max', gradients=gradients,
                                   discard_ratio=0.9)
        rollout_mean_grad = rollout(attnetions=attention_probs, head_fusion='mean', gradients=gradients,
                                    discard_ratio=0.9)
        rollout_median_grad = rollout(attnetions=attention_probs, head_fusion='median', gradients=gradients,
                                      discard_ratio=0.9)
        # rollout_min_grad = rollout(attnetions=attention_probs, head_fusion='min', gradients=gradients)

        target_desc = vit_unfreezed.config.id2label[target_idx][:10]

        visu(original_image=original_transformed_image,
             transformer_attribution=rollout_max_grad,
             file_name=Path(image_plot_folder_path, f'grad_rollout_max'))
        visu(original_image=original_transformed_image,
             transformer_attribution=rollout_mean_relu_grad,
             file_name=Path(image_plot_folder_path, f'relu_grad_rollout_mean'))
        """
        # visu(original_image=original_transformed_image,
        #      transformer_attribution=relu_gradients[-1][0, :, 0, 1:].median(dim=0)[0],
        #      file_name=Path(image_plot_folder_path, f'relu_grad_last_layer_median'))
        # visu(original_image=original_transformed_image,
        #      transformer_attribution=rollout_max_relu_grad,
        #      file_name=Path(image_plot_folder_path, f'relu_grad_rollout_max'))
        # visu(original_image=original_transformed_image,
        #      transformer_attribution=gradients[-1][0, :, 0, 1:].median(dim=0)[0],
        #       file_name=Path(image_plot_folder_path, f'grad_last_layer_median'))
        # visu(original_image=original_transformed_image,
        #      transformer_attribution=gradients[-1][0, :, 0, 1:].mean(dim=0),
        #       file_name=Path(image_plot_folder_path, f'grad_last_layer_mean'))
        # visu(original_image=original_transformed_image,
        #      transformer_attribution=gradients[-1][0, :, 0, 1:].min(dim=0)[0],
        #       file_name=Path(image_plot_folder_path, f'grad_last_layer_min'))
        # visu(original_image=original_transformed_image,
        #      transformer_attribution=gradients[-1][0, :, 0, 1:].max(dim=0)[0],
        #       file_name=Path(image_plot_folder_path, f'grad_last_layer_max'))
        # visu(original_image=original_transformed_image,
        #      transformer_attribution=relu_gradients[-1][0, :, 0, 1:].mean(dim=0),
        #       file_name=Path(image_plot_folder_path, f'relu_grad_last_layer_mean'))
        # visu(original_image=original_transformed_image,
        #      transformer_attribution=relu_gradients[-1][0, :, 0, 1:].min(dim=0)[0],
        #       file_name=Path(image_plot_folder_path, f'relu_grad_last_layer_min'))
        # visu(original_image=original_transformed_image,
        #      transformer_attribution=relu_gradients[-1][0, :, 0, 1:].max(dim=0)[0],
        #       file_name=Path(image_plot_folder_path, f'relu_grad_last_layer_max'))

        # visu(original_image=original_transformed_image,
        #      transformer_attribution=rollout_mean_grad,
        #       file_name=Path(image_plot_folder_path, f'grad_rollout_mean'))
        # visu(original_image=original_transformed_image,
        #      transformer_attribution=rollout_median_grad,
        #       file_name=Path(image_plot_folder_path, f'grad_rollout_median'))

        visu(original_image=original_transformed_image,
             transformer_attribution=rollout_min_grad,
              file_name=Path(image_plot_folder_path, f'grad_rollout_min'))
        visu(original_image=original_transformed_image,
             transformer_attribution=rollout_min_relu_grad,
              file_name=Path(image_plot_folder_path, f'relu_grad_rollout_min'))
        visu(original_image=original_transformed_image,
             transformer_attribution=rollout_median_relu_grad,
              file_name=Path(image_plot_folder_path, f'relu_grad_rollout_median'))
          visu(original_image=original_transformed_image,
             transformer_attribution=rollout_max,
              file_name=Path(image_plot_folder_path, f'rollout_max'))
        visu(original_image=original_transformed_image,
             transformer_attribution=rollout_mean,
              file_name=Path(image_plot_folder_path, f'rollout_mean'))
        visu(original_image=original_transformed_image,
             transformer_attribution=rollout_min,
              file_name=Path(image_plot_folder_path, f'rollout_min'))
        visu(original_image=original_transformed_image,
             transformer_attribution=rollout_median,
              file_name=Path(image_plot_folder_path, f'rollout_median'))
        for head_idx in range(12):
            visu(original_image=original_transformed_image,
                 transformer_attribution=gradients[-1][:, head_idx, 0, 1:][0],
                 file_name=Path(rollout_folder, f'grad_head_{head_idx}'))
        """
        # optimizer.step()


if __name__ == '__main__':
    experiment_name = f"rollout_grad_0.9_not_wolf"
    print(experiment_name)
    _ = create_folder(Path(PLOTS_PATH, experiment_name))
    optimize_params(vit_model=vit_model, criterion=objective_grad_rollout)
