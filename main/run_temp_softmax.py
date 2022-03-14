from tqdm import tqdm
from utils import *
from loss_utils import *
import wandb
from log_utils import get_wandb_config
from utils.consts import *
from pytorch_lightning import seed_everything
from typing import Callable
from objectives import objective_temp_softmax
from time import time

vit_config = config['vit']
loss_config = vit_config['loss']

seed_everything(config['general']['seed'])
feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config)


def optimize_params(vit_model: ViTForImageClassification, criterion: Callable):
    for idx, image_dict in enumerate(vit_config['images']):
        image_name, correct_class_idx, contrastive_class_idx = get_image_spec(image_dict)
        wandb_config = get_wandb_config(vit_config=vit_config, experiment_name=experiment_name, image_name=image_name)
        start_time = time()
        with wandb.init(project=config['general']['wandb_project'], entity=config['general']['wandb_entity'],
                        config=wandb_config) as run:
            vit_ours_model, optimizer = setup_model_and_optimizer(model_name='softmax_temp')

            image_plot_folder_path = get_and_create_image_plot_folder_path(images_folder_path=IMAGES_FOLDER_PATH,
                                                                           experiment_name=experiment_name,
                                                                           image_name=image_name)

            inputs, original_transformed_image = get_image_and_inputs_and_transformed_image(image_name=image_name,
                                                                                            feature_extractor=feature_extractor)
            target = vit_model(**inputs)
            target_class_idx = torch.argmax(target.logits[0])

            total_losses, prediction_losses, correct_class_logits, correct_class_probs, tokens_mask, temps = [], [], [], [], [], []

            max_folder, mean_folder, median_folder, min_folder, objects_path, temp_tokens_max_folder, \
            temp_tokens_mean_folder, temp_tokens_median_folder, temp_tokens_min_folder = start_run(
                model=vit_ours_model, image_plot_folder_path=image_plot_folder_path, inputs=inputs, run=run)

            for iteration_idx in tqdm(range(vit_config['num_steps'])):
                optimizer.zero_grad()
                output = vit_ours_model(**inputs)

                correct_class_logit, correct_class_prob, prediction_loss = get_iteration_target_class_stats(
                    output=output, target_class_idx=target_class_idx)
                temps.append(vit_ours_model.vit.encoder.x_attention.clone())
                loss = criterion(output=output.logits, target=target.logits,
                                 temp=vit_ours_model.vit.encoder.x_attention)
                loss.backward()

                compare_results_each_n_steps(iteration_idx=iteration_idx, target=target.logits, output=output.logits,
                                             prev_x_attention=vit_ours_model.vit.encoder.x_attention,
                                             sampled_binary_patches=None)
                cls_attentions_probs = get_attention_probs_by_layer_of_the_CLS(model=vit_ours_model)
                if vit_config['plot_visualizations']:
                    temp = visualize_temp_tokens_and_attention_scores(iteration_idx=iteration_idx,
                                                                      max_folder=max_folder,
                                                                      mean_folder=mean_folder,
                                                                      median_folder=median_folder,
                                                                      min_folder=min_folder,
                                                                      original_transformed_image=original_transformed_image,
                                                                      temp_tokens_max_folder=temp_tokens_max_folder,
                                                                      temp_tokens_mean_folder=temp_tokens_mean_folder,
                                                                      temp_tokens_median_folder=temp_tokens_median_folder,
                                                                      temp_tokens_min_folder=temp_tokens_min_folder,
                                                                      vit_sigmoid_model=vit_ours_model, cls_attentions_probs=cls_attentions_probs)

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
                              temps=temps, vit_sigmoid_model=vit_ours_model)
            print(f'Image Run Time: {(time() - start_time) / 60} minutes')


if __name__ == '__main__':
    # prediction_loss_multipliers = [10, 100, 1000]
    # entropy_loss_multipliers = [10, 100, 1000]
    # for pred_loss_multiplier in prediction_loss_multipliers:
    #     for entropy_loss_multiplier in entropy_loss_multipliers:
    #         loss_config['pred_loss_multiplier'] = pred_loss_multiplier
    #         loss_config['entropy_loss_multiplier'] = entropy_loss_multiplier
    experiment_name = f"visu_mul_{vit_config['objective']}_lr{str(vit_config['lr']).replace('.', '_')}+l1_{loss_config['l1_loss_multiplier']}+kl_loss_{loss_config['kl_loss_multiplier']}+entropy_loss_{loss_config['entropy_loss_multiplier']}+pred_loss_{loss_config['pred_loss_multiplier']}"
    print(experiment_name)
    _ = create_folder(Path(PLOTS_PATH, experiment_name))
    optimize_params(vit_model=vit_model, criterion=objective_temp_softmax)
