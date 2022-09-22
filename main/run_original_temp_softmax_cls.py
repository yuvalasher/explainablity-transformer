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

vit_config = config["vit"]
loss_config = vit_config["loss"]

seed_everything(config["general"]["seed"])
feature_extractor, vit_model = load_feature_extractor_and_vit_model(
    vit_config=vit_config,
    model_type="vit-for-dino",
    is_wolf_transforms=vit_config["is_wolf_transforms"],
)


def optimize_params(vit_model: ViTForImageClassification, criterion: Callable):
    for idx, image_dict in enumerate(vit_config["images"]):
        image_name, correct_class_idx, contrastive_class_idx = get_image_spec(image_dict)
        wandb_config = get_wandb_config(
            vit_config=vit_config, experiment_name=experiment_name, image_name=image_name
        )
        # with wandb.init(project=config['general']['wandb_project'], entity=config['general']['wandb_entity'],
        #                 config=wandb_config) as run:
        vit_ours_model, optimizer = setup_model_and_optimizer(model_name="softmax_temp")

        image_plot_folder_path = get_and_create_image_plot_folder_path(
            images_folder_path=IMAGES_FOLDER_PATH,
            experiment_name=experiment_name,
            image_name=image_name,
            save_image=True,
        )
        # mean_temp_rollout_max_grad_folder = create_folder(Path(image_plot_folder_path, 'mean_temp_rollout_max_grad'))
        # median_temp_rollout_max_grad_folder = create_folder(Path(image_plot_folder_path, 'median_temp_rollout_max_grad'))
        # mean_temp_rollout_relu_grad_mean_folder = create_folder(Path(image_plot_folder_path, 'mean_temp_rollout_relu_grad_mean'))
        # median_temp_rollout_relu_grad_mean_folder = create_folder(Path(image_plot_folder_path, 'median_temp_rollout_relu_grad_mean'))
        inputs, original_transformed_image = get_image_and_inputs_and_transformed_image(
            image_name=image_name, feature_extractor=feature_extractor
        )
        target = vit_model(**inputs)
        target_class_idx = torch.argmax(target.logits[0])

        (
            total_losses,
            prediction_losses,
            correct_class_logits,
            correct_class_probs,
            tokens_mask,
            temps,
        ) = ([], [], [], [], [], [])

        # max_folder, mean_folder, median_folder, min_folder, objects_path, temp_tokens_max_folder, \
        # temp_tokens_mean_folder, temp_tokens_median_folder, temp_tokens_min_folder = start_run_save_files_plot_visualizations_create_folders(
        #     model=vit_ours_model, image_plot_folder_path=image_plot_folder_path, inputs=inputs, run=None,
        #     original_image=original_transformed_image)

        # _ = vit_ours_model(**inputs)
        # attention_probs = get_attention_probs(model=vit_ours_model)
        # mask_rollout_max = rollout(attentions=attention_probs, head_fusion='max', discard_ratio=0)
        # d_masks = get_rollout_grad(vit_ours_model=vit_ours_model,
        #                            feature_extractor=feature_extractor,
        #                            inputs=inputs,
        #                            discard_ratio=0, return_resized=False)
        for iteration_idx in tqdm(range(vit_config["num_steps"])):
            optimizer.zero_grad()
            output = vit_ours_model(**inputs)
            # correct_class_logit, correct_class_prob, prediction_loss = get_iteration_target_class_stats(
            #     output=output, target_class_idx=target_class_idx)
            # temps.append(vit_ours_model.vit.encoder.x_atvit_ours_model.vit.encoder.x_attention.gradtention.clone())
            loss = criterion(
                output=output.logits,
                target=target.logits,
                temp=vit_ours_model.vit.encoder.x_attention,
            )
            loss.backward()

            if vit_config["verbose"]:
                compare_results_each_n_steps(
                    iteration_idx=iteration_idx,
                    target=target.logits,
                    output=output.logits,
                    prev_x_attention=vit_ours_model.vit.encoder.x_attention,
                    sampled_binary_patches=None,
                )
            temp = vit_ours_model.vit.encoder.x_attention.clone()
            temp = get_temp_to_visualize(temp)
            cls_attentions_probs = get_attention_probs_by_layer_of_the_CLS(model=vit_ours_model)

            if iteration_idx == 0:
                iter_0_folder = create_folder(Path(image_plot_folder_path, "0"))
                cls_attentions_probs_iter_0 = cls_attentions_probs.clone()
                temp_iter_0 = temp.clone()
                for head_idx in range(12):
                    visu(
                        original_image=original_transformed_image,
                        transformer_attribution=cls_attentions_probs[head_idx],
                        file_name=Path(iter_0_folder, f"cls_head_{head_idx}_iter_{iteration_idx}"),
                    )
                save_obj_to_disk(
                    Path(iter_0_folder, "tokens_mask.pkl"), cls_attentions_probs_iter_0
                )

            elif iteration_idx == 160:
                iter_advanced_folder = create_folder(Path(image_plot_folder_path, "advanced"))
                cls_attentions_probs_iter_advanced = cls_attentions_probs.clone()
                temp_iter_advanced = temp.clone()
                for head_idx in range(12):
                    visu(
                        original_image=original_transformed_image,
                        transformer_attribution=cls_attentions_probs[head_idx],
                        file_name=Path(
                            iter_advanced_folder, f"cls_head_{head_idx}_iter_{iteration_idx}"
                        ),
                    )
                save_obj_to_disk(
                    Path(iter_advanced_folder, "tokens_mask.pkl"),
                    cls_attentions_probs_iter_advanced,
                )
            elif iteration_idx == 199:
                print(1)
            if (
                vit_config["plot_visualizations"]
                and iteration_idx >= 160
                and iteration_idx % 10 == 0
            ):
                for head_idx in range(12):
                    entropy_iter_0 = round(entropy(cls_attentions_probs_iter_0[head_idx]).item(), 3)
                    entropy_iter_advanced = round(
                        entropy(cls_attentions_probs_iter_advanced[head_idx]).item(), 3
                    )
                    diff = round(entropy_iter_0 - entropy_iter_advanced, 3)
                    print(
                        f"Iter: {iteration_idx}; CLS entropy_head_idx: {head_idx}: {entropy_iter_0}, {entropy_iter_advanced}, Increased? {entropy_iter_advanced > entropy_iter_0}, diff: {diff}"
                    )

                for head_idx in range(12):
                    entropy_iter_0 = round(
                        entropy(F.softmax(temp_iter_0[head_idx], dim=-1)).item(), 3
                    )
                    entropy_iter_advanced = round(
                        entropy(F.softmax(temp_iter_advanced[head_idx], dim=-1)).item(), 3
                    )
                    diff = round(entropy_iter_0 - entropy_iter_advanced, 3)
                    print(
                        f"Iter: {iteration_idx}; temp head_idx: {head_idx}: {entropy_iter_0}, {entropy_iter_advanced}, Increased? {entropy_iter_advanced > entropy_iter_0}, diff: {diff}"
                    )

                # visu(original_image=original_transformed_image,
                #      transformer_attribution=cls_attentions_probs.median(dim=0)[0],
                #      file_name=Path(image_plot_folder_path, f'median_cls_{iteration_idx}'))

                # for head_idx in range(12):
                #     visu(original_image=original_transformed_image,
                #          transformer_attribution=temp[head_idx],
                #          file_name=Path(image_plot_folder_path, f'temp_head_{head_idx}'))

            optimizer.step()


if __name__ == "__main__":
    experiment_name = "research_heads"
    # experiment_name = f"rollout_temp_mul_{vit_config['objective']}_lr{str(vit_config['lr']).replace('.', '_')}+l1_{loss_config['l1_loss_multiplier']}+kl_loss_{loss_config['kl_loss_multiplier']}+entropy_loss_{loss_config['entropy_loss_multiplier']}+pred_loss_{loss_config['pred_loss_multiplier']}"
    print(experiment_name)
    _ = create_folder(Path(PLOTS_PATH, experiment_name))
    optimize_params(vit_model=vit_model, criterion=objective_temp_softmax)
