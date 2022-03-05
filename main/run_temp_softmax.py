from transformers import ViTConfig
from tqdm import tqdm
from torch import optim
from utils import *
from loss_utils import *
import wandb
from log_utils import get_wandb_config
from utils.consts import *
from pytorch_lightning import seed_everything
from utils.transformation import pil_to_resized_tensor_transform
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
        with wandb.init(project="vit-temp-softmax", entity="yuvalasher", config=wandb_config) as run:
            vit_sigmoid_model = handle_model_config_and_freezing_for_task(
                model=load_ViTModel(vit_config, model_type='softmax_temp'),
                freezing_transformer=vit_config['freezing_transformer'])
            print_number_of_trainable_and_not_trainable_params(model=vit_sigmoid_model, model_name='soft_temp')
            optimizer = optim.Adam([vit_sigmoid_model.vit.encoder.x_attention], lr=vit_config['lr'])

            image_plot_folder_path = get_and_create_image_plot_folder_path(images_folder_path=IMAGES_FOLDER_PATH,
                                                                           experiment_name=experiment_name,
                                                                           image_name=image_name)
            save_url_to_text_file(path=image_plot_folder_path, log_run=run) if run is not None else []
            image = get_image_from_path(Path(IMAGES_FOLDER_PATH, image_name))
            inputs = feature_extractor(images=image, return_tensors="pt")
            original_transformed_image = pil_to_resized_tensor_transform(image)
            target = vit_model(**inputs)
            vit_basic_for_dino = handle_model_config_and_freezing_for_task(
                model=load_ViTModel(vit_config, model_type='vit-for-dino'))
            _ = vit_basic_for_dino(**inputs)  # run forward to save attention_probs
            dino_method_attention_probs_cls_on_tokens_last_layer(vit_sigmoid_model=vit_basic_for_dino,
                                                                 path=image_plot_folder_path)
            attention_probs = get_attention_probs(model=vit_basic_for_dino)
            plot_attention_rollout(attention_probs=attention_probs, path=image_plot_folder_path,
                                   patch_size=vit_config['patch_size'], iteration_idx=0, head_fusion='max')
            plot_attention_rollout(attention_probs=attention_probs, path=image_plot_folder_path,
                                   patch_size=vit_config['patch_size'], iteration_idx=0, head_fusion='mean')
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

                # plot_attn_probs(attentions=cls_attentions_probs, image_size=vit_config['img_size'],
                #                 patch_size=vit_config['patch_size'], path=image_plot_folder_path,
                #                 iteration_idx=iteration_idx, num_heads=vit_config['n_heads'])
                save_saliency_map(image=original_transformed_image,
                                  saliency_map=torch.tensor(
                                      get_scores(cls_attentions_probs.mean(dim=0))).unsqueeze(0),
                                  filename=Path(image_plot_folder_path, f'plot_{iteration_idx}'),
                                  verbose=False)
                # verbose=is_iteration_to_action(iteration_idx=iteration_idx, action='print'))
                optimizer.step()
                if is_iteration_to_action(iteration_idx=iteration_idx, action='save'):
                    objects_dict = {'losses': prediction_losses, 'total_losses': total_losses,
                                    'tokens_mask': tokens_mask}
                    save_objects(path=image_plot_folder_path, objects_dict=objects_dict)

            minimum_predictions = get_minimum_predictions_string(image_name=image_name, total_losses=total_losses,
                                                                 prediction_losses=prediction_losses)
            print(minimum_predictions)
            save_text_to_file(path=image_plot_folder_path, file_name='minimum_predictions', text=minimum_predictions)
            # save_obj_to_disk(path=Path(image_plot_folder_path, 'temp'),
            #                  obj=x_attention[get_top_k_mimimum_values_indices(array=prediction_losses, k=1)])


experiment_name = f"head_layer_{vit_config['objective']}_lr{str(vit_config['lr']).replace('.', '_')}_temp_{vit_config['temperature']}+l1_{loss_config['l1_loss_multiplier']}+kl_loss_{loss_config['kl_loss_multiplier']}+entropy_loss_{loss_config['entropy_loss_multiplier']}+pred_loss_{loss_config['pred_loss_multiplier']}"

if __name__ == '__main__':
    os.makedirs(name=Path(PLOTS_PATH, experiment_name), exist_ok=True)
    optimize_params(vit_model=vit_model, criterion=objective_temp_softmax)
