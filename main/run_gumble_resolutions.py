from tqdm import tqdm
from torch import optim
from utils import *
from loss_utils import *
from log_utils import configure_log
from utils.consts import *
from pytorch_lightning import seed_everything
from typing import Callable
from utils.transformation import pil_to_resized_tensor_transform
from torch.nn.functional import relu

vit_config = config['vit']
loss_config = vit_config['loss']

seed_everything(config['general']['seed'])
feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config)


def objective_gumble_softmax(output: Tensor, target: Tensor, x_attention: Tensor,
                             sampled_binary_patches: Tensor = None) -> Tensor:
    prediction_loss = ce_loss(output, torch.argmax(target).unsqueeze(0)) * loss_config['pred_loss_multiplier']
    kl = kl_div(p=convert_probability_vector_to_bernoulli_kl(F.sigmoid(x_attention)),
                q=convert_probability_vector_to_bernoulli_kl(torch.zeros_like(x_attention))) * loss_config[
             'kl_loss_multiplier']
    print(f'kl_loss: {kl}, prediction_loss: {prediction_loss}')
    loss = kl + prediction_loss
    log(loss=loss, kl_loss=kl, prediction_loss=prediction_loss, x_attention=x_attention,
        sampled_binary_patches=sampled_binary_patches, output=output, target=target)
    return loss


def optimize_params(vit_model: ViTForImageClassification, criterion: Callable, log_run):
    for idx, image_name in enumerate(os.listdir(IMAGES_FOLDER_PATH)):
        if image_name in vit_config['sample_images']:

            vit_sigmoid_model = handle_model_config_and_freezing_for_task(
                model=load_ViTModel(vit_config, model_type='gumble_resolutions'),
                freezing_transformer=vit_config['freezing_transformer'])
            optimizer = optim.Adam([vit_sigmoid_model.vit.encoder.x_attention], lr=vit_config['lr'])
            image_plot_folder_path = get_and_create_image_plot_folder_path(images_folder_path=IMAGES_FOLDER_PATH,
                                                                           experiment_name=experiment_name,
                                                                           image_name=image_name)
            image = get_image_from_path(Path(IMAGES_FOLDER_PATH, image_name))
            inputs = feature_extractor(images=image, return_tensors="pt")
            original_transformed_image = pil_to_resized_tensor_transform(image)
            target = vit_model(**inputs)
            prediction_losses = []
            for iteration_idx in tqdm(range(vit_config['num_steps'])):
                optimizer.zero_grad()
                output = vit_sigmoid_model(**inputs)
                prediction_losses.append(ce_loss(output.logits, torch.argmax(target.logits).unsqueeze(0)) * loss_config[
                    'pred_loss_multiplier'])
                loss = criterion(output=output.logits, target=target.logits,
                                 x_attention=vit_sigmoid_model.vit.encoder.x_attention,
                                 sampled_binary_patches=vit_sigmoid_model.vit.encoder.sampled_binary_patches)
                loss.backward()
                compare_results_each_n_steps(iteration_idx=iteration_idx, target=target.logits, output=output.logits,
                                             prev_x_attention=vit_sigmoid_model.vit.encoder.x_attention,
                                             sampled_binary_patches=vit_sigmoid_model.vit.encoder.sampled_binary_patches.clone()
                                             )

                if vit_config['verbose'] and is_iteration_to_action(iteration_idx):
                    printed_vector = vit_sigmoid_model.vit.encoder.sampled_binary_patches
                    save_saliency_map(image=original_transformed_image,
                                      saliency_map=torch.tensor(
                                          get_scores(printed_vector)).unsqueeze(0),
                                      filename=Path(image_plot_folder_path, f'sbp_iter_idx_{iteration_idx}'),
                                      verbose=is_iteration_to_action(iteration_idx=iteration_idx, action='print'))
                optimizer.step()

            print(f'Minimum prediction_loss at iteration: {get_top_k_mimimum_values_indices(array=prediction_losses)}')


experiment_name = f"gumble_resolutions_lr{str(vit_config['lr']).replace('.', '_')}_temp_{vit_config['temperature']}+l1_{loss_config['l1_loss_multiplier']}+kl_loss_{loss_config['kl_loss_multiplier']}+entropy_loss_{loss_config['entropy_loss_multiplier']}+pred_loss_{loss_config['pred_loss_multiplier']}"

if __name__ == '__main__':
    log_run = configure_log(vit_config=vit_config, experiment_name=experiment_name)
    os.makedirs(name=Path(PLOTS_PATH, experiment_name), exist_ok=True)
    optimize_params(vit_model=vit_model, criterion=objective_gumble_softmax, log_run=log_run)
