from tqdm import tqdm
# from utils.utils import *
from loss_utils import *
from utils.consts import *
from pytorch_lightning import seed_everything
from typing import Callable

vit_config = config['vit']
loss_config = vit_config['loss']

seed_everything(config['general']['seed'])
feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config, model_type='vit-for-dino')


def objective_pure_vit(output: Tensor, target: Tensor, target_idx: Tensor) -> Tensor:
    # print(f'target_idx: {target_idx}; dog logits: {output[0][243]}, cat logits: {output[0][282]}')
    # prediction_loss = -output[0][target_idx] * loss_config['pred_loss_multiplier']
    # prediction_loss = ce_loss(output, torch.tensor(target_idx).unsqueeze(0))
    prediction_loss = output[0][target_idx]
    loss = prediction_loss
    return loss


def get_attention(module, input, output):
    attentions.append(output.cpu())


def get_attention_gradient(module, grad_input, grad_output):
    attention_gradients.append(grad_input[0].cpu())


attentions = []
attention_gradients = []


def get_parameters_hooks(pytorch_model):
    params_gradients = {}
    params_handles = {}
    for name, param in pytorch_model.named_parameters():
        # print("name=%s, shape=%s" % (name, param.shape))
        param.retain_grad()
        params_handles[name] = param.register_hook(get_params_gradients(params_gradients, name))
    return params_gradients, params_handles


def get_params_gradients(params_gradients, name):
    def hook(output):
        params_gradients[name] = output

    return hook


def get_forward_hooks(pytorch_model):
    activations = {}
    forward_handles = {}
    for name, module in pytorch_model.named_modules():
        # print("name=%s" % (name))
        # module.retain_grad()
        forward_handles[name] = module.register_forward_hook(get_activations(activations, name))
    return activations, forward_handles


def get_backward_hooks(pytorch_model):
    gradients = {}
    backward_handles = {}
    for name, module in pytorch_model.named_modules():
        # print("name=%s, shape=%s" % (name, param.shape))
        # module.retain_grad()
        backward_handles[name] = module.register_backward_hook(get_gradients(gradients, name))
    return gradients, backward_handles


def get_gradients(gradients, name):
    def hook(model, input, output):
        gradients[name] = output[0]

    return hook


def get_activations(activations, name):
    def hook(model, input, output):
        activations[name] = output

    return hook


def optimize_params(vit_model: ViTForImageClassification, criterion: Callable):
    vit_unfreezed = handle_model_config_and_freezing_for_task(
        model=load_ViTModel(vit_config, model_type='vit-for-dino-grad'),
        freezing_transformer=False)

    for idx, image_dict in enumerate(vit_config['images']):
        image_name, correct_class_idx, contrastive_class_idx = get_image_spec(image_dict)
        optimizer = optim.Adam(vit_unfreezed.parameters(), lr=vit_config['lr'])
        image_plot_folder_path = get_and_create_image_plot_folder_path(images_folder_path=IMAGES_FOLDER_PATH,
                                                                       experiment_name=experiment_name,
                                                                       image_name=image_name)

        inputs, original_transformed_image = get_image_and_inputs_and_transformed_image(image_name=image_name,
                                                                                        feature_extractor=feature_extractor)
        target = vit_model(**inputs)
        optimizer.zero_grad()
        params_gradients, params_handles = get_parameters_hooks(vit_unfreezed)
        forward_activations, forward_handles = get_forward_hooks(vit_unfreezed)
        backward_gradients, backward_handles = get_backward_hooks(vit_unfreezed)

        # print_number_of_trainable_and_not_trainable_params(model=vit_unfreezed)
        output = vit_unfreezed(**inputs)
        cat_idx = 282
        dog_idx = 243
        target_idx = cat_idx
        loss = criterion(output=output.logits, target=target.logits, target_idx=target_idx)
        loss.backward()
        registered_gradients = [mod.attention.attention.attention_probs.grad for mod in
                                vit_unfreezed.vit.encoder.layer._modules.values()]
        rollout_folder = create_folder(Path(image_plot_folder_path, 'rollout'))
        attention_probs = get_attention_probs(model=vit_model)
        gradients = get_attention_grads_probs(model=vit_unfreezed, apply_relu=False)
        # for head_idx in range(12):
        #     visu(original_image=original_transformed_image,
        #          transformer_attribution=gradients[-1][:, head_idx, 0, 1:][0],
        #          file_name=Path(rollout_folder, f'{vit_unfreezed.config.id2label[target_idx]}_grad_head_{head_idx}'))
        relu_gradients = get_attention_grads_probs(model=vit_unfreezed, apply_relu=True)
        # h_rollout_max = hila_rollout(attnetions=attention_probs, head_fusion='max')
        # h_rollout_mean = hila_rollout(attnetions=attention_probs, head_fusion='mean')
        # h_rollout_max_relu_grad = hila_rollout(attnetions=attention_probs, head_fusion='max', gradients=relu_gradients)
        # h_rollout_mean_relu_grad = hila_rollout(attnetions=attention_probs, head_fusion='mean', gradients=relu_gradients)
        h_rollout_mean_grad = hila_rollout(attnetions=attention_probs, head_fusion='mean', gradients=gradients)
        h_rollout_max_grad = hila_rollout(attnetions=attention_probs, head_fusion='max', gradients=gradients)

        visu(original_image=original_transformed_image,
             transformer_attribution=gradients[-1][0, :, 0, 1:].median(dim=0)[0],
             file_name=Path(rollout_folder, f'{vit_unfreezed.config.id2label[target_idx]}_median_last_layer_grad'))
        visu(original_image=original_transformed_image,
             transformer_attribution=gradients[-1][0, :, 0, 1:].mean(dim=0),
             file_name=Path(rollout_folder, f'{vit_unfreezed.config.id2label[target_idx]}_mean_last_layer_grad'))
        visu(original_image=original_transformed_image,
             transformer_attribution=registered_gradients[-1][0, :, 0, 1:].median(dim=0)[0],
             file_name=Path(rollout_folder,
                            f'{vit_unfreezed.config.id2label[target_idx]}_median_last_layer_grad_registered'))
        visu(original_image=original_transformed_image,
             transformer_attribution=registered_gradients[-1][0, :, 0, 1:].mean(dim=0),
             file_name=Path(rollout_folder,
                            f'{vit_unfreezed.config.id2label[target_idx]}_mean_last_layer_grad_registered'))
        visu(original_image=original_transformed_image,
             transformer_attribution=relu_gradients[-1][:, -1, 0, 1:][0],
             file_name=Path(rollout_folder, f'{vit_unfreezed.config.id2label[target_idx]}_grad_relu'))
        # visu(original_image=original_transformed_image,
        #      transformer_attribution=h_rollout_max,
        #      file_name=Path(rollout_folder, f'{vit_unfreezed.config.id2label[target_idx]}_hila_rollout_max'))
        # visu(original_image=original_transformed_image,
        #      transformer_attribution=h_rollout_mean,
        #      file_name=Path(rollout_folder, f'{vit_unfreezed.config.id2label[target_idx]}_hila_rollout_mean'))
        #
        visu(original_image=original_transformed_image,
             transformer_attribution=h_rollout_max_grad,
             file_name=Path(rollout_folder, f'{vit_unfreezed.config.id2label[target_idx]}_hila_rollout_max_grad'))
        visu(original_image=original_transformed_image,
             transformer_attribution=h_rollout_mean_grad,
             file_name=Path(rollout_folder, f'{vit_unfreezed.config.id2label[target_idx]}_hila_rollout_mean_grad'))
        # visu(original_image=original_transformed_image,
        #      transformer_attribution=h_rollout_max_relu_grad,
        #      file_name=Path(rollout_folder, f'{vit_unfreezed.config.id2label[target_idx]}_hila_rollout_max_relu_grad'))
        #
        # visu(original_image=original_transformed_image,
        #      transformer_attribution=h_rollout_mean_relu_grad,
        #      file_name=Path(rollout_folder, f'{vit_unfreezed.config.id2label[target_idx]}_hila_rollout_mean_relu_grad'))

        # optimizer.step()


if __name__ == '__main__':
    experiment_name = f"pure_vit_grad"
    print(experiment_name)
    _ = create_folder(Path(PLOTS_PATH, experiment_name))
    optimize_params(vit_model=vit_model, criterion=objective_pure_vit)
