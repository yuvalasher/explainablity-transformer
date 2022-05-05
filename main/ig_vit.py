import torch
from tqdm import tqdm
# from utils.utils import *
from evaluation.evaluation_utils import load_obj_from_path
from loss_utils import *
from utils.consts import *
from pytorch_lightning import seed_everything
from typing import Callable, Tuple
from objectives import objective_grad_rollout
from torch.autograd import grad
from torchvision import transforms

vit_config = config['vit']
loss_config = vit_config['loss']

seed_everything(config['general']['seed'])
feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config, model_type='vit-for-dino',
                                                                    is_wolf_transforms=vit_config['is_wolf_transforms'])
MEAN_AGG = False
mobile_net = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
mobile_net.eval()


def preds_and_grads(model, resized_image, baseline, target_idx) -> Tuple[torch.tensor, torch.tensor]:
    resized_image = resized_image.reshape(1, 3, 224, 224)
    n_steps = 10
    alphas = torch.linspace(start=0.0, end=1.0, steps=n_steps)
    inputs = resized_image
    alphas_x = alphas.reshape(n_steps, 1, 1, 1)
    interpolated_images = (baseline + alphas_x * (inputs - baseline)).requires_grad_()
    plot_interpolated_images(image=resized_image, interpolated_images=interpolated_images)
    preds = model(interpolated_images)
    logits = preds[:, target_idx]
    probs = F.softmax(preds, dim=-1)[:, target_idx]
    # preds = model(interpolated_images).logits[:, target_idx]
    grads = grad(outputs=torch.unbind(probs), inputs=interpolated_images)
    return preds, grads


def plot_image(image):
    to_pil_image = transforms.ToPILImage()
    to_pil_image(image.reshape(3, 224, 224)).show()


def plot_interpolated_images(image, interpolated_images: Tensor, n_steps: int = 10) -> None:
    fig = plt.figure(figsize=(20, 20))
    alphas = torch.linspace(start=0.0, end=1.0, steps=n_steps)
    i = 0
    for alpha, image in zip(alphas, interpolated_images):
        i += 1
        plt.subplot(1, len(alphas), i)
        plt.title(f'alpha: {alpha:.1f}')
        plt.imshow(image.permute(1, 2, 0).detach().numpy())
        plt.axis('off')
        plt.tight_layout();


def scores_to_plot(array):
    return abs(array).sum(dim=-1)


def avg_gradients_to_scores(acc_grads, image, baseline_tensor):
    scores = scores_to_plot(
        (((image - baseline_tensor) * acc_grads.permute(0, 1, 2).reshape(1, 3, 224, 224)).reshape(3, 224, 224)).permute(
            1, 2, 0))
    return scores


def plot_heatmap(scores, image):
    cmap = None
    fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8, 8))
    axs[0, 0].set_title('Attribution mask')
    axs[0, 0].imshow(scores, cmap=cmap)
    axs[0, 0].axis('off')
    axs[0, 1].set_title('Overlay IG on Input image ')
    axs[0, 1].imshow(scores, cmap=cmap)
    axs[0, 1].imshow(image.permute(1, 2, 0), alpha=0.4, interpolation='nearest')
    axs[0, 1].axis('off')
    plt.tight_layout()
    plt.show()


def optimize_params(vit_model: ViTForImageClassification, criterion: Callable):
    # vit_unfreezed = handle_model_config_and_freezing_for_task(
    #     model=load_ViTModel(vit_config, model_type='vit-for-dino-grad'),
    #     freezing_transformer=False)

    for idx, image_dict in enumerate(vit_config['images']):
        image_name, correct_class_idx, contrastive_class_idx = get_image_spec(image_dict)
        # optimizer = optim.Adam(vit_unfreezed.parameters(), lr=vit_config['lr'])
        # image_plot_folder_path = get_and_create_image_plot_folder_path(images_folder_path=IMAGES_FOLDER_PATH,
        #                                                                experiment_name=experiment_name,
        #                                                                image_name=image_name, save_image=False)

        image_resized = transforms.PILToTensor()(
            get_image_from_path(Path(IMAGES_FOLDER_PATH, image_name)).resize((224, 224)))
        torch_baseline = torch.zeros((1, 3, 224, 224))
        preds, grads = preds_and_grads(model=mobile_net, resized_image=image_resized, baseline=torch_baseline,
                                       target_idx=correct_class_idx)

        avg_grads = grads[0].mean(dim=0) if MEAN_AGG else torch.trapz(grads[0], dim=0)
        ig = (image_resized - torch_baseline.reshape(3, 224, 224)) * avg_grads

        ig_scores = avg_gradients_to_scores(acc_grads=ig, image=image_resized, baseline_tensor=torch_baseline)
        plot_heatmap(scores=ig_scores, image=image_resized)

        avg_grads_scores = avg_gradients_to_scores(acc_grads=avg_grads, image=image_resized,
                                                   baseline_tensor=torch_baseline)
        plot_heatmap(scores=avg_grads_scores, image=image_resized)
        # target = vit_model(**inputs)
        # optimizer.zero_grad()
        # output = vit_unfreezed(**inputs)
        # target_idx = target.logits.argmax().item()
        # loss = criterion(output=output.logits, target_idx=target_idx)
        # loss.backward()
        # optimizer.step()


if __name__ == '__main__':
    experiment_name = f"pasten"
    print(experiment_name)
    _ = create_folder(Path(PLOTS_PATH, experiment_name))
    optimize_params(vit_model=vit_model, criterion=objective_grad_rollout)
    # print_number_of_trainable_and_not_trainable_params(model=vit_unfreezed)
