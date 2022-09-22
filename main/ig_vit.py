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
import torchvision
from torchvision.io import read_image
import tensorflow as tf

vit_config = config["vit"]
loss_config = vit_config["loss"]

seed_everything(config["general"]["seed"])
feature_extractor, vit_model = load_feature_extractor_and_vit_model(
    vit_config=vit_config,
    model_type="vit-for-dino",
    is_wolf_transforms=vit_config["is_wolf_transforms"],
)
MEAN_AGG = False
mobile_net = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True)
mobile_net.eval()

MOBILE_NET_TYPE = torchvision.models.mobilenetv2.MobileNetV2


def tf_to_torch(array) -> torch.tensor:
    return torch.tensor(array.numpy())


def read_image_tf(file_name):
    image = tf.io.read_file(file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, target_height=224, target_width=224)
    return tf_to_torch(image)


def resize(image, size: int = 224):
    """
    image size should be: [3, H, W]
    """
    up_fill = int((size - image.shape[1]) / 2)
    down_fill = size - image.shape[1] - up_fill
    right_fill = int((size - image.shape[2]) / 2)
    left_fill = size - image.shape[2] - right_fill
    resized_image = F.pad(image, pad=(left_fill, right_fill, up_fill, down_fill))
    return resized_image


def read_image_from_disk(image_name: Union[WindowsPath, str], size: int):
    image = read_image(str(image_name))
    image = torchvision.transforms.functional.convert_image_dtype(
        image.permute(1, 2, 0), dtype=torch.float32
    )
    image = resize(image=image.permute(2, 0, 1), size=size)
    return image.permute(1, 2, 0).float()


def get_interpolated_images(image, baseline, n_steps):
    # print(image.shape)
    inputs_x = image
    alphas = torch.linspace(start=0.0, end=1.0, steps=n_steps + 1)
    baseline_x = baseline.permute(0, 2, 3, 1)
    alphas_x = alphas.reshape(n_steps + 1, 1, 1, 1)
    # print(f'baseline: {baseline_x.shape}, alphas_x: {alphas_x.shape}, inputs: {inputs_x.shape}')
    interpolated_images = (baseline_x + alphas_x * (inputs_x - baseline_x)).requires_grad_()
    return interpolated_images


def preds_and_grads(
    model, resized_image, baseline, target_idx, n_steps: int
) -> Tuple[torch.tensor, torch.tensor]:
    interpolated_images = get_interpolated_images(
        image=resized_image, baseline=baseline, n_steps=n_steps
    )
    preds = model(interpolated_images.permute(0, 3, 1, 2))
    if isinstance(model, ViTBasicForForImageClassification):
        preds = preds.logits
    logits = preds[:, target_idx]
    probs = F.softmax(preds, dim=-1)[:, target_idx]
    grads = grad(outputs=torch.unbind(logits), inputs=interpolated_images)
    return preds, grads


def plot_image(image):
    to_pil_image = transforms.ToPILImage()
    to_pil_image(image.reshape(3, 224, 224)).show()


def plot_interpolated_images(image, interpolated_images: Tensor, n_steps: int) -> None:
    # print(f'image_size: {image.squeeze(0).shape}')
    image = image.squeeze(0)
    fig = plt.figure(figsize=(20, 20))
    alphas = torch.linspace(start=0.0, end=1.0, steps=n_steps)
    i = 0
    for alpha, image in zip(alphas, interpolated_images):
        i += 1
        plt.subplot(1, len(alphas), i)
        plt.title(f"alpha: {alpha:.1f}")
        plt.imshow(image.detach().numpy())
        plt.axis("off")
        plt.tight_layout()


def scores_to_plot(array):
    return abs(array).sum(dim=-1)


def avg_gradients_to_scores(acc_grads, image, baseline_tensor):
    s = scores_to_plot(
        (((image - baseline_tensor) * acc_grads.permute(2, 0, 1).unsqueeze(0)).squeeze(0)).permute(
            1, 2, 0
        )
    )
    return s


def plot_heatmap(scores, image):
    cmap = None
    fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8, 8))
    axs[0, 0].set_title("Attribution mask")
    axs[0, 0].imshow(scores, cmap=cmap)
    axs[0, 0].axis("off")
    axs[0, 1].set_title("Overlay IG on Input image ")
    axs[0, 1].imshow(scores, cmap=cmap)
    axs[0, 1].imshow(image.permute(1, 2, 0), alpha=0.4, interpolation="nearest")
    axs[0, 1].axis("off")
    plt.tight_layout()


def optimize_params(model, vit_model: ViTForImageClassification, criterion: Callable):
    n_steps = 10
    # vit_unfreezed = handle_model_config_and_freezing_for_task(
    #     model=load_ViTModel(vit_config, model_type='vit-for-dino-grad'),
    #     freezing_transformer=False)

    for idx, image_dict in enumerate(vit_config["images"]):
        image_name, correct_class_idx, contrastive_class_idx = get_image_spec(image_dict)
        # optimizer = optim.Adam(vit_unfreezed.parameters(), lr=vit_config['lr'])
        # image_plot_folder_path = get_and_create_image_plot_folder_path(images_folder_path=IMAGES_FOLDER_PATH,
        #                                                                experiment_name=experiment_name,
        #                                                                image_name=image_name, save_image=False)

        # image = read_image_from_disk(image_name=Path(IMAGES_FOLDER_PATH, image_name), size=224)
        image = read_image_tf(str(Path(IMAGES_FOLDER_PATH, image_name)))
        plt.imshow(image)
        image_to_display = image.permute(2, 0, 1)
        torch_baseline = torch.zeros((1, 3, 224, 224))
        n_steps = 10
        preds, grads = preds_and_grads(
            model=model,
            resized_image=image.unsqueeze(0),
            baseline=torch_baseline,
            target_idx=correct_class_idx,
            n_steps=n_steps,
        )

        accumulated_gradients = grads[0].mean(dim=0) if MEAN_AGG else torch.trapz(grads[0], dim=0)
        s = avg_gradients_to_scores(
            acc_grads=accumulated_gradients, image=image_to_display, baseline_tensor=torch_baseline
        )
        plot_heatmap(scores=s, image=image_to_display)
        plt.show()

        # ig = (image - torch_baseline.squeeze(0).permute(1, 2, 0)) * acc_grads
        # s = avg_gradients_to_scores(acc_grads=ig, image=image_to_display, baseline_tensor=torch_baseline)
        # plot_heatmap(scores=s, image=image_to_display)
        # plt.show()

        # image_resized = transforms.PILToTensor()(
        #     get_image_from_path(Path(IMAGES_FOLDER_PATH, image_name)).resize((224, 224)))
        # torch_baseline = torch.zeros((1, 3, 224, 224))
        # preds, grads = preds_and_grads(model=vit_model, resized_image=image_resized.unsqueeze(0),
        #                                baseline=torch_baseline,
        #                                target_idx=correct_class_idx, n_steps=n_steps)
        #
        # avg_grads = grads[0].mean(dim=0) if MEAN_AGG else torch.trapz(grads[0], dim=0)
        # ig = (image_resized - torch_baseline.reshape(3, 224, 224)) * avg_grads
        #
        # ig_scores = avg_gradients_to_scores(acc_grads=ig, image=image_resized, baseline_tensor=torch_baseline)
        # plot_heatmap(scores=ig_scores, image=image_resized)
        #
        # avg_grads_scores = avg_gradients_to_scores(acc_grads=avg_grads, image=image_resized,
        #                                            baseline_tensor=torch_baseline)
        # plot_heatmap(scores=avg_grads_scores, image=image_resized)
        # target = vit_model(**inputs)
        # optimizer.zero_grad()
        # output = vit_unfreezed(**inputs)
        # target_idx = target.logits.argmax().item()
        # loss = criterion(output=output.logits, target_idx=target_idx)
        # loss.backward()
        # optimizer.step()


if __name__ == "__main__":
    experiment_name = f"pasten"
    _ = create_folder(Path(PLOTS_PATH, experiment_name))
    optimize_params(model=vit_model, vit_model=vit_model, criterion=objective_grad_rollout)
