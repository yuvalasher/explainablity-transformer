from pathlib import Path

import torch
from PIL import Image
from icecream import ic

from torch import Tensor
from matplotlib import pyplot as plt
from config import config
from main.seg_classification.image_classification_with_token_classification_model_opt import \
    OptImageClassificationWithTokenClassificationModel
from utils.consts import IMAGENET_TEST_IMAGES_FOLDER_PATH, IMAGENET_VAL_IMAGES_FOLDER_PATH
from vit_loader.load_vit import load_vit_pretrained
from vit_utils import get_warmup_steps_and_total_training_steps, freeze_multitask_model, \
    load_feature_extractor_and_vit_model


def show_mask(mask):  # [1, 1, 224, 224]
    mask = mask if len(mask.shape) == 3 else mask.squeeze(0)
    _ = plt.imshow(mask.squeeze(0).cpu().detach())
    plt.show()


def patch_score_to_image(transformer_attribution: Tensor, output_2d_tensor: bool = True) -> Tensor:
    """
    Convert Patch scores ([196]) to image size tesnor [224, 224]
    :param transformer_attribution: Tensor with score of each patch in the picture
    :return:
    """
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    if output_2d_tensor:
        transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
            transformer_attribution.max() - transformer_attribution.min())
    return transformer_attribution


def mask_patches_to_image_scores(patches_mask):
    images_mask = []
    for mask in patches_mask:
        images_mask.append(patch_score_to_image(transformer_attribution=mask, output_2d_tensor=False))
    images_mask = torch.stack(images_mask).squeeze(1)
    return images_mask


CKPT_PATH = "/home/yuvalas/explainability/research/checkpoints/token_classification/seg_cls; pred_l_1_mask_l_l1_80_sigmoid_False_freezed_seg_transformer_False_train_n_samples_6000_lr_0.002_mlp_classifier_True/None/checkpoints/epoch=3-step=751.ckpt"
DIRECT_PATH = IMAGENET_VAL_IMAGES_FOLDER_PATH
listdir = sorted(list(Path(DIRECT_PATH).iterdir()))
snake = Image.open(listdir[0])

vit_config = config["vit"]
vit_for_classification_image, vit_for_patch_classification = load_vit_pretrained(model_name=vit_config["model_name"])

warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(
    n_epochs=vit_config["n_epochs"],
    train_samples_length=len(list(Path(IMAGENET_TEST_IMAGES_FOLDER_PATH).iterdir())),
    batch_size=vit_config["batch_size"],
)

feature_extractor, _ = load_feature_extractor_and_vit_model(
    vit_config=vit_config,
    model_type="vit-basic",
    is_wolf_transforms=vit_config["is_wolf_transforms"],
)  # TODO if vit-for-dino is relevant

CHECKPOINT_EPOCH_IDX = 4  # TODO - pay attention !!!
model = OptImageClassificationWithTokenClassificationModel(
    vit_for_classification_image=vit_for_classification_image,
    vit_for_patch_classification=vit_for_patch_classification,
    feature_extractor=feature_extractor,
    plot_path=Path(""),
    warmup_steps=warmup_steps,
    total_training_steps=total_training_steps,
    batch_size=vit_config["batch_size"],
    best_auc_objects_path=Path(""),
    checkpoint_epoch_idx=CHECKPOINT_EPOCH_IDX,
    best_auc_plot_path=Path(""),
)

model = freeze_multitask_model(
    model=model,
    freezing_classification_transformer=vit_config["freezing_classification_transformer"],
)
if __name__ == '__main__':
    m = model.load_from_checkpoint(CKPT_PATH,
                                   vit_for_classification_image=vit_for_classification_image,
                                   vit_for_patch_classification=vit_for_patch_classification,
                                   feature_extractor=feature_extractor,
                                   plot_path=Path(""),
                                   warmup_steps=warmup_steps,
                                   total_training_steps=total_training_steps,
                                   batch_size=vit_config["batch_size"],
                                   best_auc_objects_path=Path(""),
                                   checkpoint_epoch_idx=CHECKPOINT_EPOCH_IDX,
                                   best_auc_plot_path=Path(""),
                                   )

    snake_inputs = feature_extractor(snake)
    print('*********************** Train ***********************')
    m.train()
    ic(m.training)
    train_mode_snake_token_mask = m(torch.tensor(snake_inputs['pixel_values'][0]).unsqueeze(0)).tokens_mask
    train_model_snake_images_mask = mask_patches_to_image_scores(train_mode_snake_token_mask)
    print(train_model_snake_images_mask)
    show_mask(train_model_snake_images_mask)

    print('*********************** Eval ***********************')
    m.eval()
    ic(m.training)
    eval_mode_snake_token_mask = m(torch.tensor(snake_inputs['pixel_values'][0]).unsqueeze(0)).tokens_mask
    eval_model_snake_images_mask = mask_patches_to_image_scores(eval_mode_snake_token_mask)
    print(eval_model_snake_images_mask)
    show_mask(eval_model_snake_images_mask)
