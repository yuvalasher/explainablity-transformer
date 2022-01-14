from icecream import ic
import torch
from torch import nn
from config import config
from utils import *
from vit_utils import *
from consts import *

vit_config = config['vit']

bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='mean')
bce_loss = nn.BCELoss(reduction='mean')
kl_loss = nn.KLDivLoss(reduction='mean')
sigmoid = nn.Sigmoid()

feature_extractor = ViTFeatureExtractor.from_pretrained(vit_config['model_name'])
vit_model = handle_model_freezing(
    setup_model_config(ViTForImageClassification.from_pretrained(vit_config['model_name'])))
vit_sigmoid_model = handle_model_freezing(
    setup_model_config(ViTSigmoidForImageClassification.from_pretrained(vit_config['model_name'])))

labels = parse_gt_labels(read_gt_labels(path=images_labels_gt_path))


def objective_loss_1(output: Tensor, target: Tensor, lambda_1: float = 1, lambda_2: float = 1) -> Tensor:
    """
    Loss between the original prediction's distribution of the model and the prediction's distribution of the new model
    + average of the BCE of the x * self-attention
    """
    target_off_patches = torch.zeros_like(vit_sigmoid_model.vit.encoder.x_attention)
    loss = lambda_1 * bce_with_logits_loss(output, target) + \
           lambda_2 * bce_with_logits_loss(vit_sigmoid_model.vit.encoder.x_attention, target_off_patches)
    return loss


if __name__ == '__main__':
    print(
        f'Number of params: {calculate_num_of_params(vit_sigmoid_model)}, Number of trainable params: {calculate_num_of_trainable_params(vit_sigmoid_model)}')
    for (idx, image_name), label in zip(enumerate(os.listdir(images_folder_path)), labels):
        if idx < 10:
            image = get_image_from_path(os.path.join(images_folder_path, image_name))
            vit_logits = get_logits_for_image(model=vit_model, feature_extractor=feature_extractor, image=image)
            s_logits = get_logits_for_image(model=vit_sigmoid_model, feature_extractor=feature_extractor,
                                            image=image)
            print(
                f'Original vit predicted idx: {torch.argmax(vit_logits[0]).item()}, Sigmoid-vit predicted idx: {torch.argmax(s_logits[0]).item()}')
            print(
                f'Diff in biggest logits class: {(abs(max(vit_logits[0]).item() - s_logits[0][torch.argmax(vit_logits[0]).item()].item()))}')
            # print(
            #     f"{os.path.join(images_folder_path, image_name)}, predicted_idx {y_pred}, Predicted class: {vit_model.config.id2label[y_pred]}, gt: {label}")
