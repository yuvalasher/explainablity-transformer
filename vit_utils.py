from torch import Tensor
from transformers import ViTFeatureExtractor, ViTForImageClassification
from modeling_vit_sigmoid import ViTSigmoidForImageClassification
from PIL import Image
from typing import Union, NewType

VitForClassification = NewType('VitForClassification',
                               Union[ViTSigmoidForImageClassification, ViTForImageClassification])


def freeze_all_model_params(model: VitForClassification) -> VitForClassification:
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_x_attention_params(model: VitForClassification) -> VitForClassification:
    for param in model.named_parameters():
        if param[0] == 'vit.encoder.x_attention':
            param[1].requires_grad = True
    return model


def handle_model_freezing(model: VitForClassification) -> VitForClassification:
    model = freeze_all_model_params(model=model)
    model = unfreeze_x_attention_params(model=model)
    return model


def setup_model_config(model: VitForClassification) -> VitForClassification:
    model.config.output_scores = True
    model.config.output_attentions = True
    return model


def get_logits_for_image(model: VitForClassification, feature_extractor: ViTFeatureExtractor, image: Image) -> Tensor:
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)  # inputs['pixel_values].shape: [batch_Size, n_channels, height, width]
    logits = outputs.logits
    return logits


def get_pred_idx_from_logits(logits: Tensor) -> int:
    predicted_class_idx = logits.argmax(-1).item()
    return predicted_class_idx


def calculate_num_of_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_num_of_params(model) -> int:
    return sum(p.numel() for p in model.parameters())

def calculate_percentage_of_trainable_params(model) -> str:
    return f'{round(calculate_num_of_trainable_params(model) / calculate_num_of_params(model), 2) * 100}%'