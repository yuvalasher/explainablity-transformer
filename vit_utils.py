from torch import Tensor
from transformers import ViTFeatureExtractor, ViTForImageClassification
from modeling_vit_sigmoid import ViTSigmoidForImageClassification
from PIL import Image
from typing import Dict, Tuple, Union, NewType

VitModelForClassification = NewType('VitModelForClassification',
                                    Union[ViTSigmoidForImageClassification, ViTForImageClassification])
vit_model_types = {'vit': ViTForImageClassification, 'vit-sigmoid': ViTSigmoidForImageClassification}


def freeze_all_model_params(model: VitModelForClassification) -> VitModelForClassification:
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_x_attention_params(model: VitModelForClassification) -> VitModelForClassification:
    for param in model.named_parameters():
        if param[0] == 'vit.encoder.x_attention':
            param[1].requires_grad = True
    return model


def handle_model_freezing(model: VitModelForClassification) -> VitModelForClassification:
    model = freeze_all_model_params(model=model)
    model = unfreeze_x_attention_params(model=model)
    return model


def setup_model_config(model: VitModelForClassification) -> VitModelForClassification:
    model.config.output_scores = True
    model.config.output_attentions = True
    return model


def get_logits_for_image(model: VitModelForClassification, feature_extractor: ViTFeatureExtractor,
                         image: Image) -> Tensor:
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


def load_feature_extractor(vit_config: Dict) -> ViTFeatureExtractor:
    feature_extractor = ViTFeatureExtractor.from_pretrained(vit_config['model_name'])
    return feature_extractor


def load_ViTModel(vit_config: Dict, model_type: str) -> VitModelForClassification:
    model = vit_model_types[model_type].from_pretrained(vit_config['model_name'])
    return model


def handle_model_for_task(model: VitModelForClassification) -> VitModelForClassification:
    model = handle_model_freezing(model=setup_model_config(model=model))
    return model


def load_handled_models_for_task(vit_config: Dict) -> Tuple[
    VitModelForClassification, ViTSigmoidForImageClassification]:
    vit_model = handle_model_for_task(model=load_ViTModel(vit_config, model_type='vit'))
    vit_sigmoid_model = handle_model_for_task(model=load_ViTModel(vit_config, model_type='vit-sigmoid'))
    return vit_model, vit_sigmoid_model
