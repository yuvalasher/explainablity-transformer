import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch import nn
from torch.functional import F
from transformers import BertForSequenceClassification, BertTokenizer

from nlp.BERT_explainability.modules.BERT.Bert_Temp_New import BertTempSoftmaxForSequenceClassification
from nlp.BERT_explainability.modules.BERT.BertForSequenceClassification import \
    BertForSequenceClassification as T_BertForSequenceClassification, Amit_Temp_BertForSequenceClassification

import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple, Union, NewType, List, Optional
from pathlib import Path, WindowsPath
from utils.consts import PLOTS_PATH
from utils import save_obj_to_disk
from config import config
from torch import optim
from utils.consts import IMAGES_FOLDER_PATH
from utils.utils_functions import get_image_from_path

ce_loss = nn.CrossEntropyLoss(reduction='mean')

bert_config = config['bert']
loss_config = bert_config['loss']

BertModelForSequenceClassification = NewType('BertModelForSequenceClassification',
                                             Union[
                                                 BertForSequenceClassification, Amit_Temp_BertForSequenceClassification])

bert_model_types = {'infer': BertForSequenceClassification,
                    'softmax_temp': T_BertForSequenceClassification,
                    'amit_softmax_temp': BertTempSoftmaxForSequenceClassification,
                    }


def get_head_num_heads(model) -> int:
    return model.bert.encoder.layer[-1].attention.self.attention_probs.shape[1]


def get_num_layers(model) -> int:
    return len(model.bert.encoder.layer)


def get_attention_probs(model) -> List[Tensor]:
    """

    :return: Tensor of size (num_heads, num_tokens)
    """
    num_layers = get_num_layers(model=model)
    attentions = [model.bert.encoder.layer[layer_idx].attention.self.attention_probs for layer_idx in
                  range(num_layers)]
    return attentions


def get_attention_probs_by_layer_of_the_CLS(model, layer: int = -1) -> Tensor:
    """

    :return: Tensor of size (num_heads, num_tokens)
    """
    num_heads = get_head_num_heads(model=model)
    attentions = model.bert.encoder.layer[layer].attention.self.attention_probs[0, :, 0, 1:].reshape(
        num_heads, -1)
    return attentions


def freeze_all_model_params(model: BertModelForSequenceClassification) -> BertForSequenceClassification:
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_x_attention_params(model: BertModelForSequenceClassification) -> BertModelForSequenceClassification:
    for param in model.named_parameters():
        if param[0] == 'bert.encoder.x_attention':
            param[1].requires_grad = True
    return model


def handle_model_freezing(model: BertModelForSequenceClassification) -> BertModelForSequenceClassification:
    model = freeze_all_model_params(model=model)
    model = unfreeze_x_attention_params(model=model)
    return model


def setup_model_config(model: BertModelForSequenceClassification) -> BertModelForSequenceClassification:
    model.config.output_scores = True
    model.config.output_attentions = True
    model.config.input_num_tokens = 5
    return model


"""
def get_logits_for_image(model: BertModelForSequenceClassification, feature_extractor: ViTFeatureExtractor,
                         image: Image) -> Tensor:
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)  # inputs['pixel_values].shape: [batch_Size, n_channels, height, width]
    logits = outputs.logits
    return logits
"""


def get_pred_idx_from_logits(logits: Tensor) -> int:
    predicted_class_idx = logits.argmax(-1).item()
    return predicted_class_idx


def calculate_num_of_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_num_of_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def calculate_percentage_of_trainable_params(model) -> str:
    return f'{round(calculate_num_of_trainable_params(model) / calculate_num_of_params(model), 2) * 100}%'


def print_number_of_trainable_and_not_trainable_params(model: BertModelForSequenceClassification) -> None:
    print(
        f'Number of params: {calculate_num_of_params(model)}, Number of trainable params: {calculate_num_of_trainable_params(model)}')


def load_BertModel(bert_config: Dict, model_type: str,device: str) -> BertModelForSequenceClassification:
    model = bert_model_types[model_type].from_pretrained(bert_config['model_name']).to(device)
    return model

def initialize_models(params: dict, model_type: str, batch_first: bool, device: str, use_half_precision=False):
    assert batch_first
    max_length = params['max_length']
    tokenizer = BertTokenizer.from_pretrained(params['model_name'])
    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    word_interner = tokenizer.vocab
    de_interner = tokenizer.ids_to_tokens

    bert_dir = params['model_name']
    evidence_classes = dict((y, x) for (x, y) in enumerate(params['evidence_classifier']['classes']))
    evidence_classifier = bert_model_types[model_type].from_pretrained(bert_dir, num_labels=len(evidence_classes)).to(device)
    optimizer = optim.Adam([evidence_classifier.bert.encoder.x_attention], lr=bert_config['evidence_classifier']['lr'])

    lmbda = lambda epoch: 0.65 ** epoch
    scheduler = None #torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_lambda=lmbda)

    return evidence_classifier,optimizer,scheduler, word_interner, de_interner, evidence_classes, tokenizer

"""
def load_feature_extractor_and_vit_model(bert_config: Dict):
    feature_extractor = load_feature_extractor(bert_config=bert_config)
    # vit_model, vit_sigmoid_model = load_handled_models_for_task(bert_config=bert_config)
    vit_model = load_bert_model_by_type(bert_config=bert_config, model_type='vit')
    return feature_extractor, vit_model
"""


def load_bert_model_by_type(bert_config: Dict, model_type: str):
    bert_model = handle_model_config_and_freezing_for_task(model=load_BertModel(bert_config, model_type=model_type))
    return bert_model


def handle_model_config_and_freezing_for_task(model: BertModelForSequenceClassification,
                                              freezing_transformer: bool = True) -> BertModelForSequenceClassification:
    model = setup_model_config(model=model)
    if freezing_transformer:
        model = handle_model_freezing(model=model)
    return model


def get_top_k_mimimum_values_indices(array: List[float], k: int = 5, is_largest: bool = False):
    return torch.topk(torch.tensor(array), k=min(len(array), k), largest=is_largest)[1]


def get_patches_by_discard_ratio(array: Tensor, discard_ratio: float, top: bool = True) -> Tensor:
    k = int(array.shape[-1] * discard_ratio)
    _, indices = torch.topk(array, k, largest=bool(1 - top))
    array[indices] = 0
    return array


def get_minimum_predictions_string(image_name: str, total_losses, prediction_losses, logits, correct_class_probs,
                                   k: int = 25) -> str:
    return f'Minimum prediction_loss at iteration: {get_top_k_mimimum_values_indices(array=prediction_losses, k=k)}\nMinimum total loss at iteration: {get_top_k_mimimum_values_indices(array=total_losses, k=k)}\nMaximum logits at iteration: {get_top_k_mimimum_values_indices(array=logits, k=k, is_largest=True)}\nMaximum probs at iteration: {get_top_k_mimimum_values_indices(array=correct_class_probs, k=k, is_largest=True)}'


def js_kl(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def kl_div(p: Tensor, q: Tensor, eps: float = 1e-7):
    q += eps
    q /= torch.sum(q, axis=1, keepdims=True)
    mask = p > 0
    return torch.sum(p[mask] * torch.log(p[mask] / q[mask])) / len(p)


def convert_probability_vector_to_bernoulli_kl(p) -> Tensor:
    bernoulli_p = torch.stack((p, 1 - p)).T
    return bernoulli_p


def compare_between_predicted_classes(vit_logits: Tensor, vit_s_logits: Tensor,
                                      contrastive_class_idx: Optional[Tensor] = None) -> \
        Tuple[bool, float]:
    target_class_idx = contrastive_class_idx.item() if contrastive_class_idx is not None else torch.argmax(
        vit_logits[0]).item()
    original_idx_logits_diff = (abs(max(vit_logits[0]).item() - vit_s_logits[0][target_class_idx].item()))
    is_predicted_same_class = target_class_idx == torch.argmax(vit_s_logits[0]).item()
    return is_predicted_same_class, original_idx_logits_diff


def compare_results_each_n_steps(iteration_idx: int, target: Tensor, output: Tensor, prev_x_attention: Tensor,
                                 sampled_binary_patches: Tensor = None, contrastive_class_idx: Tensor = None):
    target_class_idx = contrastive_class_idx.item() if contrastive_class_idx is not None else torch.argmax(
        target[0]).item()
    is_predicted_same_class, original_idx_logits_diff = compare_between_predicted_classes(
        vit_logits=target, vit_s_logits=output, contrastive_class_idx=contrastive_class_idx)
    print(
        f'Is predicted same class: {is_predicted_same_class}, Correct Class Prob: {F.softmax(output, dim=-1)[0][target_class_idx]}')
    if is_predicted_same_class is False:
        print(f'Predicted class change at {iteration_idx} iteration !!!!!!')
    # print(prev_x_attention)
    # if is_iteration_to_action(iteration_idx=iteration_idx, action='print'):
    # print(prev_x_attention)


def save_model(model: nn.Module, path: str) -> None:
    path = Path(f'{path}.pt')
    torch.save(model.state_dict(), path)
    print(f'Model Saved at {path}')


def save_objects(path: Path, objects_dict: Dict) -> None:
    for obj_name, obj in objects_dict.items():
        save_obj_to_disk(path=Path(path, obj_name), obj=obj)


def get_temp_to_visualize(temp):
    if len(temp.shape) == 1:  # [n_tokens]
        return temp[1:]
    elif len(temp.shape) == 2:  # [n_heads, n_tokens]
        return temp[:, 1:].reshape(12, -1)
    else:  # [n_layers, n_heads, n_tokens]
        return temp[-1, :, 1:].reshape(12, -1)


def save_text_to_file(path: Path, file_name: str, text: str):
    print(text)
    with open(Path(path, f'{file_name}.txt'), 'w') as f:
        f.write(text)


def get_minimum_prediction_string_and_write_to_disk(image_plot_folder_path, image_name, total_losses, prediction_losses,
                                                    correct_class_logits, correct_class_probs):
    minimum_predictions = get_minimum_predictions_string(image_name=image_name, total_losses=total_losses,
                                                         prediction_losses=prediction_losses,
                                                         logits=correct_class_logits,
                                                         correct_class_probs=correct_class_probs)
    save_text_to_file(path=image_plot_folder_path, file_name='minimum_predictions', text=minimum_predictions)


def create_folder(path: Path) -> Path:
    os.makedirs(name=path, exist_ok=True)
    return path


def create_attention_probs_folders(image_plot_folder_path: Path):
    mean_folder = create_folder(Path(image_plot_folder_path, 'mean'))
    median_folder = create_folder(Path(image_plot_folder_path, 'median'))
    max_folder = create_folder(Path(image_plot_folder_path, 'max'))
    min_folder = create_folder(Path(image_plot_folder_path, 'min'))
    return mean_folder, median_folder, max_folder, min_folder


def create_temp_tokens_folders(image_plot_folder_path):
    temp_tokens_folder = create_folder(Path(image_plot_folder_path, 'temp_tokens'))
    temp_tokens_mean_folder = create_folder(Path(temp_tokens_folder, 'mean'))
    temp_tokens_median_folder = create_folder(Path(temp_tokens_folder, 'median'))
    temp_tokens_max_folder = create_folder(Path(temp_tokens_folder, 'max'))
    temp_tokens_min_folder = create_folder(Path(temp_tokens_folder, 'min'))
    return temp_tokens_folder, temp_tokens_max_folder, temp_tokens_mean_folder, temp_tokens_median_folder, temp_tokens_min_folder


def create_folders(image_plot_folder_path: Path):
    mean_folder, median_folder, max_folder, min_folder = create_attention_probs_folders(image_plot_folder_path)
    temp_tokens_folder, temp_tokens_max_folder, temp_tokens_mean_folder, temp_tokens_median_folder, temp_tokens_min_folder = create_temp_tokens_folders(
        image_plot_folder_path)

    return mean_folder, median_folder, max_folder, min_folder, temp_tokens_folder, temp_tokens_mean_folder, temp_tokens_median_folder, temp_tokens_max_folder, temp_tokens_min_folder


def get_img_spec(text_dict: Dict) -> Tuple[str, Tensor, Optional[Tensor]]:
    image_name, correct_class_idx, contrastive_class_idx = text_dict['text'], text_dict['correct_class'], \
                                                           text_dict['contrastive_class']
    return image_name, correct_class_idx, contrastive_class_idx


def get_text_spec(df: pd.DataFrame, idx: int) -> Tuple[str, Tensor, Optional[Tensor]]:
    text_input, correct_class_idx = df['text'].iloc[idx], df['sentiment'].iloc[idx]
    correct_class_idx = np.where(correct_class_idx.lower() == 'pos', 1, 0)
    contrastive_class_idx = 1 - correct_class_idx
    return text_input, correct_class_idx, contrastive_class_idx


def get_iteration_target_class_stats(output, target_class_idx: Tensor):
    correct_class_logit = output.logits[0][target_class_idx].item()
    correct_class_prob = F.softmax(output.logits[0], dim=-1)[target_class_idx].item()
    prediction_loss = ce_loss(output.logits, target_class_idx) * loss_config['pred_loss_multiplier'] # Change - I've change it from target_class_idx.unsquuze(0)
    return correct_class_logit, correct_class_prob, prediction_loss


def is_iteration_to_action(iteration_idx: int, action: str = 'print') -> bool:
    """
    :param action: 'print' / 'save'
    """
    is_iter_to_action = iteration_idx > 0 and iteration_idx % bert_config[f'{action}_every'] == 0 or iteration_idx == \
                        bert_config['num_steps'] - 1
    if action == 'print':
        return bert_config['verbose'] and is_iter_to_action
    return is_iter_to_action


def setup_model_and_optimizer(model_type: str,device:str):
    vit_ours_model = handle_model_config_and_freezing_for_task(
        model=load_BertModel(bert_config, model_type=model_type,device=device),
        freezing_transformer=bert_config['freezing_transformer'])
    optimizer = optim.Adam([vit_ours_model.bert.encoder.x_attention], lr=bert_config['evidence_classifier']['lr'])
    return vit_ours_model, optimizer


def get_input_tokens(tokenizer, text):
    encoding = tokenizer(text, return_tensors='pt', max_length=512, truncation=True,
                         padding="max_length")
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    return encoding, input_ids, attention_mask

def scores_per_word_from_scores_per_token(input, tokenizer, input_ids, scores_per_id):
    words = tokenizer.convert_ids_to_tokens(input_ids)
    words = [word.replace('##', '') for word in words]
    score_per_char = []

    # TODO: DELETE
    input_ids_chars = []
    for word in words:
        if word in ['[CLS]', '[SEP]', '[UNK]', '[PAD]']:
            continue
        input_ids_chars += list(word)
    # TODO: DELETE

    for i in range(len(scores_per_id)):
        if words[i] in ['[CLS]', '[SEP]', '[UNK]', '[PAD]']:
            continue
        score_per_char += [scores_per_id[i]] * len(words[i])

    score_per_word = []
    start_idx = 0
    end_idx = 0
    # TODO: DELETE
    words_from_chars = []
    for inp in input:
        if start_idx >= len(score_per_char):
            break
        end_idx = end_idx + len(inp)
        temp_tensor = torch.tensor(score_per_char[start_idx:end_idx])
        score_per_word.append(temp_tensor.max().item())

        # TODO: DELETE
        words_from_chars.append(''.join(input_ids_chars[start_idx:end_idx]))

        start_idx = end_idx

    if (words_from_chars[:-1] != input[:len(words_from_chars) - 1]):
        print(words_from_chars)
        print(input[:len(words_from_chars)])
        print(words)
        print(tokenizer.convert_ids_to_tokens(input_ids))
        assert False

    return torch.tensor(score_per_word)

