from loss_utils import *
from utils.consts import *
from pathlib import WindowsPath
from pytorch_lightning import seed_everything
import pickle
from typing import Union

vit_config = config['vit']
seed_everything(config['general']['seed'])


def _print_conclusions(id2label, tokens_mask, output, target) -> None:
    print(
        f'Num of patches: {tokens_mask.sum()}, {round((tokens_mask.sum() / len(tokens_mask)).item(), 2) * 100}% of the tokens, '
        f'correct_class_pred: {F.softmax(output.logits)[0][torch.argmax(F.softmax(target.logits)).item()]}, '
        f'correct_class_logit: {output.logits[0][torch.argmax(F.softmax(target.logits[0])).item()]}, '
        f'Highest class: {torch.argmax(output.logits[0])} , {id2label[torch.argmax(output.logits[0]).item()]} with {torch.max(F.softmax(output.logits[0])).item()}, '
        f'Is highest class: {torch.argmax(output.logits[0]) == torch.argmax(target.logits[0])}')


def _load_vit_models_inputs_and_target(path: str):
    feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config)
    infer_model = load_vit_model_by_type(vit_config=vit_config, model_type='infer')
    image = get_image_from_path(Path(IMAGES_FOLDER_PATH, f"{os.path.normpath(path).split(os.path.sep)[-1]}.JPEG"))
    inputs = feature_extractor(images=image, return_tensors="pt")
    target = vit_model(**inputs)
    return inputs, target, vit_model, infer_model


def load_obj(path: Union[str, WindowsPath, Path]) -> Any:
    if type(path) == str and path[-4:] != '.pkl':
        path += '.pkl'
    elif type(path) == WindowsPath and path.suffix != '.pkl':
        path = path.with_suffix('.pkl')

    with open(path, 'rb') as f:
        return pickle.load(f)


# def load_model(path: Union[str, WindowsPath, Path]) -> nn.Module:
#     # path = Path(f'{PICKLES_FOLDER_PATH}', f'{model_name}.pt')
#     if type(path) == str and path[-3:] != '.pt':
#         path += '.pt'
#     elif type(path) == WindowsPath and path.suffix != '.pt':
#         path = path.with_suffix('.pt')
#
#     c = ViTConfig()
#     c.image_size = vit_config['img_size']
#     c.num_labels = vit_config['num_labels']
#     model = 7dForImageClassification(config=c)
#     model.load_state_dict(torch.load(path))
#     return model


def dark_random_k_patches(percentage_to_dark: float, n_patches: int = 577) -> Tensor:
    k = int(n_patches * percentage_to_dark)
    random_vector = torch.rand(n_patches)
    k_th_quant = torch.topk(random_vector, k, largest=False)[0][-1]
    mask = (random_vector >= k_th_quant).int()
    return mask


def test_dark_random_k_patches(path, num_tests: int, percentage_to_dark: float):
    correct_random_guess = []
    inputs, target, vit_model, infer_model = _load_vit_models_inputs_and_target(path=path)
    for test_idx in range(num_tests):
        tokens_mask = dark_random_k_patches(percentage_to_dark=percentage_to_dark)
        output = infer_model(**inputs, tokens_mask=tokens_mask)
        if torch.argmax(output.logits[0]) == torch.argmax(target.logits[0]):
            correct_random_guess.append((test_idx, torch.max(F.softmax(output.logits[0])).item()))
        print(f'*** {test_idx} ***')
        _print_conclusions(id2label=vit_model.config.id2label, tokens_mask=tokens_mask, output=output, target=target)
    print(f'Number of correct random guesses: {len(correct_random_guess)}')
    print(correct_random_guess)


"""
def generate_sampled_binary_patches_by_bernoulli(distribution: Tensor, percentage_to_dark: float) -> Tensor:
    dist = distribution.clone()
    return dist.detach().apply_(lambda x: bernoulli.rvs(x, size=1)[0])
    # return torch.tensor([bernoulli.rvs(p.item(),size=1)[0] for p in dist])
"""


def generate_binary_tokens_mask_by_top_scores(distribution: Tensor, tokens_to_show: int) -> Tensor:
    dist = distribution.clone()
    k = tokens_to_show  # int(len(dist) * percentage_to_dark)
    k_th_quant = torch.topk(dist, k)[0][-1]
    mask = (dist >= k_th_quant).int()
    return mask


def get_dino_probability_per_head(path: str, tokens_to_show: int):
    attentions = load_obj(path=Path(path, 'dino', 'attentions.pkl'))
    inputs, target, vit_model, infer_model = _load_vit_models_inputs_and_target(path=path)
    print('Dino heads')
    for attention_head in attentions:
        tokens_mask = generate_binary_tokens_mask_by_top_scores(distribution=attention_head,
                                                                tokens_to_show=tokens_to_show - 1)  # -1 as added one token of cls
        tokens_mask = torch.cat((torch.ones(1), tokens_mask))  # one for the [CLS] token
        forward_and_print_conclusions(infer_model=infer_model, vit_model=vit_model, inputs=inputs,
                                      tokens_mask=tokens_mask, target=target)


def infer_prediction(path: str, tokens_mask: Tensor = None, experiment_name: str = None):
    inputs, target, vit_model, infer_model = _load_vit_models_inputs_and_target(path=path)
    forward_and_print_conclusions(infer_model=infer_model, vit_model=vit_model, inputs=inputs,
                                  tokens_mask=tokens_mask, target=target)


def forward_and_print_conclusions(infer_model, vit_model, inputs, tokens_mask, target, verbose: bool = True):
    output = infer_model(**inputs, tokens_mask=tokens_mask)
    if verbose:
        _print_conclusions(id2label=vit_model.config.id2label, tokens_mask=tokens_mask, output=output, target=target)


def get_iteration_idx_of_minimum_loss(path) -> int:
    losses = load_obj(path=Path(path, 'losses'))
    return torch.argmin(torch.tensor(losses)).item()


def get_tokens_mask_by_iteration_idx(path, iteration_idx: int) -> Tensor:
    return load_obj(path=Path(path, 'tokens_mask'))[iteration_idx]


def is_tokens_mask_binary(tokens_mask: Tensor) -> bool:
    return torch.equal(torch.tensor(len(tokens_mask)),
                       torch.count_nonzero((tokens_mask == 0) | (tokens_mask == 1)))


def load_tokens_task(path) -> Tensor:
    iteration_idx = get_iteration_idx_of_minimum_loss(path=path)
    print(f'Minimum prediction loss at iteration: {iteration_idx}')
    tokens_mask = get_tokens_mask_by_iteration_idx(path=path, iteration_idx=iteration_idx)
    return tokens_mask


def get_binary_token_mask(path, tokens_to_show: int) -> Tensor:
    tokens_mask = load_tokens_task(path=path)
    if not is_tokens_mask_binary(tokens_mask=tokens_mask):
        tokens_mask = generate_binary_tokens_mask_by_top_scores(distribution=tokens_mask, tokens_to_show=tokens_to_show)

    return tokens_mask


def run_infer(path, tokens_mask: Tensor = None) -> None:
    if tokens_mask is None:
        tokens_mask = get_binary_token_mask(path=path, tokens_to_show=OBJCTIVE_1_AND_2_N_TOKENS_TO_PRED_BY)

    infer_prediction(path=path, tokens_mask=torch.ones_like(tokens_mask))
    infer_prediction(path=path, tokens_mask=tokens_mask)
    get_dino_probability_per_head(path=path, tokens_to_show=int(tokens_mask.sum().item()))


def calculate_avg_drop_percentage(full_image_confidence: float, saliency_map_confidence: float) -> float:
    return max(0, full_image_confidence - saliency_map_confidence) / full_image_confidence


def calculate_percentage_increase_in_confidence(full_image_confidence: float, saliency_map_confidence: float) -> float:
    return 1 if full_image_confidence < saliency_map_confidence else 0  # should be averaged


def calculate_metrics_for_experience(experiment_path: str):
    feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config)
    infer_model = load_vit_model_by_type(vit_config=vit_config, model_type='infer')
    percentage_increase_in_confidence_indicators = []
    avg_drop_percentage = []
    for idx, image_name in enumerate(os.listdir(IMAGES_FOLDER_PATH)):
        image = get_image_from_path(os.path.join(IMAGES_FOLDER_PATH, image_name))
        inputs = feature_extractor(images=image, return_tensors="pt")
        tokens_mask = load_tokens_task(path=Path(experiment_path, image_name.replace('ILSVRC2012_val_', '')))
        saliency_map_output = infer_model(**inputs, tokens_mask=tokens_mask)
        full_image_outputs = vit_model(**inputs)
        saliency_map_confidence = F.softmax(saliency_map_output.logits)[0][
            torch.argmax(F.softmax(full_image_outputs.logits)).item()]

        full_image_confidence = torch.argmax(F.softmax(full_image_outputs.logits)).item()
        avg_drop_percentage.append(calculate_avg_drop_percentage(full_image_confidence=full_image_confidence,
                                                                 saliency_map_confidence=saliency_map_confidence))
        percentage_increase_in_confidence_indicators.append(
            calculate_percentage_increase_in_confidence(full_image_confidence=full_image_confidence,
                                                        saliency_map_confidence=saliency_map_confidence))
    percentage_increase_in_confidence = 100 * sum(percentage_increase_in_confidence_indicators) / len(
        percentage_increase_in_confidence_indicators)
    averaged_drop_precetage = 100 * sum(avg_drop_percentage)

    print(
        f'% Increase in Confidence: {round(percentage_increase_in_confidence, 4)}%; Average Drop %: {round(averaged_drop_precetage, 4)}%')


if __name__ == '__main__':
    """
    Input: path to folder of image
    """
    OBJCTIVE_1_AND_2_N_TOKENS_TO_PRED_BY = int(0.2 * 577)  # TODO - change
    # experiment_image_path = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\objective_gumble_softmax_lr0_3_temp_1+l1_0+kl_loss_1+entropy_loss_0+pred_loss_3\00000018"
    # ALL_LAYERS_PATH = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\layer_all_objective_gumble_softmax_lr0_3_temp_1+l1_0+kl_loss_1+entropy_loss_0+pred_loss_3\00000018"
    # LAYER_0_PATH = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\layer_0_objective_gumble_softmax_lr0_3_temp_1+l1_0+kl_loss_1+entropy_loss_0+pred_loss_3\00000018"
    # LAYER_11_PATH = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\layer_11_objective_gumble_softmax_lr0_3_temp_1+l1_0+kl_loss_1+entropy_loss_0+pred_loss_3\00000018"
    # d = {'all layers': ALL_LAYERS_PATH, 'layer 0 only': LAYER_0_PATH, 'layer 11 only': LAYER_11_PATH}
    # for name, path in d.items():
    #     print(name)
        # run_infer(path=path)
        # n_tokens = int(tokens_mask.sum().item())
        # n_tokens = 10
        # get_dino_probability_per_head(path=path, tokens_to_show=n_tokens)

    plane_path = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\objective_gumble_softmax_lr0_3_temp_1+l1_0+kl_loss_1+entropy_loss_0+pred_loss_3\00000000"
    run_infer(path=plane_path)
