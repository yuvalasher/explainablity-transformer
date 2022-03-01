import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from loss_utils import *
from utils.consts import *
from pathlib import WindowsPath
from pytorch_lightning import seed_everything
import pickle
from typing import Union, Any
from utils import *
from tqdm import tqdm

vit_config = config['vit']
seed_everything(config['general']['seed'])

def get_metrics_for_infer_run(tokens_mask:Tensor, output:Tensor, target:Tensor) -> Tuple[int, float, float, float, bool]:
    num_of_patches = tokens_mask.sum()
    patches_coverage_percentage = round((tokens_mask.sum() / len(tokens_mask)).item(), 3)
    correct_class_pred_prob = F.softmax(output.logits, dim=-1)[0][torch.argmax(F.softmax(target.logits, dim=-1)).item()]
    correct_class_logit = output.logits[0][torch.argmax(F.softmax(target.logits[0], dim=-1)).item()]
    is_correct_class_highest = torch.argmax(output.logits[0]) == torch.argmax(target.logits[0])
    return num_of_patches, patches_coverage_percentage, correct_class_pred_prob, correct_class_logit, is_correct_class_highest


def _print_conclusions(id2label:Dict, tokens_mask:Tensor, output:Tensor, target:Tensor, num_of_patches:int, patches_coverage_percentage: float,
                       correct_class_pred_prob:float, correct_class_logit:float, is_highest_class:bool) -> None:
    print(
        f'Num of patches: {num_of_patches}, {patches_coverage_percentage * 100}% of the tokens, '
        f'correct_class_pred: {correct_class_pred_prob}, '
        f'correct_class_logit: {correct_class_logit}, '
        f'Highest class: {torch.argmax(output.logits[0])} , {id2label[torch.argmax(output.logits[0]).item()]} with {torch.max(F.softmax(output.logits[0], dim=-1)).item()}, '
        f'Is correct class highest: {is_highest_class}')


def _load_vit_models_inputs_and_target(path: str):
    feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config)
    infer_model = load_vit_model_by_type(vit_config=vit_config, model_type='infer')
    image = get_image_from_path(Path(IMAGES_FOLDER_PATH, f"{os.path.normpath(path).split(os.path.sep)[-1]}.JPEG"))
    inputs = feature_extractor(images=image, return_tensors="pt")
    target = vit_model(**inputs)
    infer_model.eval()
    vit_model.eval()
    return inputs, target, vit_model, infer_model


def load_obj(path: Union[str, WindowsPath, Path]) -> Any:
    if type(path) == str and path[-4:] != '.pkl':
        path += '.pkl'
    elif type(path) == WindowsPath and path.suffix != '.pkl':
        path = path.with_suffix('.pkl')

    with open(path, 'rb') as f:
        return pickle.load(f)

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
            correct_random_guess.append((test_idx, torch.max(F.softmax(output.logits[0], dim=-1)).item()))
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
                                                                tokens_to_show=tokens_to_show - 1 if tokens_to_show > 1 else tokens_to_show)  # -1 as added one token of cls
        tokens_mask = torch.cat((torch.ones(1), tokens_mask))  # one for the [CLS] token
        forward_and_print_conclusions(infer_model=infer_model, vit_model=vit_model, inputs=inputs,
                                      tokens_mask=tokens_mask, target=target)


def infer_prediction(path: str, tokens_mask: Tensor = None, experiment_name: str = None):
    inputs, target, vit_model, infer_model = _load_vit_models_inputs_and_target(path=path)
    forward_and_print_conclusions(infer_model=infer_model, vit_model=vit_model, inputs=inputs,
                                  tokens_mask=tokens_mask, target=target)


def forward_model(infer_model, tokens_mask, inputs):
    if tokens_mask.shape[-1] == 576:
        tokens_mask = torch.cat((torch.ones(1), tokens_mask))  # one for the [CLS] token
    output = infer_model(**inputs, tokens_mask=tokens_mask)
    return output


def forward_and_get_metrics(infer_model, inputs, tokens_mask, target, verbose: bool = False):
    output = forward_model(infer_model=infer_model, tokens_mask=tokens_mask, inputs=inputs)
    num_of_patches, patches_coverage_percentage, correct_class_pred_prob, correct_class_logit, is_correct_class_highest = get_metrics_for_infer_run(
        tokens_mask=tokens_mask, output=output, target=target)
    if verbose:
        _print_conclusions(id2label=infer_model.config.id2label, tokens_mask=tokens_mask, output=output, target=target,
                           num_of_patches=num_of_patches, patches_coverage_percentage=patches_coverage_percentage,
                           correct_class_pred_prob=correct_class_pred_prob, correct_class_logit=correct_class_logit,
                           is_highest_class=is_correct_class_highest)
    return output, num_of_patches, patches_coverage_percentage, correct_class_pred_prob, correct_class_logit, is_correct_class_highest


def forward_and_print_conclusions(infer_model, vit_model, inputs, tokens_mask, target, verbose: bool = True):
    output, num_of_patches, patches_coverage_percentage, correct_class_pred_prob, correct_class_logit, is_highest_class = forward_and_get_metrics(
        infer_model, inputs, tokens_mask, target)
    if verbose:
        _print_conclusions(id2label=vit_model.config.id2label, tokens_mask=tokens_mask, output=output, target=target,
                           num_of_patches=num_of_patches, patches_coverage_percentage=patches_coverage_percentage,
                           correct_class_pred_prob=correct_class_pred_prob, correct_class_logit=correct_class_logit,
                           is_highest_class=is_highest_class)


def get_iteration_idx_of_minimum_loss(path) -> int:
    losses = load_obj(path=Path(path, 'losses'))
    return torch.argmin(torch.tensor(losses)).item()


def get_tokens_mask_by_iteration_idx(path, iteration_idx: int) -> Tensor:
    return load_obj(path=Path(path, 'tokens_mask'))[iteration_idx]


def is_tokens_mask_binary(tokens_mask: Tensor) -> bool:
    return torch.equal(torch.tensor(len(tokens_mask)),
                       torch.count_nonzero((tokens_mask == 0) | (tokens_mask == 1)))


def load_tokens_mask(path, iteration_idx: int = None) -> Tuple[int, Tensor]:
    if iteration_idx is None:
        iteration_idx = get_iteration_idx_of_minimum_loss(path=path)
        print(f'Minimum prediction loss at iteration: {iteration_idx}')
    else:
        print(f'Get tokens mask of iteration: {iteration_idx}')
    tokens_mask = get_tokens_mask_by_iteration_idx(path=path, iteration_idx=iteration_idx)
    return iteration_idx, tokens_mask


def get_binary_token_mask(path, tokens_to_show: int) -> Tensor:
    iteration_idx, tokens_mask = load_tokens_mask(path=path)
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
        iteration_idx, tokens_mask = load_tokens_mask(
            path=Path(experiment_path, image_name.replace('ILSVRC2012_val_', '')))
        saliency_map_output = infer_model(**inputs, tokens_mask=tokens_mask)
        full_image_outputs = vit_model(**inputs)
        saliency_map_confidence = F.softmax(saliency_map_output.logits, dim=-1)[0][
            torch.argmax(F.softmax(full_image_outputs.logits, dim=-1)).item()]

        full_image_confidence = torch.argmax(F.softmax(full_image_outputs.logits, dim=-1)).item()
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


def get_top_k_tokens_from_mask(tokens_mask: Tensor, k: int) -> Tensor:
    mask = torch.zeros_like(tokens_mask)
    _, indices = torch.topk(tokens_mask, k, largest=True)
    mask[indices] = 1
    return mask


def get_dino_attentions(path):
    return load_obj(path=Path(path, 'dino', 'attentions.pkl'))


def mkdir_evaluation_picture_folder(picture_name: str) -> None:
    os.makedirs(name=Path(Path.cwd(), picture_name), exist_ok=True)


def overlap_of_tokens_between_methods(method_a_tokens_indices, method_b_tokens_indices):
    return len(np.intersect1d(method_a_tokens_indices, method_b_tokens_indices))


def run_infer_difference_dino_and_temp_by_k_patches_iterative(path: str, iteration_idx: int = None):
    picture_name = path.split('\\')[-1]
    mkdir_evaluation_picture_folder(picture_name=picture_name)
    iteration_idx, tokens_mask = load_tokens_mask(path=path, iteration_idx=iteration_idx)  # saved as "tokens_mask.pkl"
    ours_tokens_mask = avg_attention_scores_heads(tokens_mask)
    dino_tokens_mask = avg_attention_scores_heads(get_dino_attentions(path=path))  # saved as "dino/attentions.pkl"
    inputs, target, vit_model, infer_model = _load_vit_models_inputs_and_target(path=path)
    step_size = 2
    # forward_and_print_conclusions(infer_model=infer_model, vit_model=vit_model, inputs=inputs,
    #                               tokens_mask=torch.ones_like(ours_tokens_mask), target=target)

    tokens_percentage = []
    ours_is_correct_class = []
    dino_is_correct_class = []
    ours_logits = []
    ours_probs = []
    dino_probs = []
    dino_logits = []
    for k_patches in range(5, 576, step_size):
        our_mask = get_top_k_tokens_from_mask(tokens_mask=ours_tokens_mask, k=k_patches)
        dino_mask = get_top_k_tokens_from_mask(tokens_mask=dino_tokens_mask, k=k_patches)
        # forward_and_print_conclusions(infer_model=infer_model, vit_model=vit_model, inputs=inputs, tokens_mask=our_mask, target=target)
        ours_output, ours_num_of_patches, ours_patches_coverage_percentage, ours_correct_class_pred_prob, ours_correct_class_logit, ours_is_correct_class_highest = forward_and_get_metrics(
            infer_model=infer_model, inputs=inputs, tokens_mask=our_mask, target=target, verbose=True)

        dino_output, dino_num_of_patches, dino_patches_coverage_percentage, dino_correct_class_pred_prob, dino_correct_class_logit, dino_is_correct_class_highest = forward_and_get_metrics(
            infer_model=infer_model, inputs=inputs, tokens_mask=dino_mask, target=target, verbose=True)
        # print(torch.where(our_mask))
        # print(torch.where(dino_mask))
        # print(torch.topk(dino_mask, k_patches, largest=True)[1])
        print(f'Num of tokens intersection: {overlap_of_tokens_between_methods(our_mask, dino_mask)}')
        print(
            '------------------------------------------------------------------------------------------------------------------------------------------------')
        tokens_percentage.append(ours_patches_coverage_percentage)
        ours_is_correct_class.append(ours_is_correct_class_highest)
        dino_is_correct_class.append(dino_is_correct_class_highest)
        ours_logits.append(ours_correct_class_logit)
        ours_probs.append(ours_correct_class_pred_prob)
        dino_probs.append(dino_correct_class_pred_prob)
        dino_logits.append(dino_correct_class_logit)

    plot_metrics(tokens_percentage=tokens_percentage, dino_values=dino_is_correct_class, ours_values=ours_is_correct_class,
                 metric_value_name='Is Correct Class Highest', name=picture_name, iteration_idx=iteration_idx, step_size=step_size)
    plot_metrics(tokens_percentage=tokens_percentage, dino_values=dino_logits, ours_values=ours_logits,
                 metric_value_name='Logit', name=picture_name, iteration_idx=iteration_idx, step_size=step_size)
    plot_metrics(tokens_percentage=tokens_percentage, dino_values=dino_probs, ours_values=ours_probs,
                 metric_value_name='Prob', name=picture_name, iteration_idx=iteration_idx, step_size=step_size)
    plot_sorted_vector(our_attention=ours_tokens_mask, dino_attention=dino_tokens_mask, title=f'iter_{iteration_idx}',
                       folder_name=picture_name)

def plot_metrics(tokens_percentage, dino_values, ours_values, metric_value_name: str, name: str, iteration_idx: int,
                 step_size: int):
    multiline(tokens_percentage=tokens_percentage, ours_values=ours_values, dino_values=dino_values,
              title=f'{metric_value_name} Per Tokens Percentage ({name}); iter: {iteration_idx}, step_size={step_size}',
              x_label='Tokens Percentage',
              y_label=f'{metric_value_name}', filename=f'{metric_value_name}_iter_{iteration_idx}', folder_name=name)


def plot_sorted_vector(our_attention: Tensor, dino_attention: Tensor, title: str, folder_name: str):
    plt.plot(sorted(our_attention.detach().numpy()))
    plt.plot(sorted(dino_attention.detach().numpy()))
    plt.legend(['Ours', 'Dino'])
    plt.title(f'Sorted Attention Values - {title}')
    plt.savefig(fname=Path(Path.cwd(), folder_name, f'sorted_mask_{title}.png'), format='png')
    plt.show()


def multiline(tokens_percentage, ours_values, dino_values, title: str, x_label: str, y_label: str, filename: str,
              folder_name):
    if type(ours_values) is list:
        ours_values = np.array(ours_values)
    if type(dino_values) is list:
        dino_values = np.array(dino_values)
    if type(tokens_percentage) is list:
        tokens_percentage = np.array(tokens_percentage)

    p = sns.lineplot(np.array(tokens_percentage), np.array(ours_values))
    sns.lineplot(np.array(tokens_percentage), np.array(dino_values))
    p.set(title=title, xlabel=x_label, ylabel=y_label)
    plt.legend(labels=["Ours", "Dino"])
    plt.savefig(fname=Path(Path.cwd(), folder_name, f'{filename}.png'), format='png')
    plt.show()


def lineplot(x, y, title: str, x_label: str, y_label: str):
    if type(x) is list:
        x = np.array(x)
    if type(y) is list:
        y = np.array(y)

    p = sns.lineplot(np.array(x), np.array(y))
    p.set(title=title, xlabel=x_label, ylabel=y_label)
    plt.show()


def avg_attention_scores_heads(attentions: Tensor):
    return attentions.mean(dim=0)


if __name__ == '__main__':
    # run_infer_difference_dino_and_temp_by_k_patches_iterative(path=path)
    # run_infer_difference_dino_and_temp_by_k_patches_iterative(path=path, iteration_idx=13)

    # p = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\head_layer_temp_softmax_lr0_003_temp_1+l1_0+kl_loss_0+entropy_loss_100+pred_loss_10\00000001"
    # p = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\head_layer_temp_softmax_lr0_003_temp_1+l1_0+kl_loss_0+entropy_loss_100+pred_loss_10\00000018"
    # p = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\head_layer_temp_softmax_lr0_003_temp_1+l1_0+kl_loss_0+entropy_loss_100+pred_loss_10\00000003"
    # run_infer_difference_dino_and_temp_by_k_patches_iterative(path=p, iteration_idx=140)
    # run_infer_difference_dino_and_temp_by_k_patches_iterative(path=p, iteration_idx=113)

    # p = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\head_layer_temp_softmax_lr0_003_temp_1+l1_0+kl_loss_0+entropy_loss_100+pred_loss_10\00000009"
    # run_infer_difference_dino_and_temp_by_k_patches_iterative(path=p)
    # run_infer_difference_dino_and_temp_by_k_patches_iterative(path=p, iteration_idx=99)

    # p = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\head_layer_temp_softmax_lr0_003_temp_1+l1_0+kl_loss_0+entropy_loss_100+pred_loss_10\00000001"
    # run_infer_difference_dino_and_temp_by_k_patches_iterative(path=p)

    # p = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\head_layer_temp_softmax_lr0_003_temp_1+l1_0+kl_loss_0+entropy_loss_100+pred_loss_10\00000003"
    # run_infer_difference_dino_and_temp_by_k_patches_iterative(path=p, iteration_idx=140)
    # run_infer_difference_dino_and_temp_by_k_patches_iterative(path=p, iteration_idx=113)

    # p = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\head_layer_temp_softmax_lr0_003_temp_1+l1_0+kl_loss_0+entropy_loss_100+pred_loss_10\00000327"
    # run_infer_difference_dino_and_temp_by_k_patches_iterative(path=p)
    # run_infer_difference_dino_and_temp_by_k_patches_iterative(path=p, iteration_idx=103)
    # p = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\head_layer_temp_softmax_lr0_003_temp_1+l1_0+kl_loss_0+entropy_loss_1000+pred_loss_10\00000009"
    # run_infer_difference_dino_and_temp_by_k_patches_iterative(path=p)
    # run_infer_difference_dino_and_temp_by_k_patches_iterative(path=p, iteration_idx=103)

    p = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\head_layer_temp_softmax_lr0_003_temp_1+l1_0+kl_loss_0+entropy_loss_1000+pred_loss_10\00000003"
    run_infer_difference_dino_and_temp_by_k_patches_iterative(path=p)
