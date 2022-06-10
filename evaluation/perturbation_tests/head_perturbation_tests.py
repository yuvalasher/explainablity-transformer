import glob
from typing import Union
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.imagenet_results_dataset import ImagenetResults
from evaluation.evaluation_utils import normalize, calculate_auc, patch_score_to_image, get_precision_at_k
from utils.consts import EXPERIMENTS_FOLDER_PATH
from vit_utils import *
import torch
import os

from torch import Tensor
from tqdm import tqdm
import numpy as np
import argparse
from config import config

vit_config = config['vit']
evaluation_config = vit_config['evaluation']

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


def eval(experiment_dir: Path, model, feature_extractor, heads_masks: Tensor, image: Tensor) -> Dict[int, int]:
    """
    heads mask is a tensor with shape of [12, 196] for 12 heads
    """
    num_samples = 0
    # num_correct_model = np.zeros((len(imagenet_ds, )))
    model_index = 0

    base_size = 224 * 224
    perturbation_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # num_correct_pertub = np.zeros((len(perturbation_steps), len(imagenet_ds)))  # 9 is the number of perturbation steps
    perturb_index = 0
    change_predicted_class_by_head = {}

    data = image.reshape(1, 3, 224, 224)
    for head_idx, vis in enumerate(heads_masks):
        if len(heads_masks.shape) == 1:
            vis = patch_score_to_image(heads_masks)
        else:
            vis = patch_score_to_image(vis)
        num_samples += len(data)

        inputs = feature_extractor(images=data.squeeze(0).cpu(), return_tensors="pt")
        inputs = {'pixel_values': inputs['pixel_values'].to(device)}
        pred = model(**inputs)
        probs = torch.softmax(pred.logits, dim=1)
        target = torch.tensor([torch.argmax(probs).item()]).to(device)
        pred_probabilities, pred_org_logit, pred_org_prob, pred_class, tgt_pred, num_correct_model = get_model_infer_metrics(
            model_index=model_index, pred=pred.logits, target=target)
        if vit_config['verbose']:
            print(
                f'\nOriginal Image. Top Class: {pred.logits[0].argmax(dim=0).item()}, Max logits: {round(pred.logits[0].max(dim=0)[0].item(), 2)}, Max prob: {round(probs[0].max(dim=0)[0].item(), 5)}; Correct class logit: {round(pred.logits[0][target].item(), 2)} Correct class prob: {round(probs[0][target].item(), 5)}')

        org_shape = data.shape
        if evaluation_config['perturbation_type'] == 'neg':
            vis = -vis
        elif evaluation_config['perturbation_type'] == 'pos':
            vis = vis
        else:
            raise (NotImplementedError(f'perturbation_type config {vit_config["perturbation_type"]} not exists'))
        vis = vis.reshape(org_shape[0], -1)  # org_shape[0] = 1

        for i in range(len(perturbation_steps)):
            _data = data.clone()
            _data = get_perturbated_data(vis=torch.tensor(vis), image=_data, perturbation_step=perturbation_steps[i],
                                         base_size=base_size)
            # if vit_config['verbose']:
            #     plot_image(_data, step_idx=i)
            inputs = feature_extractor(images=_data.squeeze(0).cpu(), return_tensors="pt")
            inputs = {'pixel_values': inputs['pixel_values'].to(device)}
            out = model(**inputs)

            if vit_config['verbose']:
                print(
                    f'Head: {head_idx}; {100 * perturbation_steps[i]}% pixels blacked. Top Class: {out.logits[0].argmax(dim=0).item()}, Max logits: {round(out.logits[0].max(dim=0)[0].item(), 2)}, Max prob: {round(F.softmax(pred_probabilities[0], dim=0).max(dim=0)[0].item(), 5)}; Correct class logit: {round(out.logits[0][target].item(), 2)} Correct class prob: {round(pred_probabilities[0][target].item(), 5)}')

            # Target-Class Comparison
            target_class = out.logits.data.max(1, keepdim=True)[1].squeeze(1)
            temp = (target == target_class).type(target.type()).data.cpu().numpy().item()
            if temp == 0 and head_idx not in change_predicted_class_by_head.keys():  # class changed
                change_predicted_class_by_head[head_idx] = i

        model_index += len(target)
        perturb_index += len(target)
        if len(heads_masks.shape) == 1:
            print(change_predicted_class_by_head)
            return change_predicted_class_by_head
    print(change_predicted_class_by_head)
    return change_predicted_class_by_head


def get_auc(num_correct_pertub: np.ndarray):
    mean_accuracy_by_step = np.mean(num_correct_pertub, axis=1)
    mean_accuracy_by_step = np.insert(mean_accuracy_by_step, 0, 1)  # TODO - accuracy for class = top-class (predicted)
    auc = calculate_auc(mean_accuracy_by_step=mean_accuracy_by_step) * 100
    print(f'AUC: {auc}%')
    return auc



def get_precision_at_k_by_k_heads(gt_order_heads: List[int], heads_order: List[int], k: int = 5):
    y_true = gt_order_heads
    y_pred = heads_order
    precision_at_k = get_precision_at_k(y_true=y_true, y_pred=y_pred, k=k)
    return precision_at_k


def get_gt_heads_order(change_predicted_class_by_head:Dict[int, int]):
    y_true_items: List[Tuple[int, int]] = sorted(change_predicted_class_by_head.items(), key=lambda x: x[1])
    y_true = [item[0] for item in y_true_items]
    return y_true


def save_objects(experiment_dir: Path, num_correct_model, dissimilarity_model, num_correct_pertub, dissimilarity_pertub,
                 logit_diff_pertub, prob_diff_pertub, perturb_index, perturbation_steps):
    np.save(Path(experiment_dir, 'model_hits.npy'), num_correct_model)
    np.save(Path(experiment_dir, 'model_dissimilarities.npy'), dissimilarity_model)
    np.save(Path(experiment_dir, 'perturbations_hits.npy'), num_correct_pertub[:, :perturb_index])
    np.save(Path(experiment_dir, 'perturbations_dissimilarities.npy'), dissimilarity_pertub[:, :perturb_index])
    np.save(Path(experiment_dir, 'perturbations_logit_diff.npy'), logit_diff_pertub[:, :perturb_index])
    np.save(Path(experiment_dir, 'perturbations_prob_diff.npy'), prob_diff_pertub[:, :perturb_index])
    print(f'Mean num correct: {np.mean(num_correct_model)}, std num correct {np.std(num_correct_model)}')
    print(f'Mean dissimilarity : {np.mean(dissimilarity_model)}, std dissimilarity {np.std(dissimilarity_model)}')
    print(f'Perturbation Steps: {perturbation_steps}')
    print(
        f'Mean num_correct_pertub: {np.mean(num_correct_pertub, axis=1)}, std num_correct_pertub {np.std(num_correct_pertub, axis=1)}')
    print(
        f'Mean dissimilarity_pertub : {np.mean(dissimilarity_pertub, axis=1)}, std dissimilarity_pertub {np.std(dissimilarity_pertub, axis=1)}')


def plot_image(data, step_idx: int = None) -> None:
    if (step_idx in [1, 2, 3, 4] or step_idx is None) and not torch.cuda.is_available():
        im = transforms.ToPILImage()(data.squeeze(0))
        plt.imshow(im)
        plt.show()


def get_perturbated_data(vis: Tensor, image: Tensor, perturbation_step: Union[float, int], base_size: int):
    """
    vis - Masking of the image (1, 224, 224)
    pic - original image (3, 224, 224)
    """
    _data = image.clone()
    org_shape = (1, 3, 224, 224)
    _, idx = torch.topk(vis, int(base_size * perturbation_step), dim=-1)  # vis.shape (50176) / 2 = 25088
    idx = idx.unsqueeze(1).repeat(1, org_shape[1], 1)
    _data = _data.reshape(org_shape[0], org_shape[1], -1)
    _data = _data.scatter_(-1, idx.reshape(1, org_shape[1], -1), 0)
    _data = _data.reshape(*org_shape)
    return _data


def get_model_infer_metrics(model_index, pred, target, num_correct_model=None):
    pred_probabilities = torch.softmax(pred, dim=1)
    pred_org_logit = pred.data.max(1, keepdim=True)[0].squeeze(1)
    pred_org_prob = pred_probabilities.data.max(1, keepdim=True)[0].squeeze(1)
    pred_class = pred.data.max(1, keepdim=True)[1].squeeze(1)
    tgt_pred = (target == pred_class).type(target.type()).data.cpu().numpy()
    if num_correct_model is not None:
        num_correct_model[model_index:model_index + len(tgt_pred)] = tgt_pred
        return pred_probabilities, pred_org_logit, pred_org_prob, pred_class, tgt_pred
    return pred_probabilities, pred_org_logit, pred_org_prob, pred_class, tgt_pred, num_correct_model


def get_data_vis_and_target(data, target, vis):
    data = data.to(device)
    vis = vis.to(device)
    target = target.to(device)
    return data, target, vis


def update_results_df(results_df: pd.DataFrame, vis_type: str, auc: float):
    return results_df.append({'vis_type': vis_type, 'auc': auc}, ignore_index=True)


if __name__ == "__main__":
    results_df = pd.DataFrame(columns=['vis_type', 'auc'])
    # VIS_TYPES = ['vis_min_pred_loss_rollout_grad_max', 'vis_max_logits_rollout_grad_max', 'vis_90_rollout_grad_max',
    #              'vis_100_rollout_grad_max', 'vis_110_rollout_grad_max', 'vis_120_rollout_grad_max',
    #              'vis_130_rollout_grad_max', 'vis_140_rollout_grad_max', 'vis_150_rollout_grad_max',
    #              'vis_160_rollout_grad_max', 'vis_165_rollout_grad_max', 'vis_170_rollout_grad_max',
    #              'vis_175_rollout_grad_max', 'vis_180_rollout_grad_max', 'vis_185_rollout_grad_max',
    #              'vis_190_rollout_grad_max', 'vis_min_pred_loss_rollout_mean_relu_grad',
    #              'vis_max_logits_rollout_mean_relu_grad', 'vis_90_rollout_mean_relu_grad',
    #              'vis_100_rollout_mean_relu_grad', 'vis_110_rollout_mean_relu_grad', 'vis_120_rollout_mean_relu_grad',
    #              'vis_130_rollout_mean_relu_grad', 'vis_140_rollout_mean_relu_grad', 'vis_150_rollout_mean_relu_grad',
    #              'vis_160_rollout_mean_relu_grad', 'vis_165_rollout_mean_relu_grad', 'vis_170_rollout_mean_relu_grad',
    #              'vis_175_rollout_mean_relu_grad', 'vis_180_rollout_mean_relu_grad', 'vis_185_rollout_mean_relu_grad',
    #              'vis_190_rollout_mean_relu_grad']
    # experiment_path = Path(EXPERIMENTS_FOLDER_PATH, 'temp', vit_config['evaluation']['experiment_folder_name'])
    feature_extractor, model = load_feature_extractor_and_vit_model(vit_config=vit_config,
                                                                    model_type='vit-for-dino',
                                                                    is_wolf_transforms=vit_config['is_wolf_transforms'])
    model.to(device)
    model.eval()
    for vis_type in VIS_TYPES:
        print(vis_type)
        vit_type_experiment_path = create_folder(Path(experiment_path, vis_type))
        imagenet_ds = ImagenetResults(path=experiment_path, vis_type=vis_type)
        sample_loader = DataLoader(imagenet_ds, batch_size=evaluation_config['batch_size'], shuffle=False)
        auc = eval(experiment_dir=vit_type_experiment_path, model=model, feature_extractor=feature_extractor)
        results_df = update_results_df(results_df=results_df, vis_type=vis_type, auc=auc)
        print(results_df)
        results_df.to_csv(Path(experiment_path, 'results_df.csv'), index=False)
