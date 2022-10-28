from icecream import ic
import glob
from typing import Union, Any
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.imagenet_results_dataset import ImagenetResults
from evaluation.evaluation_utils import normalize, calculate_auc, load_obj_from_path

from utils.consts import EXPERIMENTS_FOLDER_PATH

# from vit_utils import *
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import os

from torch import Tensor
from tqdm import tqdm
import numpy as np
import argparse
from config import config

vit_config = config['vit']
evaluation_config = vit_config['evaluation']

# device = torch.device(type='cuda', index=config["general"]["gpu_index"])
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

def normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def eval_perturbation_test(experiment_dir: Path, model, outputs, perturbation_type: str = "POS",
                           vis_class: str = "TOP", target_class: int = None) -> float:
    # ic(perturbation_type, vis_class, target_class)
    # print(f"Target class:{model.config.id2label[target_class] if target_class is not None else None}")

    num_samples = 0
    n_samples = sum(output["image_resized"].shape[0] for output in outputs)
    num_correct_model = np.zeros((n_samples))
    model_index = 0

    base_size = 224 * 224
    perturbation_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    num_correct_pertub = np.zeros((len(perturbation_steps), n_samples))  # 9 is the num perturbation steps
    dissimilarity_pertub = np.zeros((len(perturbation_steps), n_samples))
    logit_diff_pertub = np.zeros((len(perturbation_steps), n_samples))
    prob_diff_pertub = np.zeros((len(perturbation_steps), n_samples))
    perturb_index = 0
    for batch in outputs:
        for data, vis in zip(batch["image_resized"], batch["image_mask"]):
            data = data.unsqueeze(0)
            vis = vis.unsqueeze(0)
            num_samples += len(data)
            # target = torch.tensor(1)  # run by the model or injected
            # data = image
            # vis = outputs.vis
            data, vis = move_to_device_data_vis_and_target(data=data, vis=vis)

            # Compute model accuracy
            if vit_config['verbose']:
                plot_image(data)

            norm_data = normalize(data.clone())
            inputs = {'pixel_values': norm_data}
            pred = model(**inputs)
            probs = torch.softmax(pred.logits, dim=1)
            # target_probs = torch.gather(probs, 1, target[:, None])[:, 0]
            # second_probs = probs.data.topk(2, dim=1)[0][:, 1]
            # temp = torch.log(target_probs / second_probs).data.cpu().numpy()
            # dissimilarity_model[model_index:model_index + len(temp)] = temp
            if vis_class == "TARGET":
                if target_class is not None:
                    target = torch.tensor([target_class])
                else:
                    raise (f"vis_class can't be {vis_class} and target_class be {target_class}")
            elif vis_class == "TOP":
                target = torch.tensor([torch.argmax(probs).item()])
            else:
                raise (f"vis_class can't be {vis_class}")
            target = target.to(device)
            pred_probabilities, pred_org_logit, pred_org_prob, pred_class, tgt_pred, num_correct_model = get_model_infer_metrics(
                # TODO - verify
                model_index=model_index, num_correct_model=num_correct_model, pred=pred.logits, target=target)
            if vit_config['verbose']:
                print(
                    f'\nOriginal Image. Top Class: {pred.logits[0].argmax(dim=0).item()}, Max logits: {round(pred.logits[0].max(dim=0)[0].item(), 2)}, Max prob: {round(probs[0].max(dim=0)[0].item(), 5)}; Correct class logit: {round(pred.logits[0][target].item(), 2)} Correct class prob: {round(probs[0][target].item(), 5)}')

            # Save original shape
            org_shape = data.shape

            if perturbation_type == 'NEG':
                vis = -vis
            elif perturbation_type == 'POS':
                vis = vis
            else:
                raise (NotImplementedError(f'perturbation_type config {perturbation_type} not exists'))

            vis = vis.reshape(org_shape[0], -1)

            for i in range(len(perturbation_steps)):
                _data = data.clone()
                _data = get_perturbated_data(vis=vis, image=_data, perturbation_step=perturbation_steps[i],
                                             base_size=base_size)
                if vit_config['verbose']:
                    plot_image(_data)
                _norm_data = normalize(_data)
                inputs = {'pixel_values': _norm_data}
                out = model(**inputs)

                # Probabilities Comparison
                pred_probabilities = torch.softmax(out.logits, dim=1)
                pred_prob = pred_probabilities.data.max(1, keepdim=True)[0].squeeze(1)  # hila
                # pred_prob = pred_probabilities[0][target.item()].unsqueeze(0)
                prob_diff = (pred_prob - pred_org_prob).data.cpu().numpy()
                prob_diff_pertub[i, perturb_index:perturb_index + len(prob_diff)] = prob_diff

                # Logits Comparison
                pred_logit = out.logits.data.max(1, keepdim=True)[0].squeeze(1)  # hila
                # pred_logit = out.logits[0][target.item()].unsqueeze(0)
                logit_diff = (pred_logit - pred_org_logit).data.cpu().numpy()
                logit_diff_pertub[i, perturb_index:perturb_index + len(logit_diff)] = logit_diff
                if vit_config['verbose']:
                    print(
                        f'{100 * perturbation_steps[i]}% pixels blacked. Top Class: {out.logits[0].argmax(dim=0).item()}, Max logits: {round(out.logits[0].max(dim=0)[0].item(), 2)}, Max prob: {round(pred_probabilities[0].max(dim=0)[0].item(), 5)}; Correct class logit: {round(out.logits[0][target].item(), 2)} Correct class prob: {round(pred_probabilities[0][target].item(), 5)}')

                # Target-Class Comparison
                target_class = out.logits.data.max(1, keepdim=True)[1].squeeze(1)
                temp = (target == target_class).type(target.type()).data.cpu().numpy()
                num_correct_pertub[i, perturb_index:perturb_index + len(
                    temp)] = temp  # num_correct_pertub is matrix of each row represents perurbation step. Each column represents masked image

                probs_pertub = torch.softmax(out.logits, dim=1)
                target_probs = torch.gather(probs_pertub, 1, target[:, None])[:, 0]
                second_probs = probs_pertub.data.topk(2, dim=1)[0][:, 1]
                temp = torch.log(target_probs / second_probs).data.cpu().numpy()
                dissimilarity_pertub[i, perturb_index:perturb_index + len(temp)] = temp

            model_index += len(target)
            perturb_index += len(target)
    # print(f'Mean num_correct_perturbation: {np.mean(num_correct_pertub, axis=1)}')
    auc = get_auc(num_correct_pertub=num_correct_pertub, num_correct_model=num_correct_model)
    # save_objects(experiment_dir=experiment_dir, num_correct_model=num_correct_model,
    #              dissimilarity_model=dissimilarity_model, num_correct_pertub=num_correct_pertub,
    #              dissimilarity_pertub=dissimilarity_pertub, logit_diff_pertub=logit_diff_pertub,
    #              prob_diff_pertub=prob_diff_pertub, perturb_index=perturb_index,
    #              perturbation_steps=perturbation_steps)
    return auc


def get_auc(num_correct_pertub, num_correct_model):
    """
    num_correct_pertub is matrix of each row represents perurbation step. Each column represents masked image
    Each cell represents if the prediction at that step is the right prediction (0 / 1) and average of the images axis to
    get the number of average correct prediction at each perturbation step and then trapz integral (auc) to get final val
    """
    mean_accuracy_by_step = np.mean(num_correct_pertub, axis=1)
    # mean_accuracy_by_step = np.insert(mean_accuracy_by_step, 0,
    #                                   1)  # TODO - accuracy for class. Now its top-class (predicted)
    mean_accuracy_by_step = np.insert(mean_accuracy_by_step, 0, np.mean(num_correct_model))
    auc = calculate_auc(mean_accuracy_by_step=mean_accuracy_by_step) * 100
    # print(num_correct_pertub)
    # print(f'AUC: {round(auc, 4)}% for {num_correct_pertub.shape[1]} records')
    return auc


def save_objects(experiment_dir: Path, num_correct_model, dissimilarity_model, num_correct_pertub, dissimilarity_pertub,
                 logit_diff_pertub, prob_diff_pertub, perturb_index, perturbation_steps):
    np.save(Path(experiment_dir, 'model_hits.npy'), num_correct_model)
    np.save(Path(experiment_dir, 'model_dissimilarities.npy'), dissimilarity_model)
    np.save(Path(experiment_dir, 'perturbations_hits.npy'), num_correct_pertub[:, :perturb_index])
    np.save(Path(experiment_dir, 'perturbations_dissimilarities.npy'), dissimilarity_pertub[:, :perturb_index])
    np.save(Path(experiment_dir, 'perturbations_logit_diff.npy'), logit_diff_pertub[:, :perturb_index])
    np.save(Path(experiment_dir, 'perturbations_prob_diff.npy'), prob_diff_pertub[:, :perturb_index])
    # print(f'Mean num correct: {np.mean(num_correct_model)}, std num correct {np.std(num_correct_model)}')
    # print(f'Mean dissimilarity : {np.mean(dissimilarity_model)}, std dissimilarity {np.std(dissimilarity_model)}')
    # print(f'Perturbation Steps: {perturbation_steps}')
    # print(f'Mean num_correct_perturbation: {np.mean(num_correct_pertub, axis=1)}')  # , std num_correct_pertub {np.std(num_correct_pertub, axis=1)}')
    # print(
    #     f'Mean dissimilarity_pertub : {np.mean(dissimilarity_pertub, axis=1)}, std dissimilarity_pertub {np.std(dissimilarity_pertub, axis=1)}')


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


def get_model_infer_metrics(model_index, num_correct_model, pred, target):
    pred_probabilities = torch.softmax(pred, dim=1)
    pred_org_logit = pred.data.max(1, keepdim=True)[0].squeeze(1)
    pred_org_prob = pred_probabilities.data.max(1, keepdim=True)[0].squeeze(1)
    pred_class = pred.data.max(1, keepdim=True)[1].squeeze(1)
    tgt_pred = (target == pred_class).type(target.type()).data.cpu().numpy()
    num_correct_model[model_index:model_index + len(tgt_pred)] = tgt_pred
    return pred_probabilities, pred_org_logit, pred_org_prob, pred_class, tgt_pred, num_correct_model


def move_to_device_data_vis_and_target(data, target=None, vis=None):
    data = data.to(device)
    vis = vis.to(device)
    if target is not None:
        target = target.to(device)
        return data, target, vis
    return data, vis


def update_results_df(results_df: pd.DataFrame, vis_type: str, auc: float):
    return results_df.append({'vis_type': vis_type, 'auc': auc}, ignore_index=True)


import pickle


def save_obj_to_disk(path, obj) -> None:
    if type(path) == str and path[-4:] != '.pkl':
        path += '.pkl'

    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def save_best_auc_objects_to_disk(path, auc: float, vis, original_image, epoch_idx: int) -> None:
    object = {'auc': auc, 'vis': vis, 'original_image': original_image, 'epoch_idx': epoch_idx}
    save_obj_to_disk(path=path, obj=object)


def run_perturbation_test_opt(model, outputs, stage: str, epoch_idx: int, experiment_path=None):
    VIS_TYPES = [f'{stage}_vis_seg_cls_epoch_{epoch_idx}']

    if experiment_path is None:
        experiment_path = Path(EXPERIMENTS_FOLDER_PATH, vit_config['evaluation']['experiment_folder_name'])
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path, exist_ok=True)

    # model.to(device)
    model.eval()
    for vis_type in VIS_TYPES:
        # print(vis_type)
        vit_type_experiment_path = Path(experiment_path, vis_type)
        auc = eval_perturbation_test(experiment_dir=vit_type_experiment_path, model=model,
                                     outputs=outputs)
        return auc


def plot_image(image) -> None:  # [1,3,224,224] or [3,224,224]
    image = image if len(image.shape) == 3 else image.squeeze(0)
    plt.imshow(image.cpu().detach().permute(1, 2, 0))
    plt.show();


def run_perturbation_test(model, outputs, stage: str, epoch_idx: int, experiment_path):
    VIS_TYPES = [f'{stage}_vis_seg_cls_epoch_{epoch_idx}']

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path, exist_ok=True)
    output_csv_path = Path(experiment_path, f'{stage}_results_df.csv')
    if os.path.exists(output_csv_path):
        results_df = pd.read_csv(output_csv_path)
    else:
        results_df = pd.DataFrame(columns=['vis_type', 'auc'])
    model.to(device)
    model.eval()
    for vis_type in VIS_TYPES:
        print(vis_type)
        vit_type_experiment_path = Path(experiment_path, vis_type)
        # vit_type_experiment_path = create_folder(vit_type_experiment_path)
        auc = eval_perturbation_test(experiment_dir=vit_type_experiment_path, model=model,
                                     outputs=outputs)
        results_df = update_results_df(results_df=results_df, vis_type=vis_type, auc=auc)
        print(results_df)
        results_df.to_csv(output_csv_path, index=False)
        print(f"Saved results at: {output_csv_path}")
    return auc

# if __name__ == "__main__":
#     outputs = load_obj_from_path("/home/yuvalas/explainability/pickles/outputs.pkl")
#     _, model = load_feature_extractor_and_vit_model(vit_config=vit_config,
#                                                                     model_type='vit-basic',
#                                                                     is_wolf_transforms=vit_config['is_wolf_transforms'])
#     run_perturbation_test(model=model, outputs=outputs, stage='train', epoch_idx=0)
