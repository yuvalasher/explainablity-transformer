import glob
from typing import Union
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.imagenet_results_dataset import ImagenetResults
from evaluation.evaluation_utils import normalize, calculate_auc
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


def eval(experiment_dir: Path, model, feature_extractor) -> float:
    num_samples = 0
    num_correct_model = np.zeros((len(imagenet_ds, )))
    dissimilarity_model = np.zeros((len(imagenet_ds, )))
    model_index = 0

    # if args.scale == 'per':
    base_size = 224 * 224
    perturbation_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # elif args.scale == '100':  # TODO - what is it mean?
    #     base_size = 100
    #     perturbation_steps = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    # else:
    #     raise Exception('scale not valid')

    num_correct_pertub = np.zeros((9, len(imagenet_ds)))  # 9 is the number of perturbation steps
    dissimilarity_pertub = np.zeros((9, len(imagenet_ds)))
    logit_diff_pertub = np.zeros((9, len(imagenet_ds)))
    prob_diff_pertub = np.zeros((9, len(imagenet_ds)))
    perturb_index = 0

    for batch_idx, (data, vis, target) in enumerate(tqdm(sample_loader)):
        num_samples += len(data)

        data, target, vis = get_data_vis_and_target(data, target, vis)

        # Compute model accuracy
        if vit_config['verbose']:
            plot_image(data)
        inputs = feature_extractor(images=data.squeeze(0).cpu(), return_tensors="pt")
        inputs = {'pixel_values': inputs['pixel_values'].to(device)}
        pred = model(**inputs)
        probs = torch.softmax(pred.logits, dim=1)
        target_probs = torch.gather(probs, 1, target[:, None])[:, 0]
        second_probs = probs.data.topk(2, dim=1)[0][:, 1]
        temp = torch.log(target_probs / second_probs).data.cpu().numpy()
        dissimilarity_model[model_index:model_index + len(temp)] = temp
        target = torch.tensor([torch.argmax(probs).item()]).to(device)
        pred_probabilities, pred_org_logit, pred_org_prob, pred_class, tgt_pred, num_correct_model = get_model_infer_metrics(
            model_index=model_index, num_correct_model=num_correct_model, pred=pred.logits, target=target)
        if vit_config['verbose']:
            print(
                f'\nOriginal Image. Top Class: {pred.logits[0].argmax(dim=0).item()}, Max logits: {round(pred.logits[0].max(dim=0)[0].item(), 2)}, Max prob: {round(probs[0].max(dim=0)[0].item(), 5)}; Correct class logit: {round(pred.logits[0][target].item(), 2)} Correct class prob: {round(probs[0][target].item(), 5)}')

        # Save original shape
        org_shape = data.shape

        # if args.neg:
        #     vis = -vis

        vis = vis.reshape(org_shape[0], -1)

        for i in range(len(perturbation_steps)):
            _data = data.clone()
            _data = get_perturbated_data(vis=vis, image=_data, perturbation_step=perturbation_steps[i],
                                         base_size=base_size)
            if vit_config['verbose']:
                plot_image(_data, step_idx=i)
            inputs = feature_extractor(images=_data.squeeze(0).cpu(), return_tensors="pt")
            inputs = {'pixel_values': inputs['pixel_values'].to(device)}
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
            num_correct_pertub[i, perturb_index:perturb_index + len(temp)] = temp

            probs_pertub = torch.softmax(out.logits, dim=1)
            target_probs = torch.gather(probs_pertub, 1, target[:, None])[:, 0]
            second_probs = probs_pertub.data.topk(2, dim=1)[0][:, 1]
            temp = torch.log(target_probs / second_probs).data.cpu().numpy()
            dissimilarity_pertub[i, perturb_index:perturb_index + len(temp)] = temp

        model_index += len(target)
        perturb_index += len(target)
    auc = get_auc(num_correct_pertub=num_correct_pertub)
    save_objects(experiment_dir=experiment_dir, num_correct_model=num_correct_model,
                 dissimilarity_model=dissimilarity_model, num_correct_pertub=num_correct_pertub,
                 dissimilarity_pertub=dissimilarity_pertub, logit_diff_pertub=logit_diff_pertub,
                 prob_diff_pertub=prob_diff_pertub, perturb_index=perturb_index,
                 perturbation_steps=perturbation_steps)
    return auc


def get_auc(num_correct_pertub):
    mean_accuracy_by_step = np.mean(num_correct_pertub, axis=1)
    mean_accuracy_by_step = np.insert(mean_accuracy_by_step, 0, 1)  # TODO - accuracy for class = top-class (predicted)
    auc = calculate_auc(mean_accuracy_by_step=mean_accuracy_by_step) * 100
    print(f'AUC: {auc}%')
    return auc


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


def get_model_infer_metrics(model_index, num_correct_model, pred, target):
    pred_probabilities = torch.softmax(pred, dim=1)
    pred_org_logit = pred.data.max(1, keepdim=True)[0].squeeze(1)
    pred_org_prob = pred_probabilities.data.max(1, keepdim=True)[0].squeeze(1)
    pred_class = pred.data.max(1, keepdim=True)[1].squeeze(1)
    tgt_pred = (target == pred_class).type(target.type()).data.cpu().numpy()
    num_correct_model[model_index:model_index + len(tgt_pred)] = tgt_pred
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
    VIS_TYPES = ['max_logits', 'min_pred_loss', 'iter_90', 'iter_100', 'iter_110', 'iter_120', 'iter_130', 'iter_140',
                 'iter_150', 'iter_160', 'iter_165', 'iter_170', 'iter_175', 'iter_180', 'iter_185', 'iter_190', ]

    experiment_path = Path(EXPERIMENTS_FOLDER_PATH, vit_config['evaluation']['experiment_folder_name'])
    # experiment_path = Path(EXPERIMENTS_FOLDER_PATH, 'test')
    for vis_type in VIS_TYPES:
        vit_type_experiment_path = create_folder(Path(experiment_path, vis_type))
        print(vis_type)
        feature_extractor, model = load_feature_extractor_and_vit_model(vit_config=vit_config,
                                                                        model_type='vit-for-dino')
        model.to(device)
        model.eval()
        imagenet_ds = ImagenetResults(path=experiment_path, vis_type=vis_type)
        sample_loader = DataLoader(imagenet_ds, batch_size=evaluation_config['batch_size'], shuffle=False)
        auc = eval(experiment_dir=vit_type_experiment_path, model=model, feature_extractor=feature_extractor)
        results_df = update_results_df(results_df=results_df, vis_type=vis_type, auc=auc)
        print(results_df)
        results_df.to_csv(Path(experiment_path, 'results_df.csv'), index=False)
