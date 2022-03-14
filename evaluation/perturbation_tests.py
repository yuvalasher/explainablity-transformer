import glob
from typing import Union

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


# def eval(args):
def eval():
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
    auc_pertub = np.zeros((9, len(imagenet_ds)))
    logit_diff_pertub = np.zeros((9, len(imagenet_ds)))
    prob_diff_pertub = np.zeros((9, len(imagenet_ds)))
    perturb_index = 0

    for batch_idx, (data, vis, target) in enumerate(tqdm(sample_loader)):
        # Update the number of samples
        num_samples += len(data)

        data, target, vis = get_data_vis_and_target(data, target, vis)
        # norm_data = normalize(data.clone()) # TODO - verify

        # Compute model accuracy

        plot_image(data)
        inputs = feature_extractor(images=data.squeeze(0), return_tensors="pt")
        pred = model(**inputs)
        pred_probabilities, pred_org_logit, pred_org_prob, pred_class, tgt_pred, num_correct_model = get_model_infer_metrics(
            model_index, num_correct_model, pred.logits, target)

        probs = torch.softmax(pred.logits, dim=1)
        target_probs = torch.gather(probs, 1, target[:, None])[:, 0]
        second_probs = probs.data.topk(2, dim=1)[0][:, 1]
        temp = torch.log(target_probs / second_probs).data.cpu().numpy()
        dissimilarity_model[model_index:model_index + len(temp)] = temp

        """
        if args.wrong:
            wid = np.argwhere(tgt_pred == 0).flatten()
            if len(wid) == 0:
                continue
            wid = torch.from_numpy(wid).to(vis.device)
            vis = vis.index_select(0, wid)
            data = data.index_select(0, wid)
            target = target.index_select(0, wid)
        """
        # Save original shape
        org_shape = data.shape

        # if args.neg:
        #     vis = -vis

        vis = vis.reshape(org_shape[0], -1)

        for i in range(len(perturbation_steps)):
            _data = data.clone()
            _data = get_perturbated_data(vis=vis, image=_data, perturbation_step=perturbation_steps[i],
                                         base_size=base_size)
            if i in [1, 2, 3, 4]:
                plot_image(_data)
            inputs = feature_extractor(images=_data.squeeze(0), return_tensors="pt")
            out = model(**inputs)

            # Probabilities Comparison
            pred_probabilities = torch.softmax(out.logits, dim=1)
            pred_prob = pred_probabilities.data.max(1, keepdim=True)[0].squeeze(1)  # hila
            # pred_prob = pred_probabilities[0][target.item()].unsqueeze(0)
            prob_diff = (pred_prob - pred_org_prob).data.cpu().numpy()
            print(
                f'%: {perturbation_steps[i]}, original_prob: {pred_org_prob.item()}, perb_prob: {pred_prob.item()}, diff: {prob_diff.item()}')
            prob_diff_pertub[i, perturb_index:perturb_index + len(prob_diff)] = prob_diff

            # Logits Comparison
            pred_logit = out.logits.data.max(1, keepdim=True)[0].squeeze(1)  # hila
            # pred_logit = out.logits[0][target.item()].unsqueeze(0)
            logit_diff = (pred_logit - pred_org_logit).data.cpu().numpy()
            logit_diff_pertub[i, perturb_index:perturb_index + len(logit_diff)] = logit_diff

            # AUC - Area Under Curve
            y_true_one_hot = torch.nn.functional.one_hot(target, num_classes=1000)
            y_pred_probabilities = pred_probabilities
            auc = calculate_auc(y_true=y_true_one_hot[0], y_pred=y_pred_probabilities[0])
            auc_pertub[i, perturb_index:perturb_index + len(logit_diff)] = auc
            print(f'Auc: {auc}')

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
    print(1)
    """
    np.save(os.path.join(experiment_dir, 'model_hits.npy'), num_correct_model)
    np.save(os.path.join(experiment_dir, 'model_dissimilarities.npy'), dissimilarity_model)
    np.save(os.path.join(experiment_dir, 'perturbations_hits.npy'), num_correct_pertub[:, :perturb_index])
    np.save(os.path.join(experiment_dir, 'perturbations_dissimilarities.npy'), dissimilarity_pertub[:, :perturb_index])
    np.save(os.path.join(experiment_dir, 'perturbations_logit_diff.npy'), logit_diff_pertub[:, :perturb_index])
    np.save(os.path.join(experiment_dir, 'perturbations_prob_diff.npy'), prob_diff_pertub[:, :perturb_index])
    """
    print(f'Mean num correct: {np.mean(num_correct_model)}, std num correct {np.std(num_correct_model)}')
    print(f'Mean dissimilarity : {np.mean(dissimilarity_model)}, std dissimilarity {np.std(dissimilarity_model)}')
    print(f'Perturbation Steps: {perturbation_steps}')
    print(f'Mean auc_pertub: {np.mean(auc_pertub)}, std auc_pertub {np.std(auc_pertub)}')
    print(
        f'Mean num_correct_pertub: {np.mean(num_correct_pertub, axis=1)}, std num_correct_pertub {np.std(num_correct_pertub, axis=1)}')
    print(
        f'Mean dissimilarity_pertub : {np.mean(dissimilarity_pertub, axis=1)}, std dissimilarity_pertub {np.std(dissimilarity_pertub, axis=1)}')


def plot_image(data) -> None:
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


if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description='Train a segmentation')
    parser.add_argument('--batch-size', type=int,
                        default=16,
                        help='')
    parser.add_argument('--neg', type=bool,
                        default=True,
                        help='')
    parser.add_argument('--value', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--scale', type=str,
                        default='per',
                        choices=['per', '100'],
                        help='')
    parser.add_argument('--method', type=str,
                        default='grad_rollout',
                        choices=['rollout', 'lrp', 'transformer_attribution', 'full_lrp', 'v_gradcam', 'lrp_last_layer',
                                 'lrp_second_layer', 'gradcam',
                                 'attn_last_layer', 'attn_gradcam', 'input_grads'],
                        help='')
    parser.add_argument('--vis-class', type=str,
                        default='top',
                        choices=['top', 'target', 'index'],
                        help='')
    parser.add_argument('--wrong', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--class-id', type=int,
                        default=0,
                        help='')
    parser.add_argument('--is-ablation', type=bool,
                        default=False,
                        help='')
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    # PATH variables
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    dataset = PATH + 'dataset/'
    os.makedirs(os.path.join(PATH, 'experiments'), exist_ok=True)
    os.makedirs(os.path.join(PATH, 'experiments/perturbations'), exist_ok=True)

    exp_name = args.method
    exp_name += '_neg' if args.neg else '_pos'
    print(exp_name)

    if args.vis_class == 'index':
        args.runs_dir = os.path.join(PATH, 'experiments/perturbations/{}/{}_{}'.format(exp_name,
                                                                                       args.vis_class,
                                                                                       args.class_id))
    else:
        ablation_fold = 'ablation' if args.is_ablation else 'not_ablation'
        args.runs_dir = os.path.join(PATH, 'experiments/perturbations/{}/{}/{}'.format(exp_name,
                                                                                       args.vis_class, ablation_fold))
        # args.runs_dir = os.path.join(PATH, 'experiments/perturbations/{}/{}'.format(exp_name,
        #                                                                             args.vis_class))

    if args.wrong:
        args.runs_dir += '_wrong'

    experiments = sorted(glob.glob(os.path.join(args.runs_dir, 'experiment_*')))
    experiment_id = int(experiments[-1].split('_')[-1]) + 1 if experiments else 0
    args.experiment_dir = os.path.join(args.runs_dir, 'experiment_{}'.format(str(experiment_id)))
    os.makedirs(args.experiment_dir, exist_ok=True)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    if args.vis_class == 'index':
        vis_method_dir = os.path.join(PATH, 'visualizations/{}/{}_{}'.format(args.method,
                                                                             args.vis_class,
                                                                             args.class_id))
    else:
        ablation_fold = 'ablation' if args.is_ablation else 'not_ablation'
        vis_method_dir = os.path.join(PATH, 'visualizations/{}/{}/{}'.format(args.method,
                                                                             args.vis_class, ablation_fold))
        # vis_method_dir = os.path.join(PATH, 'visualizations/{}/{}'.format(args.method,
        #                                                                      args.vis_class))

    # imagenet_ds = ImagenetResults('visualizations/{}'.format(args.method))
    imagenet_ds = ImagenetResults(vis_method_dir)

    # Model
    model = vit_base_patch16_224(pretrained=True).cuda()
    model.eval()

    save_path = PATH + 'results/'

    sample_loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False)

    eval(args)
"""
    experiment_path = Path(EXPERIMENTS_FOLDER_PATH, 'test')
    results_path = Path(experiment_path, 'results.hdf5')
    _ = create_folder(experiment_path)

    feature_extractor, model = load_feature_extractor_and_vit_model(vit_config=vit_config, model_type='vit-for-dino')
    model.eval()
    # torch.multiprocessing.set_start_method('spawn')
    imagenet_ds = ImagenetResults(path=results_path)
    sample_loader = DataLoader(imagenet_ds, batch_size=evaluation_config['batch_size'], shuffle=False)
    eval()
