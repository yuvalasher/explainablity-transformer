import os

from main.segmentation_eval.segmentation_utils import print_segmentation_results

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
from pathlib import Path

os.chdir('/home/amiteshel1/Projects/explainablity-transformer-cv/')

print('START !')
sys.path.append('/home/amiteshel1/Projects/explainablity-transformer-cv/')

from utils.saver import Saver

import yaml
from icecream import ic

from main.segmentation_eval.ViT_explanation_generator import LRP

import numpy as np
import torch
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from numpy import *
import argparse
from PIL import Image
import imageio

from tqdm import tqdm

from main.seg_classification.image_token_data_module_opt_segmentation import ImageSegOptDataModuleSegmentation
from utils.metrices import *

from config import config
from utils import render
from utils.iou import IoU

from data.imagenet import Imagenet_Segmentation, Imagenet_Segmentation_Loop

import matplotlib.pyplot as plt

import torch.nn.functional as F
from main.segmentation_eval.segmentation_model_opt import \
    OptImageClassificationWithTokenClassificationModel_Segmentation
from vit_loader.load_vit import load_vit_pretrained
from vit_utils import load_feature_extractor_and_vit_model, get_warmup_steps_and_total_training_steps, \
    get_loss_multipliers, freeze_multitask_model, get_checkpoint_idx

from utils.consts import (
    IMAGENET_VAL_IMAGES_FOLDER_PATH,
    IMAGENET_TEST_IMAGES_FOLDER_PATH,
    EXPERIMENTS_FOLDER_PATH,
    IMAGENET_TEST_IMAGES_ES_FOLDER_PATH,
)
from ViT_LRP import vit_base_patch16_224 as vit_LRP

import pytorch_lightning as pl
import gc
from PIL import ImageFile
import warnings
import logging

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger('checkpoint').setLevel(0)
logging.getLogger('lightning').setLevel(0)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
vit_config = config["vit"]
loss_config = vit_config["seg_cls"]["loss"]

CKPT_PATH = "/home/yuvalas/explainability/research/checkpoints/token_classification/model_google/vit-base-patch16-224_train_uni_True_val_unif_True_activation_sigmoid__norm_by_max_p_False_pred_1_mask_l_bce_50__train_n_samples_6000_lr_0.002_mlp_classifier_True__bs_32/None/checkpoints/epoch=27_val/epoch_auc=18.545.ckpt"
CHECKPOINT_EPOCH_IDX = get_checkpoint_idx(ckpt_path=CKPT_PATH)
RUN_BASE_MODEL = vit_config[
    'run_base_model']  # TODO If True, Running only forward of the image to create visualization of the base model


def compute_pred(output):
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

    T = pred.squeeze().cpu().numpy()
    T = np.expand_dims(T, 0)
    T = (T[:, np.newaxis] == np.arange(1000)) * 1.0
    T = torch.from_numpy(T).type(torch.FloatTensor)
    Tt = T.cuda()

    return Tt


def eval_results_per_res(Res, index, image=None, labels=None, q=-1):
    if args.save_img:
        save_original_image_and_gt_mask(image, index, labels)

    Res = (Res - Res.min()) / (Res.max() - Res.min())

    if q == -1:
        ret = Res.mean()
    else:
        ret = torch.quantile(Res, q=q)

    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())

    Res_1_AP = Res
    Res_0_AP = 1 - Res

    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0
    Res_1_AP[Res_1_AP != Res_1_AP] = 0
    Res_0_AP[Res_0_AP != Res_0_AP] = 0

    # TEST
    pred = Res.clamp(min=args.thr) / Res.max()
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()
    # print("target", target.shape)

    output = torch.cat((Res_0, Res_1), 1)
    output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)

    if args.save_img:
        # Save predicted mask
        mask = F.interpolate(Res_1, [64, 64], mode='bilinear')
        mask = mask[0].squeeze().data.cpu().numpy()
        # mask = Res_1[0].squeeze().data.cpu().numpy()
        mask = 255 * mask
        mask = mask.astype('uint8')
        imageio.imsave(os.path.join(args.exp_img_path, 'mask_' + str(index) + '.jpg'), mask)

        relevance = F.interpolate(Res, [64, 64], mode='bilinear')
        relevance = relevance[0].permute(1, 2, 0).data.cpu().numpy()
        # relevance = Res[0].permute(1, 2, 0).data.cpu().numpy()
        hm = np.sum(relevance, axis=-1)
        maps = (render.hm_to_rgb(hm, scaling=3, sigma=1, cmap='seismic') * 255).astype(np.uint8)
        imageio.imsave(os.path.join(args.exp_img_path, 'heatmap_' + str(index) + '.jpg'), maps)

    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
    batch_ap, batch_f1 = 0, 0

    # Segmentation resutls
    correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])  # labels should be [224,224]
    inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)
    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union
    # print("output", output.shape)
    # print("ap labels", labels.shape)
    # ap = np.nan_to_num(get_ap_scores(output, labels))
    ap = np.nan_to_num(get_ap_scores(output_AP, labels))
    f1 = np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0]))
    batch_ap += ap
    batch_f1 += f1

    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, pred, target


# hyperparameters
num_workers = 0


def save_original_image_and_gt_mask(image, index, labels):
    img = image[0].permute(1, 2, 0).data.cpu().numpy()
    img = 255 * (img - img.min()) / (img.max() - img.min())
    img = img.astype('uint8')
    os.makedirs(os.path.join(saver.results_dir, f'input/{index}'), exist_ok=True)
    Image.fromarray(img, 'RGB').save(os.path.join(saver.results_dir, f'input/{index}/{index}_input.png'))
    Image.fromarray((labels.repeat(3, 1, 1).permute(1, 2, 0).data.cpu().numpy() * 255).astype('uint8'),
                    'RGB').save(
        os.path.join(saver.results_dir, f'input/{index}/{index}_mask.png'))


def save_heatmap_and_seg_mask(Res, index, exp_name):
    Res = (Res - Res.min()) / (Res.max() - Res.min())
    ret = Res.mean()
    Res_1 = Res.gt(ret).type(Res.type())
    Res_1_AP = Res
    Res_1[Res_1 != Res_1] = 0
    # Save predicted mask
    mask = F.interpolate(Res_1, [64, 64], mode='bilinear')
    mask = mask[0].squeeze().data.cpu().numpy()
    # mask = Res_1[0].squeeze().data.cpu().numpy()
    mask = 255 * mask
    mask = mask.astype('uint8')
    imageio.imsave(os.path.join(saver.results_dir, f'input/{index}', f'{str(index)}_mask_{exp_name}.jpg'), mask)
    relevance = F.interpolate(Res, [64, 64], mode='bilinear')
    relevance = relevance[0].permute(1, 2, 0).data.cpu().numpy()
    # relevance = Res[0].permute(1, 2, 0).data.cpu().numpy()
    hm = np.sum(relevance, axis=-1)
    maps = (render.hm_to_rgb(hm, scaling=3, sigma=1, cmap='seismic') * 255).astype(np.uint8)
    imageio.imsave(os.path.join(saver.results_dir, f'input/{index}', f'{str(index)}_heatmap_{exp_name}.jpg'), maps)
    return


def calculate_metrics_segmentations(epochs, ds, q: int):
    total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    total_ap, total_f1 = [], []
    predictions, targets = [], []

    for batch_idx in tqdm(epochs, position=0, leave=True, total=len(epochs)):
        img, labels, image_resized = ds[batch_idx]

        Res = lrp.generate_LRP(image_resized.unsqueeze(0).cuda(), start_layer=1,
                               method="transformer_attribution").reshape(
            1, 1, 14, 14)

        Res = torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear').cuda()

        correct, labeled, inter, union, ap, f1, pred, target = eval_results_per_res(Res,
                                                                                    index=batch_idx,
                                                                                    q=q,
                                                                                    labels=labels,
                                                                                    image=image_resized)

        predictions.append(pred)
        targets.append(target)

        total_correct += correct.astype('int64')
        total_label += labeled.astype('int64')
        total_inter += inter.astype('int64')
        total_union += union.astype('int64')
        total_ap += [ap]
        total_f1 += [f1]
        pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
        IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
        mIoU = IoU.mean()
        mAp = np.mean(total_ap)
        mF1 = np.mean(total_f1)

    return mIoU, pixAcc, mAp, mF1


def plot_metric(q_arr, metric_a, metric_b, metrics_title, n_samples):
    plt.plot(q_arr, metric_a, label='ours')
    plt.plot(q_arr, metric_b, label='hila')
    plt.legend()
    plt.grid()
    plt.title(f'{metrics_title} - num_samples = {n_samples}')
    plt.savefig(
        f'/home/amiteshel1/Projects/explainablity-transformer-cv/amit_th_plots/{metrics_title}__{n_samples}.png')
    plt.close()


def init_get_normalize_and_trns():
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    test_img_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    test_img_trans_only_resize = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_lbl_trans = transforms.Compose([
        transforms.Resize((224, 224), Image.NEAREST),
    ])

    return test_img_trans, test_img_trans_only_resize, test_lbl_trans


if __name__ == '__main__':
    # Args
    parser = argparse.ArgumentParser(description='Training multi-class classifier')
    parser.add_argument('--arc', type=str, default='vgg', metavar='N',
                        help='Model architecture')
    parser.add_argument('--train_dataset', type=str, default='imagenet', metavar='N',
                        help='Testing Dataset')
    parser.add_argument('--method', type=str,
                        default='grad_rollout',
                        choices=['token_to_learn', 'rollout', 'lrp', 'transformer_attribution', 'full_lrp',
                                 'lrp_last_layer',
                                 'attn_last_layer', 'attn_gradcam'],
                        help='')
    parser.add_argument('--thr', type=float, default=0.,
                        help='threshold')

    parser.add_argument('--save-img', action='store_true',
                        default=False,
                        help='')

    parser.add_argument('--imagenet-seg-path', type=str, required=True)
    args = parser.parse_args()
    args.checkname = args.method + '_' + args.arc
    args.save_img = False

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # Define Saver
    saver = Saver(args)
    saver.results_dir = os.path.join(saver.experiment_dir, 'results')
    if not os.path.exists(saver.results_dir):
        os.makedirs(saver.results_dir)
    if not os.path.exists(os.path.join(saver.results_dir, 'input')):
        os.makedirs(os.path.join(saver.results_dir, 'input'))
    if not os.path.exists(os.path.join(saver.results_dir, 'explain')):
        os.makedirs(os.path.join(saver.results_dir, 'explain'))

    args.exp_img_path = os.path.join(saver.results_dir, 'explain/img')
    if not os.path.exists(args.exp_img_path):
        os.makedirs(args.exp_img_path)
    args.exp_np_path = os.path.join(saver.results_dir, 'explain/np')
    if not os.path.exists(args.exp_np_path):
        os.makedirs(args.exp_np_path)

    # Data
    batch_size = 1

    test_img_trans, test_img_trans_only_resize, test_lbl_trans = init_get_normalize_and_trns()
    ds = Imagenet_Segmentation(args.imagenet_seg_path,
                               batch_size=batch_size,
                               transform=test_img_trans,
                               transform_resize=test_img_trans_only_resize, target_transform=test_lbl_trans)

    seed_everything(config["general"]["seed"])
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    gc.collect()
    OBT_OBJECTS_PLOT_FOLDER_NAME = 'objects_png'
    OBT_OBJECTS_FOLDER_NAME = 'objects_pkl'

    loss_multipliers = get_loss_multipliers(loss_config=loss_config)
    BASE_AUC_OBJECTS_PATH = Path(EXPERIMENTS_FOLDER_PATH, vit_config['evaluation'][
        'experiment_folder_name'])  # /home/yuvalas/explainability/research/experiments/seg_cls/

    EXP_NAME = 'del_fixed_ft_50000_new_model_seg_only_base_new'  # TODO - pay attention !!!

    BEST_AUC_PLOT_PATH = Path(BASE_AUC_OBJECTS_PATH, EXP_NAME, 'opt_model', OBT_OBJECTS_PLOT_FOLDER_NAME)
    BEST_AUC_OBJECTS_PATH = Path(BASE_AUC_OBJECTS_PATH, EXP_NAME, 'opt_model', OBT_OBJECTS_FOLDER_NAME)

    os.makedirs(BEST_AUC_PLOT_PATH, exist_ok=True)
    os.makedirs(BEST_AUC_OBJECTS_PATH, exist_ok=True)

    BASE_MODEL_BEST_AUC_PLOT_PATH = Path(BASE_AUC_OBJECTS_PATH, EXP_NAME, 'base_model', OBT_OBJECTS_PLOT_FOLDER_NAME)
    BASE_MODEL_BEST_AUC_OBJECTS_PATH = Path(BASE_AUC_OBJECTS_PATH, EXP_NAME, 'base_model', OBT_OBJECTS_FOLDER_NAME)
    os.makedirs(BASE_MODEL_BEST_AUC_PLOT_PATH, exist_ok=True)
    os.makedirs(BASE_MODEL_BEST_AUC_OBJECTS_PATH, exist_ok=True)

    feature_extractor, _ = load_feature_extractor_and_vit_model(
        vit_config=vit_config,
        model_type="vit-basic",
        is_wolf_transforms=vit_config["is_wolf_transforms"],
    )

    vit_for_classification_image, vit_for_patch_classification = load_vit_pretrained(
        model_name=vit_config["model_name"])

    warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(
        n_epochs=vit_config["n_epochs"],
        train_samples_length=len(list(Path(IMAGENET_TEST_IMAGES_FOLDER_PATH).iterdir())),
        batch_size=vit_config["batch_size"],
    )

    metric = IoU(2, ignore_index=-1)

    model = OptImageClassificationWithTokenClassificationModel_Segmentation(
        vit_for_classification_image=vit_for_classification_image,
        vit_for_patch_classification=vit_for_patch_classification,
        feature_extractor=feature_extractor,
        plot_path='',
        warmup_steps=warmup_steps,
        total_training_steps=total_training_steps,
        batch_size=batch_size,
        best_auc_objects_path=BASE_MODEL_BEST_AUC_OBJECTS_PATH if RUN_BASE_MODEL else BEST_AUC_OBJECTS_PATH,
        checkpoint_epoch_idx=CHECKPOINT_EPOCH_IDX,
        best_auc_plot_path='',
        run_base_model_only=RUN_BASE_MODEL,
        model_runtype='train',
        experiment_path='exp_name_amitt'  # choose 'train' or 'test'
    )

    model = freeze_multitask_model(
        model=model,
        freezing_classification_transformer=vit_config["freezing_classification_transformer"],
        segmentation_transformer_n_first_layers_to_freeze=vit_config[
            "segmentation_transformer_n_first_layers_to_freeze"]
    )
    total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    total_ap, total_f1 = [], []
    predictions, targets = [], []

    epochs = range(len(ds))[:2]
    for batch_idx in tqdm(epochs, leave=True, position=0):
        ds_loop = Imagenet_Segmentation_Loop(*ds[batch_idx])
        dl = DataLoader(ds_loop, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
        data_module = ImageSegOptDataModuleSegmentation(
            train_data_loader=dl
        )
        trainer = pl.Trainer(
            logger=[],
            accelerator='gpu',
            gpus=1,
            devices=[1, 2],
            num_sanity_val_steps=0,
            check_val_every_n_epoch=100,
            max_epochs=CHECKPOINT_EPOCH_IDX + 25,
            resume_from_checkpoint=CKPT_PATH,
            enable_progress_bar=False,
            enable_checkpointing=False,
            default_root_dir=vit_config["default_root_dir"],
            weights_summary=None
        )

        trainer.fit(model=model, datamodule=data_module)

        image_resized = model.image_resized
        Res = model.best_auc_vis
        labels = model.target
        correct, labeled, inter, union, ap, f1, pred, target = eval_results_per_res(Res, labels=labels, index=batch_idx,
                                                                                    image=image_resized)

        predictions.append(pred)
        targets.append(target)

        total_correct += correct.astype('int64')
        total_label += labeled.astype('int64')
        total_inter += inter.astype('int64')
        total_union += union.astype('int64')
        total_ap += [ap]
        total_f1 += [f1]
        pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
        IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
        mIoU = IoU.mean()
        mAp = np.mean(total_ap)
        mF1 = np.mean(total_f1)
        if (batch_idx % 100 == 0) or (batch_idx == epochs[-1]):
            print_segmentation_results(pixAcc=pixAcc, mAp=mAp, mIoU=mIoU, mF1=mF1)

    print("FINISH !!!!!")
    print_segmentation_results(pixAcc=pixAcc, mAp=mAp, mIoU=mIoU, mF1=mF1)
