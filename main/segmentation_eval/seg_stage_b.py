import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
from pathlib import Path

from models.modeling_vit_patch_classification import ViTForMaskGeneration
from utils.saver import Saver
from icecream import ic
import numpy as np
import torch
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from numpy import *
import argparse
from PIL import Image
import imageio
from transformers import ViTForImageClassification

from tqdm import tqdm

from main.seg_classification.image_token_data_module_opt_segmentation import ImageSegOptDataModuleSegmentation
from utils.metrices import *

from config import config
from utils import render
from utils.iou import IoU
from data.imagenet import Imagenet_Segmentation, Imagenet_Segmentation_Loop
import torch.nn.functional as F
from main.segmentation_eval.segmentation_model_opt import \
    OptImageClassificationWithTokenClassificationModel_Segmentation
from vit_loader.load_vit import load_vit_pretrained
from vit_utils import load_feature_extractor_and_vit_model, get_warmup_steps_and_total_training_steps, \
    get_loss_multipliers, freeze_multitask_model, get_checkpoint_idx

from utils.consts import (
    IMAGENET_VAL_IMAGES_FOLDER_PATH,
    EXPERIMENTS_FOLDER_PATH,
    IMAGENET_SEG_PATH,
)

from main.seg_classification.vit_backbone_to_details import VIT_BACKBONE_DETAILS
from main.segmentation_eval.segmentation_utils import print_segmentation_results

import pytorch_lightning as pl
import gc
from PIL import ImageFile
import warnings
import logging

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger('checkpoint').setLevel(0)
logging.getLogger('lightning').setLevel(0)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
vit_config = config["vit"]
loss_config = vit_config["seg_cls"]["loss"]
vit_config["enable_checkpointing"] = False
vit_config["train_model_by_target_gt_class"] = False
num_workers = 0

target_or_predicted_model = "predicted"
CKPT_PATH, IMG_SIZE, PATCH_SIZE, MASK_LOSS_MUL = VIT_BACKBONE_DETAILS[vit_config["model_name"]]["ckpt_path"][
                                                     target_or_predicted_model], \
                                                 VIT_BACKBONE_DETAILS[vit_config["model_name"]][
                                                     "img_size"], VIT_BACKBONE_DETAILS[vit_config["model_name"]][
                                                     "patch_size"], VIT_BACKBONE_DETAILS[vit_config["model_name"]][
                                                     "mask_loss"]
CHECKPOINT_EPOCH_IDX = get_checkpoint_idx(ckpt_path=CKPT_PATH)
vit_config["img_size"] = IMG_SIZE
loss_config["mask_loss_mul"] = MASK_LOSS_MUL
vit_config["patch_size"] = PATCH_SIZE
ic(CKPT_PATH)
ic(loss_config["mask_loss_mul"], IMG_SIZE)
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
    pred = Res.clamp(min=0.0) / Res.max()
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()

    output = torch.cat((Res_0, Res_1), 1)
    output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)

    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
    batch_ap, batch_f1 = 0, 0

    # Segmentation results
    correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])  # labels should be [224,224]
    inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)
    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union

    ap = np.nan_to_num(get_ap_scores(output_AP, labels))
    f1 = np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0]))
    batch_ap += ap
    batch_f1 += f1

    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, pred, target


def save_original_image_and_gt_mask(image, labels, plot_path):
    img = image[0].permute(1, 2, 0).data.cpu().numpy()
    img = 255 * (img - img.min()) / (img.max() - img.min())
    img = img.astype('uint8')
    Image.fromarray(img, 'RGB').save(Path(plot_path, 'input.jpg'))
    Image.fromarray((labels.repeat(3, 1, 1).permute(1, 2, 0).data.cpu().numpy() * 255).astype('uint8'),
                    'RGB').save(Path(plot_path, 'gt.jpg'))


def save_heatmap_and_seg_mask(Res, plot_path):
    Res_cloned = Res.clone()
    Res = (Res_cloned - Res_cloned.min()) / (Res_cloned.max() - Res_cloned.min())
    ret = Res_cloned.mean()
    Res_1 = Res_cloned.gt(ret).type(Res_cloned.type())
    Res_1_AP = Res_cloned
    Res_1[Res_1 != Res_1] = 0
    # Save predicted mask
    mask = F.interpolate(Res_1, [64, 64], mode='bilinear')
    mask = mask[0].squeeze().data.cpu().numpy()
    # mask = Res_1[0].squeeze().data.cpu().numpy()
    mask = 255 * mask
    mask = mask.astype('uint8')
    imageio.imsave(os.path.join(plot_path, 'mask.jpg'), mask)
    relevance = F.interpolate(Res_cloned, [64, 64], mode='bilinear')
    relevance = relevance[0].permute(1, 2, 0).data.cpu().numpy()
    # relevance = Res[0].permute(1, 2, 0).data.cpu().numpy()
    hm = np.sum(relevance, axis=-1)
    maps = (render.hm_to_rgb(hm, scaling=3, sigma=1, cmap='seismic') * 255).astype(np.uint8)
    imageio.imsave(os.path.join(plot_path, 'heatmap.jpg'), maps)
    return


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
    ic(vit_config["segmentation_transformer_n_first_layers_to_freeze"])
    ic(vit_config["n_epochs_to_optimize_stage_b"])
    ic(loss_config["use_logits_only"])
    ic(vit_config['run_base_model'])
    batch_size = 1

    test_img_trans, test_img_trans_only_resize, test_lbl_trans = init_get_normalize_and_trns()
    ds = Imagenet_Segmentation(IMAGENET_SEG_PATH,
                               batch_size=batch_size,
                               transform=test_img_trans,
                               transform_resize=test_img_trans_only_resize, target_transform=test_lbl_trans)

    seed_everything(config["general"]["seed"])
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    gc.collect()

    loss_multipliers = get_loss_multipliers(loss_config=loss_config)
    BASE_AUC_OBJECTS_PATH = Path(EXPERIMENTS_FOLDER_PATH, vit_config['evaluation'][
        'experiment_folder_name'])

    EXP_NAME = ''  # TODO

    feature_extractor, _ = load_feature_extractor_and_vit_model(
        vit_config=vit_config,
        model_type="vit-basic",
        is_competitive_method_transforms=vit_config["is_competitive_method_transforms"],
    )

    ic(vit_config["model_name"])
    if vit_config["model_name"] in ["google/vit-base-patch16-224"]:
        vit_for_classification_image, vit_for_patch_classification = load_vit_pretrained(
            model_name=vit_config["model_name"])
    else:
        vit_for_classification_image = ViTForImageClassification.from_pretrained(vit_config["model_name"])
        vit_for_patch_classification = ViTForMaskGeneration.from_pretrained(vit_config["model_name"])

    warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(
        n_epochs=vit_config["n_epochs_to_optimize_stage_b"],
        train_samples_length=len(list(Path(IMAGENET_VAL_IMAGES_FOLDER_PATH).iterdir())),
        batch_size=vit_config["batch_size"],
    )

    metric = IoU(num_classes=2, ignore_index=-1)

    model = OptImageClassificationWithTokenClassificationModel_Segmentation(
        vit_for_classification_image=vit_for_classification_image,
        vit_for_patch_classification=vit_for_patch_classification,
        feature_extractor=feature_extractor,
        plot_path='',
        warmup_steps=warmup_steps,
        total_training_steps=total_training_steps,
        batch_size=batch_size,
        best_auc_objects_path=Path(""),
        checkpoint_epoch_idx=CHECKPOINT_EPOCH_IDX,
        best_auc_plot_path='',
        run_base_model_only=RUN_BASE_MODEL,
        model_runtype='train',
        experiment_path='exp_name'
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
    random.seed(config["general"]["seed"])
    n_batches = range(len(ds))
    for batch_idx in tqdm(n_batches, leave=True, position=0):
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
            max_epochs=CHECKPOINT_EPOCH_IDX + vit_config["n_epochs_to_optimize_stage_b"],
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
        correct, labeled, inter, union, ap, f1, pred, target = eval_results_per_res(Res=Res,
                                                                                    labels=labels,
                                                                                    index=batch_idx,
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
        if (batch_idx % 100 == 0) or (batch_idx == n_batches[-1]):
            print_segmentation_results(pixAcc=pixAcc, mAp=mAp, mIoU=mIoU, mF1=mF1)

    print_segmentation_results(pixAcc=pixAcc, mAp=mAp, mIoU=mIoU, mF1=mF1)
