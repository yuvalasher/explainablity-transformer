import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
from pathlib import Path

from main.segmentation_eval.segmentation_model_opt import \
    OptImageClassificationWithTokenClassificationModel_Segmentation

# os.chdir('/home/amiteshel1/Projects/explainablity-transformer-cv/')
# sys.path.append('/home/amiteshel1/Projects/explainablity-transformer-cv/')


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

from data.imagenet import Imagenet_Segmentation

import matplotlib.pyplot as plt

import torch.nn.functional as F

from vit_loader.load_vit import load_vit_pretrained
from vit_utils import load_feature_extractor_and_vit_model, get_warmup_steps_and_total_training_steps, \
    get_loss_multipliers, freeze_multitask_model

from utils.consts import IMAGENET_TEST_IMAGES_FOLDER_PATH

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
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    IMAGENET_SEG_PATH = '/home/amiteshel1/Projects/explainablity-transformer-cv/datasets/gtsegs_ijcv.mat'
    # Data
    batch_size = 32
    test_img_trans, test_img_trans_only_resize, test_lbl_trans = init_get_normalize_and_trns()
    ds = Imagenet_Segmentation(IMAGENET_SEG_PATH,
                               batch_size=batch_size,
                               transform=test_img_trans,
                               transform_resize=test_img_trans_only_resize, target_transform=test_lbl_trans)

    vit_config = config["vit"]
    loss_config = vit_config["seg_cls"]["loss"]
    seed_everything(config["general"]["seed"])
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    gc.collect()
    loss_multipliers = get_loss_multipliers(loss_config=loss_config)

    CKPT_PATH = "/home/yuvalas/explainability/research/checkpoints/token_classification/model_google/vit-base-patch16-224_train_uni_True_val_unif_True_activation_sigmoid__norm_by_max_p_False_pred_1_mask_l_bce_50__train_n_samples_6000_lr_0.002_mlp_classifier_True__bs_32/None/checkpoints/epoch=27_val/epoch_auc=18.545.ckpt"
    CHECKPOINT_EPOCH_IDX = 28  # TODO - pay attention !!!
    RUN_BASE_MODEL = vit_config[
        'run_base_model']  # TODO If True, Running only forward of the image to create visualization of the base model

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
        best_auc_objects_path='',
        checkpoint_epoch_idx=CHECKPOINT_EPOCH_IDX,
        best_auc_plot_path='',
        run_base_model_only=RUN_BASE_MODEL,
        model_runtype='test',
        experiment_path='exp_name_amitt'
    )
    model = freeze_multitask_model(
        model=model,
        freezing_classification_transformer=vit_config["freezing_classification_transformer"],
        segmentation_transformer_n_first_layers_to_freeze=vit_config[
            "segmentation_transformer_n_first_layers_to_freeze"]
    )

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
    data_module = ImageSegOptDataModuleSegmentation(train_data_loader=dl)
    trainer = pl.Trainer(
        logger=[],
        accelerator='gpu',
        gpus=1,
        devices=1,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=100,
        max_epochs=vit_config["n_epochs"],
        resume_from_checkpoint=CKPT_PATH,
        enable_progress_bar=True,
        enable_checkpointing=False,
        default_root_dir=vit_config["default_root_dir"],
        weights_summary=None
    )
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)

    mIoU, pixAcc, mAp, mF1 = model.seg_results['mIoU'], model.seg_results['pixAcc'], model.seg_results['mAp'], \
                             model.seg_results['mF1']
    print(f"Pixel-wise Accuracy: {round(pixAcc * 100, 4)}")
    print(f"Mean AP over {2} classes: {round(mAp * 100, 4)}")
    print(f"Mean IoU over {2} classes: {round(mIoU * 100, 4)}")
    print(f"Mean F1 over {2} classes:{round(mF1 * 100, 4)}")
