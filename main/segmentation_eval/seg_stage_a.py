import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from icecream import ic
from transformers import ViTForImageClassification
from feature_extractor import ViTFeatureExtractor
from main.seg_classification.vit_backbone_to_details import VIT_BACKBONE_DETAILS
from main.segmentation_eval.segmentation_utils import print_segmentation_results
from models.modeling_vit_patch_classification import ViTForMaskGeneration
from pathlib import Path
from main.segmentation_eval.segmentation_model_opt import \
    OptImageClassificationWithTokenClassificationModel_Segmentation

import torch
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from numpy import *
from PIL import Image
from main.seg_classification.image_token_data_module_opt_segmentation import ImageSegOptDataModuleSegmentation
from utils.metrices import *
from config import config
from utils.iou import IoU

from data.imagenet import Imagenet_Segmentation
import torch.nn.functional as F
from vit_loader.load_vit import load_vit_pretrained
from vit_utils import get_warmup_steps_and_total_training_steps, \
    get_loss_multipliers, freeze_multitask_model, get_checkpoint_idx

from utils.consts import IMAGENET_SEG_PATH, IMAGENET_VAL_IMAGES_FOLDER_PATH

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
    batch_size = 32
    test_img_trans, test_img_trans_only_resize, test_lbl_trans = init_get_normalize_and_trns()
    ds = Imagenet_Segmentation(IMAGENET_SEG_PATH,
                               batch_size=batch_size,
                               transform=test_img_trans,
                               transform_resize=test_img_trans_only_resize, target_transform=test_lbl_trans)

    vit_config = config["vit"]
    loss_config = vit_config["seg_cls"]["loss"]
    vit_config["train_model_by_target_gt_class"] = False
    vit_config["enable_checkpointing"] = False
    seed_everything(config["general"]["seed"])
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    gc.collect()
    loss_multipliers = get_loss_multipliers(loss_config=loss_config)
    target_or_predicted_model = "predicted"
    CKPT_PATH, IMG_SIZE, PATCH_SIZE, MASK_LOSS_MUL = VIT_BACKBONE_DETAILS[vit_config["model_name"]]["ckpt_path"][
                                                         target_or_predicted_model], \
                                                     VIT_BACKBONE_DETAILS[vit_config["model_name"]][
                                                         "img_size"], VIT_BACKBONE_DETAILS[vit_config["model_name"]][
                                                         "patch_size"], VIT_BACKBONE_DETAILS[vit_config["model_name"]][
                                                         "mask_loss"]
    ic(CKPT_PATH)
    vit_config["img_size"] = IMG_SIZE
    vit_config["patch_size"] = PATCH_SIZE
    loss_config["mask_loss_mul"] = MASK_LOSS_MUL
    ic(loss_config["mask_loss_mul"])
    ic(vit_config["model_name"])

    CHECKPOINT_EPOCH_IDX = get_checkpoint_idx(ckpt_path=CKPT_PATH)
    RUN_BASE_MODEL = vit_config[
        'run_base_model']  # TODO If True, Running only forward of the image to create visualization of the base model

    feature_extractor = ViTFeatureExtractor.from_pretrained(vit_config["model_name"])
    if vit_config["model_name"] in ["google/vit-base-patch16-224"]:
        vit_for_classification_image, vit_for_patch_classification = load_vit_pretrained(
            model_name=vit_config["model_name"])
    else:
        vit_for_classification_image = ViTForImageClassification.from_pretrained(vit_config["model_name"])
        vit_for_patch_classification = ViTForMaskGeneration.from_pretrained(vit_config["model_name"])

    warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(
        n_epochs=vit_config["n_epochs"],
        train_samples_length=len(list(Path(IMAGENET_VAL_IMAGES_FOLDER_PATH).iterdir())),
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
        experiment_path='exp_name'
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
    print_segmentation_results(pixAcc=pixAcc, mAp=mAp, mIoU=mIoU, mF1=mF1)
