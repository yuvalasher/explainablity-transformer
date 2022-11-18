import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
from transformers import ViTForImageClassification

from feature_extractor import ViTFeatureExtractor
from tqdm import tqdm

from main.seg_classification.image_classification_with_token_classification_model_opt import \
    OptImageClassificationWithTokenClassificationModel
from main.seg_classification.image_token_data_module_opt import ImageSegOptDataModule
import torch

from config import config
from icecream import ic

from main.seg_classification.seg_cls_utils import load_pickles_and_calculate_auc, create_folder_hierarchy, \
    get_gt_classes
from models.modeling_vit_patch_classification import ViTForMaskGeneration
from utils import remove_old_results_dfs
from vit_loader.load_vit import load_vit_pretrained
from pathlib import Path
import pytorch_lightning as pl
from utils.consts import (
    IMAGENET_VAL_IMAGES_FOLDER_PATH,
    EXPERIMENTS_FOLDER_PATH, RESULTS_PICKLES_FOLDER_PATH,
    GT_VALIDATION_PATH_LABELS,
)
from vit_utils import (
    get_warmup_steps_and_total_training_steps,
    freeze_multitask_model,
    print_number_of_trainable_and_not_trainable_params, get_loss_multipliers, get_checkpoint_idx, get_ckpt_model_auc,
)
from pytorch_lightning import seed_everything
import gc
from PIL import ImageFile
import logging
import warnings

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger('checkpoint').setLevel(0)
logging.getLogger('lightning').setLevel(0)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

vit_config = config["vit"]
loss_config = vit_config["seg_cls"]["loss"]
seed_everything(config["general"]["seed"])
vit_config["enable_checkpointing"] = False
ImageFile.LOAD_TRUNCATED_IMAGES = True
gc.collect()
from main.seg_classification.vit_backbone_to_details import VIT_BACKBONE_DETAILS

loss_multipliers = get_loss_multipliers(loss_config=loss_config)
target_or_predicted_model = "target" if vit_config["train_model_by_target_gt_class"] else "predicted"

CKPT_PATH, IMG_SIZE, PATCH_SIZE, MASK_LOSS_MUL = VIT_BACKBONE_DETAILS[vit_config["model_name"]]["ckpt_path"][
                                                     target_or_predicted_model], \
                                                 VIT_BACKBONE_DETAILS[vit_config["model_name"]][
                                                     "img_size"], VIT_BACKBONE_DETAILS[vit_config["model_name"]][
                                                     "patch_size"], VIT_BACKBONE_DETAILS[vit_config["model_name"]][
                                                     "mask_loss"]
CHECKPOINT_EPOCH_IDX = get_checkpoint_idx(ckpt_path=CKPT_PATH)
BASE_CKPT_MODEL_AUC = get_ckpt_model_auc(ckpt_path=CKPT_PATH)
loss_config["mask_loss_mul"] = MASK_LOSS_MUL
vit_config["img_size"] = IMG_SIZE
vit_config["patch_size"] = PATCH_SIZE
ic(loss_config["mask_loss_mul"])

exp_name = f'TESTTEST_direct_opt_ckpt_{CHECKPOINT_EPOCH_IDX}_auc_{BASE_CKPT_MODEL_AUC}_model_{vit_config["model_name"].replace("/", "_")}_train_uni_{vit_config["is_sampled_train_data_uniformly"]}_val_unif_{vit_config["is_sampled_val_data_uniformly"]}_activation_{vit_config["activation_function"]}_pred_{loss_multipliers["prediction_loss_mul"]}_mask_l_{loss_config["mask_loss"]}_{loss_multipliers["mask_loss_mul"]}__train_n_samples_{vit_config["seg_cls"]["train_n_label_sample"] * 1000}_lr_{vit_config["lr"]}__bs_{vit_config["batch_size"]}__layers_freezed_{vit_config["segmentation_transformer_n_first_layers_to_freeze"]}_by_target_gt__{vit_config["train_model_by_target_gt_class"]}'

plot_path = Path(vit_config["plot_path"], exp_name)
RUN_BASE_MODEL = vit_config["run_base_model"]

BASE_AUC_OBJECTS_PATH = Path(RESULTS_PICKLES_FOLDER_PATH, 'target' if vit_config[
    "train_model_by_target_gt_class"] else 'predicted')

ic(vit_config["model_name"])
ic(vit_config["train_model_by_target_gt_class"])

EXP_PATH = Path(BASE_AUC_OBJECTS_PATH, exp_name)
os.makedirs(EXP_PATH, exist_ok=True)
ic(EXP_PATH, RUN_BASE_MODEL)

BEST_AUC_PLOT_PATH, BEST_AUC_OBJECTS_PATH, BASE_MODEL_BEST_AUC_PLOT_PATH, BASE_MODEL_BEST_AUC_OBJECTS_PATH = create_folder_hierarchy(
    base_auc_objects_path=BASE_AUC_OBJECTS_PATH, exp_name=exp_name)

feature_extractor = ViTFeatureExtractor.from_pretrained(vit_config["model_name"])
if vit_config["model_name"] in ["google/vit-base-patch16-224"]:
    vit_for_classification_image, vit_for_patch_classification = load_vit_pretrained(
        model_name=vit_config["model_name"])
else:
    vit_for_classification_image = ViTForImageClassification.from_pretrained(vit_config["model_name"])
    vit_for_patch_classification = ViTForMaskGeneration.from_pretrained(vit_config["model_name"])

ic(
    str(IMAGENET_VAL_IMAGES_FOLDER_PATH),
)

warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(
    n_epochs=vit_config["n_epochs_to_optimize_stage_b"],
    train_samples_length=len(list(Path(IMAGENET_VAL_IMAGES_FOLDER_PATH).iterdir())),
    batch_size=vit_config["batch_size"],
)

model = OptImageClassificationWithTokenClassificationModel(
    vit_for_classification_image=vit_for_classification_image,
    vit_for_patch_classification=vit_for_patch_classification,
    feature_extractor=feature_extractor,
    is_clamp_between_0_to_1=vit_config["is_clamp_between_0_to_1"],
    plot_path=plot_path,
    warmup_steps=warmup_steps,
    total_training_steps=total_training_steps,
    batch_size=vit_config["batch_size"],
    best_auc_objects_path=BASE_MODEL_BEST_AUC_OBJECTS_PATH if RUN_BASE_MODEL else BEST_AUC_OBJECTS_PATH,
    checkpoint_epoch_idx=CHECKPOINT_EPOCH_IDX,
    best_auc_plot_path=BASE_MODEL_BEST_AUC_PLOT_PATH if RUN_BASE_MODEL else BEST_AUC_PLOT_PATH,
    run_base_model_only=RUN_BASE_MODEL,
)

experiment_path = Path(EXPERIMENTS_FOLDER_PATH, vit_config["evaluation"]["experiment_folder_name"])
remove_old_results_dfs(experiment_path=experiment_path)
model = freeze_multitask_model(
    model=model,
    freezing_classification_transformer=vit_config["freezing_classification_transformer"],
    segmentation_transformer_n_first_layers_to_freeze=vit_config["segmentation_transformer_n_first_layers_to_freeze"]
)
print(exp_name)
print_number_of_trainable_and_not_trainable_params(model)


if __name__ == '__main__':
    IMAGES_PATH = IMAGENET_VAL_IMAGES_FOLDER_PATH
    ic(exp_name)
    print(f"Total Images in path: {len(os.listdir(IMAGES_PATH))}")
    ic(vit_config['lr'], loss_multipliers["mask_loss_mul"], loss_multipliers["prediction_loss_mul"])
    listdir = sorted(list(Path(IMAGES_PATH).iterdir()))
    targets = get_gt_classes(path=GT_VALIDATION_PATH_LABELS)
    for idx, (image_path, target) in tqdm(enumerate(zip(listdir, targets)), position=0, leave=True, total=len(listdir)):
        data_module = ImageSegOptDataModule(
            feature_extractor=feature_extractor,
            batch_size=1,
            train_image_path=str(image_path),
            val_image_path=str(image_path),
            target=target,
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
    mean_auc = load_pickles_and_calculate_auc(
        path=BASE_MODEL_BEST_AUC_OBJECTS_PATH if RUN_BASE_MODEL else BEST_AUC_OBJECTS_PATH)
    print(f"Mean AUC: {mean_auc}")
    ic(exp_name)
