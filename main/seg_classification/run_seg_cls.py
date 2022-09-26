import os

from utils import remove_old_results_dfs

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from pathlib import Path
from typing import Tuple

import wandb

from main.seg_classification.image_classification_with_token_classification_model import (
    ImageClassificationWithTokenClassificationModel,
)
from models.modeling_vit_patch_classification import ViTForMaskGeneration
from main.seg_classification.image_token_data_module import ImageSegDataModule
import pytorch_lightning as pl
from config import config
from utils.consts import IMAGENET_VAL_IMAGES_FOLDER_PATH, IMAGENET_TEST_IMAGES_FOLDER_PATH, EXPERIMENTS_FOLDER_PATH, \
    IMAGENET_TEST_IMAGES_ES_FOLDER_PATH
from vit_utils import (
    load_feature_extractor_and_vit_model,
    get_warmup_steps_and_total_training_steps,
    freeze_multitask_model,
    print_number_of_trainable_and_not_trainable_params,
)
from transformers import AutoModel, ViTForImageClassification
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch

if torch.cuda.is_available():
    print(torch.cuda.current_device())
    torch.cuda.empty_cache()

vit_config = config["vit"]
loss_config = vit_config["seg_cls"]["loss"]
seed_everything(config["general"]["seed"])
import gc
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
gc.collect()
base_exp_name = f'pred_loss_{vit_config["seg_cls"]["loss"]["prediction_loss_mul"]}_mask_loss'
if loss_config["mask_loss"] == "bce":
    exp_name = f'test_data_{base_exp_name}_bce_to_0_{vit_config["seg_cls"]["loss"]["mask_loss_mul"]}_sigmoid_{vit_config["is_sigmoid_segmentation"]}'
else:
    exp_name = f'test_data_{base_exp_name}_l1_{vit_config["seg_cls"]["loss"]["mask_loss_mul"]}_sigmoid_{vit_config["is_sigmoid_segmentation"]}'

feature_extractor, _ = load_feature_extractor_and_vit_model(
    vit_config=vit_config,
    model_type="vit-basic",
    is_wolf_transforms=vit_config["is_wolf_transforms"],
)  # TODO if vit-for-dino is relevant

vit_for_classification_image = ViTForImageClassification.from_pretrained(vit_config["model_name"])
vit_for_patch_classification = ViTForMaskGeneration.from_pretrained(vit_config["model_name"])
ic(IMAGENET_TEST_IMAGES_FOLDER_PATH, IMAGENET_TEST_IMAGES_ES_FOLDER_PATH, IMAGENET_VAL_IMAGES_FOLDER_PATH)

data_module = ImageSegDataModule(
    feature_extractor=feature_extractor,
    batch_size=vit_config["batch_size"],
    train_images_path=str(IMAGENET_TEST_IMAGES_FOLDER_PATH),
    train_n_samples=vit_config["seg_cls"]["train_n_samples"],
    val_images_path=str(IMAGENET_TEST_IMAGES_ES_FOLDER_PATH),
    val_n_samples=vit_config["seg_cls"]["val_n_samples"],
    test_images_path=str(IMAGENET_VAL_IMAGES_FOLDER_PATH),
    test_n_samples=vit_config["seg_cls"]["test_n_samples"],
)

warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(
    n_epochs=vit_config["n_epochs"],
    train_samples_length=len(list(Path(IMAGENET_TEST_IMAGES_FOLDER_PATH).iterdir())),
    batch_size=vit_config["batch_size"],
)
plot_path = Path(vit_config["plot_path"], exp_name)
model = ImageClassificationWithTokenClassificationModel(
    vit_for_classification_image=vit_for_classification_image,
    vit_for_patch_classification=vit_for_patch_classification,
    feature_extractor=feature_extractor,
    plot_path=plot_path,
    warmup_steps=warmup_steps,
    total_training_steps=total_training_steps,
    batch_size=vit_config["batch_size"],
)

WANDB_PROJECT = "run_seg_cls_4"
run = wandb.init(project=WANDB_PROJECT, entity="yuvalasher", config=wandb.config)
# wandb.login()
wandb_logger = WandbLogger(name=f'seg_cls; {exp_name}', project=WANDB_PROJECT)


# early_stop_callback = EarlyStopping(
#    monitor='val_loss',
#    min_delta=0.0001,
#    patience=3,
#    verbose=False,
#    mode='min')


experiment_path = Path(EXPERIMENTS_FOLDER_PATH, "seg_cls", vit_config["evaluation"]["experiment_folder_name"])
remove_old_results_dfs(experiment_path=experiment_path)
model = freeze_multitask_model(
    model=model,
    freezing_transformer=vit_config["freezing_transformer"],
    is_segmentation_transformer_freeze=vit_config["is_segmentation_transformer_freeze"],
)
print(exp_name)
print_number_of_trainable_and_not_trainable_params(model)
trainer = pl.Trainer(
    # callbacks=[ModelCheckpoint(monitor="val_loss", mode="min", filename="{epoch}--{val_loss:.1f}", save_top_k=1)],
    # TODO - change
    # , early_stop_callback],
    logger=[wandb_logger],
    # logger=[],
    max_epochs=vit_config["n_epochs"],
    gpus=vit_config["gpus"],
    progress_bar_refresh_rate=30,
    default_root_dir=vit_config["default_root_dir"],
)

trainer.fit(model=model, datamodule=data_module)
