import os
from typing import Tuple

import yaml

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from config import config

device = torch.device(type='cuda', index=config["general"]["gpu_index"])
from icecream import ic

from utils import remove_old_results_dfs
from vit_loader.load_vit import load_vit_pretrained

from pathlib import Path

import wandb

from main.seg_classification.image_classification_with_token_classification_model import (
    ImageClassificationWithTokenClassificationModel,
)
from models.modeling_vit_patch_classification import ViTForMaskGeneration
from main.seg_classification.image_token_data_module import ImageSegDataModule
import pytorch_lightning as pl
from utils.consts import (
    IMAGENET_VAL_IMAGES_FOLDER_PATH,
    IMAGENET_TEST_IMAGES_FOLDER_PATH,
    EXPERIMENTS_FOLDER_PATH,
    IMAGENET_TEST_IMAGES_ES_FOLDER_PATH,
)
from vit_utils import (
    load_feature_extractor_and_vit_model,
    get_warmup_steps_and_total_training_steps,
    freeze_multitask_model,
    print_number_of_trainable_and_not_trainable_params, get_loss_multipliers,
)
from transformers import AutoModel, ViTForImageClassification
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch

torch.set_default_tensor_type('torch.cuda.FloatTensor')

vit_config = config["vit"]

os.makedirs(vit_config['default_root_dir'], exist_ok=True)
loss_config = vit_config["seg_cls"]["loss"]

if torch.cuda.is_available():
    print(torch.cuda.current_device())
    torch.cuda.empty_cache()

seed_everything(config["general"]["seed"])
import gc
import torch
from PIL import ImageFile


def save_config_to_root_dir():
    path_dir = os.path.join(vit_config["default_root_dir"], f"seg_cls; {exp_name}")
    os.makedirs(path_dir, exist_ok=True)
    with open(os.path.join(path_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

ImageFile.LOAD_TRUNCATED_IMAGES = True
gc.collect()

loss_multipliers = get_loss_multipliers(loss_config=loss_config)
exp_name = f'asher_pred_{loss_multipliers["prediction_loss_mul"]}_mask_l_{loss_config["mask_loss"]}_{loss_multipliers["mask_loss_mul"]}_sigmoid_{vit_config["is_sigmoid_segmentation"]}_train_n_samples_{vit_config["seg_cls"]["train_n_samples"]}_lr_{vit_config["lr"]}_mlp_classifier_{vit_config["is_mlp_on_segmentation"]}_is_relu_{vit_config["is_relu_segmentation"]}'

feature_extractor, _ = load_feature_extractor_and_vit_model(
    vit_config=vit_config,
    model_type="vit-basic",
    is_wolf_transforms=vit_config["is_wolf_transforms"],
)  # TODO if vit-for-dino is relevant

vit_for_classification_image, vit_for_patch_classification = load_vit_pretrained(model_name=vit_config["model_name"])

ic(
    str(IMAGENET_TEST_IMAGES_FOLDER_PATH),
    str(IMAGENET_TEST_IMAGES_ES_FOLDER_PATH),
    str(IMAGENET_VAL_IMAGES_FOLDER_PATH),
)

data_module = ImageSegDataModule(
    feature_extractor=feature_extractor,
    batch_size=vit_config["batch_size"],
    train_images_path=str(IMAGENET_TEST_IMAGES_FOLDER_PATH),
    val_images_path=str(IMAGENET_TEST_IMAGES_FOLDER_PATH),
    is_sampled_data_uniformly=vit_config["is_sampled_data_uniformly"],
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
    is_clamp_between_0_to_1=vit_config["is_clamp_between_0_to_1"],
    plot_path=plot_path,
    warmup_steps=warmup_steps,
    total_training_steps=total_training_steps,
    batch_size=vit_config["batch_size"],
)

experiment_path = Path(EXPERIMENTS_FOLDER_PATH, vit_config["evaluation"]["experiment_folder_name"])
remove_old_results_dfs(experiment_path=experiment_path)
model = freeze_multitask_model(
    model=model,
    freezing_transformer=vit_config["freezing_transformer"],
)
print(exp_name)
print_number_of_trainable_and_not_trainable_params(model)

WANDB_PROJECT = config["general"]["wandb_project"]
run = wandb.init(project=WANDB_PROJECT, entity=config["general"]["wandb_entity"], config=wandb.config)
wandb_logger = WandbLogger(name=f"{exp_name}", project=WANDB_PROJECT)

trainer = pl.Trainer(
    callbacks=[
        ModelCheckpoint(monitor="val/epoch_auc", mode="min", filename="{epoch}_{val/epoch_auc:.3f}", save_top_k=20)],
    logger=[wandb_logger],
    accelerator='gpu',
    auto_select_gpus=True,
    max_epochs=vit_config["n_epochs"],
    gpus=vit_config["gpus"],
    progress_bar_refresh_rate=30,
    default_root_dir=vit_config["default_root_dir"],
    enable_checkpointing=vit_config["enable_checkpointing"]
)
if vit_config["enable_checkpointing"]:
    save_config_to_root_dir()
trainer.fit(model=model, datamodule=data_module)
