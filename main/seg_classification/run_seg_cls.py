import os
from feature_extractor import ViTFeatureExtractor
from main.seg_classification.seg_cls_utils import save_config_to_root_dir

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from config import config
from icecream import ic
from utils import remove_old_results_dfs
from vit_loader.load_vit import load_vit_pretrained
from pathlib import Path
import wandb
from main.seg_classification.image_classification_with_token_classification_model import (
    ImageClassificationWithTokenClassificationModel,
)
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
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch

vit_config = config["vit"]

os.makedirs(vit_config['default_root_dir'], exist_ok=True)
loss_config = vit_config["seg_cls"]["loss"]

if torch.cuda.is_available():
    print(torch.cuda.current_device())
    torch.cuda.empty_cache()

seed_everything(config["general"]["seed"])
import gc
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
gc.collect()

loss_multipliers = get_loss_multipliers(loss_config=loss_config)
exp_name = f'model_{vit_config["model_name"]}__is_uniformly_{vit_config["is_sampled_data_uniformly"]}__use_logits_{loss_config["use_logits_only"]}_activation_{vit_config["activation_function"]}__normalize_by_max_patch_{vit_config["normalize_by_max_patch"]}_pred_{loss_multipliers["prediction_loss_mul"]}_mask_l_{loss_config["mask_loss"]}_{loss_multipliers["mask_loss_mul"]}__train_n_samples_{vit_config["seg_cls"]["train_n_label_sample"] * 1000}_lr_{vit_config["lr"]}_mlp_classifier_{vit_config["is_mlp_on_segmentation"]}__bs_{vit_config["batch_size"]}__n_layers_seg_freezed_{vit_config["segmentation_transformer_n_first_layers_to_freeze"]}__add_epsilon_{vit_config["add_epsilon_to_patches_scores"]}'

feature_extractor = ViTFeatureExtractor.from_pretrained(vit_config["model_name"])
if vit_config["model_name"] == "google/vit-base-patch16-224":
    vit_for_classification_image, vit_for_patch_classification = load_vit_pretrained(
        model_name=vit_config["model_name"])
else:
    vit_for_classification_image = ViTForImageClassification.from_pretrained(vit_config["model_name"])
    vit_for_patch_classification = ViTForMaskGeneration.from_pretrained(vit_config["model_name"])

ic(vit_config["model_name"])
ic(
    str(IMAGENET_TEST_IMAGES_FOLDER_PATH),
    str(IMAGENET_TEST_IMAGES_ES_FOLDER_PATH),
    str(IMAGENET_VAL_IMAGES_FOLDER_PATH),
)

data_module = ImageSegDataModule(
    feature_extractor=feature_extractor,
    batch_size=vit_config["batch_size"],
    train_images_path=str(IMAGENET_VAL_IMAGES_FOLDER_PATH),
    val_images_path=str(IMAGENET_VAL_IMAGES_FOLDER_PATH),
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
ic(model.device)

experiment_path = Path(EXPERIMENTS_FOLDER_PATH, vit_config["evaluation"]["experiment_folder_name"])
remove_old_results_dfs(experiment_path=experiment_path)
model = freeze_multitask_model(
    model=model,
    freezing_classification_transformer=vit_config["freezing_classification_transformer"],
    segmentation_transformer_n_first_layers_to_freeze=vit_config["segmentation_transformer_n_first_layers_to_freeze"]
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
    enable_checkpointing=vit_config["enable_checkpointing"],
    num_sanity_val_steps=0,
)
if vit_config["enable_checkpointing"]:
    save_config_to_root_dir(exp_name=exp_name)
trainer.fit(model=model, datamodule=data_module)
