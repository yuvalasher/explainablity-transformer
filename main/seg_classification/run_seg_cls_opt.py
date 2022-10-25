import os
import sys
# os.chdir('/home/amiteshel1/Projects/explainablity-transformer-cv/')
# sys.path.append('/home/amiteshel1/Projects/explainablity-transformer-cv/')

import wandb
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm

from main.seg_classification.image_classification_with_token_classification_model_opt import \
    OptImageClassificationWithTokenClassificationModel
from main.seg_classification.image_token_data_module_opt import ImageSegOptDataModule
import torch

from config import config
from icecream import ic
from datetime import datetime as dt

from main.seg_classification.seg_cls_utils import load_pickles_and_calculate_auc, create_folder_hierarchy
from utils import remove_old_results_dfs
from vit_loader.load_vit import load_vit_pretrained
from pathlib import Path
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
import gc
from PIL import ImageFile
import logging
import warnings

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger('checkpoint').setLevel(0)
logging.getLogger('lightning').setLevel(0)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

vit_config = config["vit"]
loss_config = vit_config["seg_cls"]["loss"]
seed_everything(config["general"]["seed"])
ImageFile.LOAD_TRUNCATED_IMAGES = True
gc.collect()

loss_multipliers = get_loss_multipliers(loss_config=loss_config)
exp_name = f'direct_opt_from_ckpt__use_logits_only_{loss_config["use_logits_only"]}_activation_func_{vit_config["activation_function"]}__normalize_by_max_patch_{vit_config["normalize_by_max_patch"]}__is_sampled_data_uniformly_{vit_config["is_sampled_data_uniformly"]}_pred_{loss_multipliers["prediction_loss_mul"]}_mask_l_{loss_config["mask_loss"]}_{loss_multipliers["mask_loss_mul"]}__train_n_samples_{vit_config["seg_cls"]["train_n_label_sample"] * 1000}_lr_{vit_config["lr"]}_mlp_classifier_{vit_config["is_mlp_on_segmentation"]}__n_layers_seg_transformer_freezed_{vit_config["segmentation_transformer_n_first_layers_to_freeze"]}'

plot_path = Path(vit_config["plot_path"], exp_name)

CKPT_PATH = "/home/amiteshel1/Projects/explainablity-transformer-cv/research/checkpoints/token_classification/seg_cls; amit__pred_1_mask_l_bce_50_sigmoid_True_train_n_samples_6000_lr_0.002_mlp_classifier_True_is_relu_False/None/checkpoints/epoch=4--val/epoch_auc=19.940.ckpt"
CHECKPOINT_EPOCH_IDX = 5  # TODO - pay attention !!!
RUN_BASE_MODEL = vit_config["run_base_model"]  # TODO - Need to pay attention! If True, Running only forward of the image to create visualization of the base model

BASE_AUC_OBJECTS_PATH = Path(EXPERIMENTS_FOLDER_PATH, vit_config['evaluation'][
    'experiment_folder_name'])  # /home/yuvalas/explainability/research/experiments/seg_cls/

EXP_NAME = 'EXP_NAME'

EXP_PATH = Path(BASE_AUC_OBJECTS_PATH, EXP_NAME)
os.makedirs(EXP_PATH, exist_ok=True)
ic(EXP_PATH)

BEST_AUC_PLOT_PATH, BEST_AUC_OBJECTS_PATH, BASE_MODEL_BEST_AUC_PLOT_PATH, BASE_MODEL_BEST_AUC_OBJECTS_PATH = create_folder_hierarchy(
    base_auc_objects_path=BASE_AUC_OBJECTS_PATH, exp_name=EXP_NAME)

feature_extractor, _ = load_feature_extractor_and_vit_model(
    vit_config=vit_config,
    model_type="vit-basic",
    is_wolf_transforms=vit_config["is_wolf_transforms"],
)

vit_for_classification_image, vit_for_patch_classification = load_vit_pretrained(model_name=vit_config["model_name"])

ic(
    str(IMAGENET_TEST_IMAGES_FOLDER_PATH),
    str(IMAGENET_TEST_IMAGES_ES_FOLDER_PATH),
    str(IMAGENET_VAL_IMAGES_FOLDER_PATH),
)

warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(
    n_epochs=vit_config["n_epochs"],
    train_samples_length=len(list(Path(IMAGENET_TEST_IMAGES_FOLDER_PATH).iterdir())),
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

WANDB_PROJECT = config["general"]["wandb_project"]
# run = wandb.init(project=WANDB_PROJECT, entity=config["general"]["wandb_entity"], config=wandb.config)
# wandb_logger = WandbLogger(name=f"{exp_name}", project=WANDB_PROJECT)

if __name__ == '__main__':
    IMAGES_PATH = IMAGENET_VAL_IMAGES_FOLDER_PATH
    print(f"Total Images in path: {len(os.listdir(IMAGES_PATH))}")
    ic(vit_config['lr'], loss_multipliers["mask_loss_mul"], loss_multipliers["prediction_loss_mul"])
    start_time = dt.now()
    listdir = sorted(list(Path(IMAGES_PATH).iterdir()))
    for idx, image_path in tqdm(enumerate(listdir), position=0, leave=True, total=len(listdir)):

        data_module = ImageSegOptDataModule(
            feature_extractor=feature_extractor,
            batch_size=1,
            train_image_path=str(image_path),
            val_image_path=str(image_path),
        )
        trainer = pl.Trainer(
            # callbacks=[early_stop_callback],
            # logger=[wandb_logger],
            logger=[],
            accelerator='gpu',
            gpus=1,
            devices=[1, 2],
            num_sanity_val_steps=0,
            check_val_every_n_epoch=100,
            max_epochs=vit_config["n_epochs"],
            resume_from_checkpoint=CKPT_PATH,
            enable_progress_bar=False,
            enable_checkpointing=False,
            default_root_dir=vit_config["default_root_dir"],
            weights_summary=None
        )
        trainer.fit(model=model, datamodule=data_module)
        if (idx % 1000 == 0):
            mean_auc = load_pickles_and_calculate_auc(
                path=BASE_MODEL_BEST_AUC_OBJECTS_PATH if RUN_BASE_MODEL else BEST_AUC_OBJECTS_PATH)
            print(f"Epoch: {idx}  ---> Mean AUC: {mean_auc}")
    # print(f"Time diff: {dt.now() - start_time}")
    mean_auc = load_pickles_and_calculate_auc(
        path=BASE_MODEL_BEST_AUC_OBJECTS_PATH if RUN_BASE_MODEL else BEST_AUC_OBJECTS_PATH)
    print(f"Mean AUC: {mean_auc}")
    print("FINISH!!!")
