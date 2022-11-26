import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from main.seg_classification.model_types_loading import load_explainer_explaniee_models_and_feature_extractor, \
    CONVNET_MODELS_BY_NAME
from main.seg_classification.seg_cls_utils import save_config_to_root_dir
from config import config
from icecream import ic
from utils import remove_old_results_dfs
from pathlib import Path
from main.seg_classification.image_classification_with_token_classification_model import (
    ImageClassificationWithTokenClassificationModel,
)
from main.seg_classification.image_token_data_module import ImageSegDataModule
import pytorch_lightning as pl
from utils.consts import (
    IMAGENET_VAL_IMAGES_FOLDER_PATH,
    EXPERIMENTS_FOLDER_PATH,
)
from vit_utils import (
    get_warmup_steps_and_total_training_steps,
    freeze_multitask_model,
    print_number_of_trainable_and_not_trainable_params, get_loss_multipliers,
)
from pytorch_lightning import seed_everything
import torch
import gc
from PIL import ImageFile

if torch.cuda.is_available():
    print(torch.cuda.current_device())
    torch.cuda.empty_cache()
seed_everything(config["general"]["seed"])

vit_config = config["vit"]
os.makedirs(vit_config['default_root_dir'], exist_ok=True)
loss_config = vit_config["seg_cls"]["loss"]

ImageFile.LOAD_TRUNCATED_IMAGES = True
gc.collect()

batch_size = vit_config["batch_size"]
n_epochs = vit_config["n_epochs"]
is_sampled_train_data_uniformly = vit_config["is_sampled_train_data_uniformly"]
is_sampled_val_data_uniformly = vit_config["is_sampled_val_data_uniformly"]
train_model_by_target_gt_class = vit_config["train_model_by_target_gt_class"]
freezing_classification_transformer = vit_config["freezing_classification_transformer"]
segmentation_transformer_n_first_layers_to_freeze = vit_config["segmentation_transformer_n_first_layers_to_freeze"]
is_clamp_between_0_to_1 = vit_config["is_clamp_between_0_to_1"]
enable_checkpointing = vit_config["enable_checkpointing"]
is_competitive_method_transforms = vit_config["is_competitive_method_transforms"]
explainer_model_name = vit_config["explainer_model_name"]
explainee_model_name = vit_config["explainee_model_name"]
plot_path = vit_config["plot_path"]
default_root_dir = vit_config["default_root_dir"]
train_n_samples = vit_config["seg_cls"]["train_n_label_sample"]
mask_loss_mul = loss_config["mask_loss_mul"]
prediction_loss_mul = loss_config["prediction_loss_mul"]
IS_EXPLANIEE_CONVNET = True if explainee_model_name in CONVNET_MODELS_BY_NAME.keys() else False

loss_multipliers = get_loss_multipliers(normalize=False,
                                        mask_loss_mul=mask_loss_mul,
                                        prediction_loss_mul=prediction_loss_mul)
ic(vit_config["verbose"])
ic(train_model_by_target_gt_class)
ic(is_sampled_train_data_uniformly)
ic(is_sampled_val_data_uniformly)
ic(is_competitive_method_transforms)
ic(explainer_model_name)
ic(explainee_model_name)
ic(str(IMAGENET_VAL_IMAGES_FOLDER_PATH))

exp_name = f'explanier_{explainer_model_name.replace("/", "_")}__explaniee_{explainee_model_name.replace("/", "_")}__train_uni_{is_sampled_train_data_uniformly}_val_unif_{is_sampled_val_data_uniformly}_activation_{vit_config["activation_function"]}_pred_{loss_multipliers["prediction_loss_mul"]}_mask_l_{loss_config["mask_loss"]}_{loss_multipliers["mask_loss_mul"]}__train_n_samples_{train_n_samples * 1000}_lr_{vit_config["lr"]}__bs_{batch_size}_by_target_gt__{train_model_by_target_gt_class}'

model_for_classification_image, model_for_mask_generation, feature_extractor = load_explainer_explaniee_models_and_feature_extractor(
    explainee_model_name=explainee_model_name, explainer_model_name=explainer_model_name)


data_module = ImageSegDataModule(
    feature_extractor=feature_extractor,
    is_explaniee_convnet=IS_EXPLANIEE_CONVNET,
    batch_size=batch_size,
    train_images_path=str(IMAGENET_VAL_IMAGES_FOLDER_PATH),
    val_images_path=str(IMAGENET_VAL_IMAGES_FOLDER_PATH),
    is_sampled_train_data_uniformly=is_sampled_train_data_uniformly,
    is_sampled_val_data_uniformly=is_sampled_val_data_uniformly,
)

warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(
    n_epochs=n_epochs,
    train_samples_length=len(list(Path(IMAGENET_VAL_IMAGES_FOLDER_PATH).iterdir())),
    batch_size=batch_size,
)

plot_path = Path(plot_path, exp_name)

experiment_perturbation_results_path = Path(EXPERIMENTS_FOLDER_PATH, "results_df", exp_name)

ic(experiment_perturbation_results_path)

model = ImageClassificationWithTokenClassificationModel(
    model_for_classification_image=model_for_classification_image,
    model_for_mask_generation=model_for_mask_generation,
    is_clamp_between_0_to_1=is_clamp_between_0_to_1,
    plot_path=plot_path,
    warmup_steps=warmup_steps,
    total_training_steps=total_training_steps,
    batch_size=batch_size,
    experiment_path=experiment_perturbation_results_path,
    is_convnet=IS_EXPLANIEE_CONVNET,
)

remove_old_results_dfs(experiment_path=experiment_perturbation_results_path)
model = freeze_multitask_model(
    model=model,
    freezing_classification_transformer=freezing_classification_transformer,
    segmentation_transformer_n_first_layers_to_freeze=segmentation_transformer_n_first_layers_to_freeze
)
print(exp_name)
print_number_of_trainable_and_not_trainable_params(model)

checkpoints_default_root_dir = str(Path(default_root_dir, 'target' if train_model_by_target_gt_class else 'predicted',
                                        exp_name))

ic(checkpoints_default_root_dir)
WANDB_PROJECT = config["general"]["wandb_project"]
# run = wandb.init(project=WANDB_PROJECT, entity=config["general"]["wandb_entity"], config=wandb.config)
# wandb_logger = WandbLogger(name=f"{exp_name}", project=WANDB_PROJECT)

trainer = pl.Trainer(
    # callbacks=[
    # ModelCheckpoint(monitor="val/epoch_auc", mode="min", dirpath=checkpoints_default_root_dir, verbose=True,
    #                 filename="{epoch}_{val/epoch_auc:.3f}", save_top_k=50)],
    # logger=[wandb_logger],
    accelerator='gpu',
    auto_select_gpus=True,
    max_epochs=n_epochs,
    gpus=vit_config["gpus"],
    progress_bar_refresh_rate=30,
    num_sanity_val_steps=0,
    default_root_dir=checkpoints_default_root_dir,
    enable_checkpointing=enable_checkpointing,
)

if enable_checkpointing:
    save_config_to_root_dir(exp_name=exp_name)
model.p = 1
trainer.fit(model=model, datamodule=data_module)
