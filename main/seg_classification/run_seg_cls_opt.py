import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from main.seg_classification.model_types_loading import load_explainer_explaniee_models_and_feature_extractor, \
    CONVNET_MODELS_BY_NAME
from tqdm import tqdm
from main.seg_classification.image_classification_with_token_classification_model_opt import \
    OptImageClassificationWithTokenClassificationModel
from main.seg_classification.image_token_data_module_opt import ImageSegOptDataModule
from config import config
from icecream import ic

from main.seg_classification.seg_cls_utils import load_pickles_and_calculate_auc, create_folder_hierarchy, \
    get_gt_classes
from utils import remove_old_results_dfs
from pathlib import Path
import pytorch_lightning as pl
from utils.consts import (
    IMAGENET_VAL_IMAGES_FOLDER_PATH,
    EXPERIMENTS_FOLDER_PATH, RESULTS_PICKLES_FOLDER_PATH,
    GT_VALIDATION_PATH_LABELS,
)
from main.seg_classification.backbone_to_details import BACKBONE_DETAILS
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
mask_loss_mul = loss_config["mask_loss_mul"]
prediction_loss_mul = loss_config["prediction_loss_mul"]
lr = vit_config['lr']
n_epochs_to_optimize_stage_b = vit_config["n_epochs_to_optimize_stage_b"]
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
train_n_samples = vit_config["seg_cls"]["train_n_label_sample"]
RUN_BASE_MODEL = vit_config["run_base_model"]
IS_EXPLANIEE_CONVNET = True if explainee_model_name in CONVNET_MODELS_BY_NAME.keys() else False

seed_everything(config["general"]["seed"])
vit_config["enable_checkpointing"] = False
ImageFile.LOAD_TRUNCATED_IMAGES = True
gc.collect()

get_loss_multipliers(normalize=False,
                     mask_loss_mul=mask_loss_mul,
                     prediction_loss_mul=prediction_loss_mul)
target_or_predicted_model = "target" if train_model_by_target_gt_class else "predicted"

CKPT_PATH, IMG_SIZE, PATCH_SIZE, MASK_LOSS_MUL = BACKBONE_DETAILS[explainee_model_name]["ckpt_path"][
                                                     target_or_predicted_model], \
                                                 BACKBONE_DETAILS[explainee_model_name]["img_size"], \
                                                 BACKBONE_DETAILS[explainee_model_name]["patch_size"], \
                                                 BACKBONE_DETAILS[explainee_model_name]["mask_loss"]
CHECKPOINT_EPOCH_IDX = get_checkpoint_idx(ckpt_path=CKPT_PATH)
BASE_CKPT_MODEL_AUC = get_ckpt_model_auc(ckpt_path=CKPT_PATH)
mask_loss_mul = MASK_LOSS_MUL
vit_config["img_size"] = IMG_SIZE
vit_config["patch_size"] = PATCH_SIZE

exp_name = f'direct_opt_ckpt_{CHECKPOINT_EPOCH_IDX}_auc_{BASE_CKPT_MODEL_AUC}_explanier_{explainer_model_name.replace("/", "_")}__explaniee_{explainee_model_name.replace("/", "_")}__train_uni_{is_sampled_train_data_uniformly}_val_unif_{is_sampled_val_data_uniformly}_activation_{vit_config["activation_function"]}_pred_{prediction_loss_mul}_mask_l_{loss_config["mask_loss"]}_{mask_loss_mul}__train_n_samples_{train_n_samples * 1000}_lr_{vit_config["lr"]}__bs_{batch_size}_by_target_gt__{train_model_by_target_gt_class}'
plot_path = Path(vit_config["plot_path"], exp_name)

BASE_AUC_OBJECTS_PATH = Path(RESULTS_PICKLES_FOLDER_PATH, 'target' if train_model_by_target_gt_class else 'predicted')

EXP_PATH = Path(BASE_AUC_OBJECTS_PATH, exp_name)
os.makedirs(EXP_PATH, exist_ok=True)
ic(vit_config["verbose"])
ic(EXP_PATH, RUN_BASE_MODEL)
ic(mask_loss_mul)
ic(train_model_by_target_gt_class)
ic(str(IMAGENET_VAL_IMAGES_FOLDER_PATH))

BEST_AUC_PLOT_PATH, BEST_AUC_OBJECTS_PATH, BASE_MODEL_BEST_AUC_PLOT_PATH, BASE_MODEL_BEST_AUC_OBJECTS_PATH = create_folder_hierarchy(
    base_auc_objects_path=BASE_AUC_OBJECTS_PATH, exp_name=exp_name)

model_for_classification_image, model_for_patch_classification, feature_extractor = load_explainer_explaniee_models_and_feature_extractor(
    explainee_model_name=explainee_model_name, explainer_model_name=explainer_model_name)

warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(
    n_epochs=n_epochs_to_optimize_stage_b,
    train_samples_length=len(list(Path(IMAGENET_VAL_IMAGES_FOLDER_PATH).iterdir())),
    batch_size=batch_size,
)

model = OptImageClassificationWithTokenClassificationModel(
    model_for_classification_image=model_for_classification_image,
    model_for_patch_classification=model_for_patch_classification,
    is_clamp_between_0_to_1=is_clamp_between_0_to_1,
    plot_path=plot_path,
    warmup_steps=warmup_steps,
    total_training_steps=total_training_steps,
    batch_size=batch_size,
    best_auc_objects_path=BASE_MODEL_BEST_AUC_OBJECTS_PATH if RUN_BASE_MODEL else BEST_AUC_OBJECTS_PATH,
    checkpoint_epoch_idx=CHECKPOINT_EPOCH_IDX,
    best_auc_plot_path=BASE_MODEL_BEST_AUC_PLOT_PATH if RUN_BASE_MODEL else BEST_AUC_PLOT_PATH,
    run_base_model_only=RUN_BASE_MODEL,
    is_convnet=IS_EXPLANIEE_CONVNET,
)

experiment_path = Path(EXPERIMENTS_FOLDER_PATH, vit_config["evaluation"]["experiment_folder_name"])
remove_old_results_dfs(experiment_path=experiment_path)
model = freeze_multitask_model(
    model=model,
    freezing_classification_transformer=freezing_classification_transformer,
    segmentation_transformer_n_first_layers_to_freeze=segmentation_transformer_n_first_layers_to_freeze
)
print(exp_name)
print_number_of_trainable_and_not_trainable_params(model)

if __name__ == '__main__':
    IMAGES_PATH = IMAGENET_VAL_IMAGES_FOLDER_PATH
    ic(exp_name)
    print(f"Total Images in path: {len(os.listdir(IMAGES_PATH))}")
    ic(mask_loss_mul, prediction_loss_mul)
    listdir = sorted(list(Path(IMAGES_PATH).iterdir()))
    targets = get_gt_classes(path=GT_VALIDATION_PATH_LABELS)
    for idx, (image_path, target) in tqdm(enumerate(zip(listdir, targets)), position=0, leave=True, total=len(listdir)):
        data_module = ImageSegOptDataModule(
            batch_size=1,
            train_image_path=str(image_path),
            val_image_path=str(image_path),
            target=target,
            feature_extractor=feature_extractor,
            is_explaniee_convnet=IS_EXPLANIEE_CONVNET,
        )
        trainer = pl.Trainer(
            logger=[],
            accelerator='gpu',
            gpus=1,
            devices=[1, 2],
            num_sanity_val_steps=0,
            check_val_every_n_epoch=100,
            max_epochs=CHECKPOINT_EPOCH_IDX + n_epochs_to_optimize_stage_b,
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
