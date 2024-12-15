import sys
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["WANDB__SERVICE_WAIT"] = "1000"
os.chdir('/home/amitesh/Projects/explainablity-transformer-cv')
sys.path.append('/home/amitesh/Projects/explainablity-transformer-cv')

from icecream import ic

# Print Python interpreter information
ic(f"Python Interpreter: {sys.executable}")
ic(f"Python Version: {sys.version}")

# Print current working directory
ic(f"Current Working Directory: {os.getcwd()}")


ic('start!')
import gc
from pathlib import Path

# Third-party imports
ic('Import Third-party')
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from PIL import ImageFile


# Local imports
ic('Import Local imports')
from utils.parser import get_parser
from utils.logging_utils import log_configuration
from config.config_reader import config
from utils import remove_old_results_dfs
from utils.consts import (
    IMAGENET_VAL_IMAGES_FOLDER_PATH,
    EXPERIMENTS_FOLDER_PATH,
    MODEL_OPTIONS,
    MODEL_ALIAS_MAPPING,
)
from utils.vit_utils import (
    get_warmup_steps_and_total_training_steps,
    freeze_multitask_model,
    print_number_of_trainable_and_not_trainable_params,
    get_loss_multipliers,
    get_params_from_config,
)
from main.seg_classification.model_types_loading import (
    load_explainer_explaniee_models_and_feature_extractor,
    CONVNET_MODELS_BY_NAME,
)
from main.seg_classification.seg_cls_utils import save_config_to_root_dir
from main.seg_classification.image_classification_with_token_classification_model import (
    ImageClassificationWithTokenClassificationModel,
)
from main.seg_classification.image_token_data_module import ImageSegDataModule

# Set up CUDA and seed
ic('Set up CUDA and seed')
if torch.cuda.is_available():
    print(torch.cuda.current_device())
    torch.cuda.empty_cache()
seed_everything(config["general"]["seed"])

# Configure PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Collect garbage
gc.collect()

if __name__ == '__main__':
    """
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH nohup python main/seg_classification/run_seg_cls.py --enable-checkpointing False --explainer-model-name vit_base_224 --explainee-model-name vit_base_224 --mask-loss-mul 50 --train-model-by-target-gt-class True --n-epochs 30 --train-n-label-sample 6 &> nohups_logs/journal/vit_base_224_bs32_ml50_target.out &
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH nohup python main/seg_classification/run_seg_cls.py --enable-checkpointing False --explainer-model-name vit_base_224 --explainee-model-name vit_base_224 --mask-loss-mul 50 --train-model-by-target-gt-class False --n-epochs 30 --train-n-label-sample 6 &> nohups_logs/journal/vit_base_224_bs32_ml50_predicted.out &
    """

    params_config = get_params_from_config(config_vit=config["vit"])
    parser = get_parser(params_config)
    args = parser.parse_args()

    EXPLAINEE_MODEL_NAME, EXPLAINER_MODEL_NAME = MODEL_ALIAS_MAPPING[args.explainee_model_name], \
        MODEL_ALIAS_MAPPING[args.explainer_model_name]

    IS_EXPLANIEE_CONVNET = True if EXPLAINEE_MODEL_NAME in CONVNET_MODELS_BY_NAME.keys() else False
    IS_EXPLAINER_CONVNET = True if EXPLAINER_MODEL_NAME in CONVNET_MODELS_BY_NAME.keys() else False

    loss_multipliers = get_loss_multipliers(normalize=False,
                                            mask_loss_mul=args.mask_loss_mul,
                                            prediction_loss_mul=args.prediction_loss_mul,
                                            )

    # Usage
    log_configuration(args, EXPLAINER_MODEL_NAME, EXPLAINEE_MODEL_NAME, IMAGENET_VAL_IMAGES_FOLDER_PATH)
    # exp_name = f'ARGPARSE_explanier_{EXPLAINER_MODEL_NAME.replace("/", "_")}__explaniee_{EXPLAINEE_MODEL_NAME.replace("/", "_")}__train_uni_{args.is_sampled_train_data_uniformly}_val_unif_{args.is_sampled_val_data_uniformly}_activation_{args.activation_function}_pred_{loss_multipliers["prediction_loss_mul"]}_mask_l_{args.mask_loss}_{loss_multipliers["mask_loss_mul"]}__train_n_samples_{args.train_n_label_sample * 1000}_lr_{args.lr}__bs_{args.batch_size}_by_target_gt__{args.train_model_by_target_gt_class}'
    exp_name = 'test'
    model_for_classification_image, model_for_mask_generation, feature_extractor = load_explainer_explaniee_models_and_feature_extractor(
        explainee_model_name=EXPLAINEE_MODEL_NAME,
        explainer_model_name=EXPLAINER_MODEL_NAME,
        activation_function=args.activation_function,
        img_size=args.img_size,
    )

    data_module = ImageSegDataModule(
        feature_extractor=feature_extractor,
        is_explaniee_convnet=IS_EXPLANIEE_CONVNET,
        batch_size=args.batch_size,
        train_images_path=str(IMAGENET_VAL_IMAGES_FOLDER_PATH),
        val_images_path=str(IMAGENET_VAL_IMAGES_FOLDER_PATH),
        is_sampled_train_data_uniformly=args.is_sampled_train_data_uniformly,
        is_sampled_val_data_uniformly=args.is_sampled_val_data_uniformly,
        is_competitive_method_transforms=args.is_competitive_method_transforms,
        train_n_label_sample=args.train_n_label_sample,
        val_n_label_sample=args.val_n_label_sample
    )

    warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(
        n_epochs=args.n_epochs,
        train_samples_length=len(list(Path(IMAGENET_VAL_IMAGES_FOLDER_PATH).iterdir())),
        batch_size=args.batch_size,
    )

    plot_path = Path(args.plot_path, exp_name)

    experiment_perturbation_results_path = Path(EXPERIMENTS_FOLDER_PATH, "results_df", exp_name)

    ic(experiment_perturbation_results_path)

    model = ImageClassificationWithTokenClassificationModel(
        model_for_classification_image=model_for_classification_image,
        model_for_mask_generation=model_for_mask_generation,
        is_clamp_between_0_to_1=args.is_clamp_between_0_to_1,
        plot_path=plot_path,
        warmup_steps=warmup_steps,
        total_training_steps=total_training_steps,
        experiment_path=experiment_perturbation_results_path,
        is_explainer_convnet=IS_EXPLAINER_CONVNET,
        is_explainee_convnet=IS_EXPLANIEE_CONVNET,
        lr=args.lr,
        start_epoch_to_evaluate=args.start_epoch_to_evaluate,
        n_batches_to_visualize=args.n_batches_to_visualize,
        mask_loss=args.mask_loss,
        mask_loss_mul=args.mask_loss_mul,
        prediction_loss_mul=args.prediction_loss_mul,
        activation_function=args.activation_function,
        train_model_by_target_gt_class=args.train_model_by_target_gt_class,
        use_logits_only=args.use_logits_only,
        img_size=args.img_size,
        patch_size=args.patch_size,
        is_ce_neg=args.is_ce_neg,
        verbose=args.verbose,
    )

    remove_old_results_dfs(experiment_path=experiment_perturbation_results_path)
    model = freeze_multitask_model(
        model=model,
        is_freezing_explaniee_model=args.is_freezing_explaniee_model,
        explainer_model_n_first_layers_to_freeze=args.explainer_model_n_first_layers_to_freeze,
        is_explainer_convnet=IS_EXPLAINER_CONVNET,
    )
    print(exp_name)
    print_number_of_trainable_and_not_trainable_params(model)

    checkpoints_default_root_dir = str(
        Path(args.default_root_dir, 'target' if args.train_model_by_target_gt_class else 'predicted',
             exp_name))

    ic(checkpoints_default_root_dir)
    callbacks = []
    if args.enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(monitor="val/epoch_auc", mode="min", dirpath=checkpoints_default_root_dir, verbose=True,
                            filename="{epoch}_{val/epoch_auc:.3f}", save_top_k=args.n_epochs)
        )

    WANDB_PROJECT = config["general"]["wandb_project"]
    run = wandb.init(project=WANDB_PROJECT, entity=config["general"]["wandb_entity"], config=wandb.config)
    wandb_logger = WandbLogger(name=f"{exp_name}", project=WANDB_PROJECT)

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=[wandb_logger],
        accelerator='gpu',
        auto_select_gpus=True,
        max_epochs=args.n_epochs,
        gpus=1,
        progress_bar_refresh_rate=30,
        num_sanity_val_steps=0,
        default_root_dir=checkpoints_default_root_dir,
        enable_checkpointing=args.enable_checkpointing,
    )

    if args.enable_checkpointing:
        save_config_to_root_dir(exp_name=exp_name)
    trainer.fit(model=model, datamodule=data_module)
