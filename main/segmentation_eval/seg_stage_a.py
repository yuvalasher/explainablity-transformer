import argparse
import os

from main.seg_classification.cnns.cnn_utils import CONVNET_NORMALIZATION_STD, CONVENT_NORMALIZATION_MEAN

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from main.seg_classification.model_types_loading import CONVNET_MODELS_BY_NAME, \
    load_explainer_explaniee_models_and_feature_extractor
from icecream import ic
from main.segmentation_eval.segmentation_utils import print_segmentation_results, init_get_normalize_and_transform
from pathlib import Path
from main.segmentation_eval.segmentation_model_opt import \
    OptImageClassificationWithTokenClassificationModelSegmentation
import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from main.seg_classification.image_token_data_module_opt_segmentation import ImageSegOptDataModuleSegmentation
from config import config
from utils.iou import IoU
from main.segmentation_eval.imagenet import Imagenet_Segmentation
from vit_utils import get_warmup_steps_and_total_training_steps, \
    get_loss_multipliers, freeze_multitask_model, get_params_from_config, suppress_warnings, get_backbone_details
from utils.consts import IMAGENET_SEG_PATH, IMAGENET_VAL_IMAGES_FOLDER_PATH, MODEL_ALIAS_MAPPING, MODEL_OPTIONS
import pytorch_lightning as pl
import gc
from PIL import ImageFile

suppress_warnings()
seed_everything(config["general"]["seed"])

ImageFile.LOAD_TRUNCATED_IMAGES = True
gc.collect()
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

if __name__ == '__main__':
    params_config = get_params_from_config(config_vit=config["vit"])
    parser = argparse.ArgumentParser(description='Run segmentation of pLTX model')
    parser.add_argument('--explainer-model-name', type=str, default="vit_base_224", choices=MODEL_OPTIONS)
    parser.add_argument('--explainee-model-name', type=str, default="densenet", choices=MODEL_OPTIONS)
    parser.add_argument('--train-model-by-target-gt-class',
                        type=bool,
                        default=params_config["train_model_by_target_gt_class"])
    parser.add_argument('--RUN-BASE-MODEL', type=bool, default=params_config["RUN_BASE_MODEL"])

    parser.add_argument('--verbose', type=bool, default=params_config["verbose"])
    parser.add_argument('--n_epochs_to_optimize_stage_b', type=int, default=params_config["n_epochs"])
    parser.add_argument('--n-epochs', type=int, default=params_config["n_epochs"])
    parser.add_argument('--mask-loss-mul', type=int, default=params_config["mask_loss_mul"])
    parser.add_argument('--prediction-loss-mul', type=int, default=params_config["prediction_loss_mul"])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--is-freezing-explaniee-model',
                        type=bool,
                        default=params_config["is_freezing_explaniee_model"])
    parser.add_argument('--explainer-model-n-first-layers-to-freeze',
                        type=int,
                        default=params_config["explainer_model_n_first_layers_to_freeze"])
    parser.add_argument('--is-clamp-between-0-to-1', type=bool, default=params_config["is_clamp_between_0_to_1"])
    parser.add_argument('--is-competitive-method-transforms',
                        type=bool,
                        default=params_config["is_competitive_method_transforms"])
    parser.add_argument('--plot-path', type=str, default=params_config["plot_path"])
    parser.add_argument('--default-root-dir', type=str, default=params_config["default_root_dir"])
    parser.add_argument('--mask-loss', type=str, default=params_config["mask_loss"])
    parser.add_argument('--train-n-label-sample', type=str, default=params_config["train_n_label_sample"])
    parser.add_argument('--lr', type=float, default=params_config["lr"])
    parser.add_argument('--start-epoch-to-evaluate', type=int, default=params_config["start_epoch_to_evaluate"])
    parser.add_argument('--n-batches-to-visualize', type=int, default=params_config["n_batches_to_visualize"])
    parser.add_argument('--is-ce-neg', type=str, default=params_config["is_ce_neg"])
    parser.add_argument('--activation-function', type=str, default=params_config["activation_function"])
    parser.add_argument('--use-logits-only', type=bool, default=params_config["use_logits_only"])
    parser.add_argument('--img-size', type=int, default=params_config["img_size"])
    parser.add_argument('--patch-size', type=int, default=params_config["patch_size"])
    parser.add_argument('--evaluation-experiment-folder-name',
                        type=str,
                        default=params_config["evaluation_experiment_folder_name"])

    args = parser.parse_args()

    EXPLAINEE_MODEL_NAME, EXPLAINER_MODEL_NAME = MODEL_ALIAS_MAPPING[args.explainee_model_name], \
                                                 MODEL_ALIAS_MAPPING[args.explainer_model_name]

    IS_EXPLANIEE_CONVNET = True if EXPLAINEE_MODEL_NAME in CONVNET_MODELS_BY_NAME.keys() else False
    IS_EXPLAINER_CONVNET = True if EXPLAINER_MODEL_NAME in CONVNET_MODELS_BY_NAME.keys() else False

    loss_multipliers = get_loss_multipliers(normalize=False,
                                            mask_loss_mul=args.mask_loss_mul,
                                            prediction_loss_mul=args.prediction_loss_mul)

    args.train_model_by_target_gt_class = False
    target_or_predicted_model = "predicted"

    CKPT_PATH, IMG_SIZE, PATCH_SIZE, MASK_LOSS_MUL, CHECKPOINT_EPOCH_IDX, BASE_CKPT_MODEL_AUC = get_backbone_details(
        explainer_model_name=args.explainer_model_name,
        explainee_model_name=args.explainee_model_name,
        target_or_predicted_model=target_or_predicted_model,
    )

    ic(CKPT_PATH)
    ic(args.mask_loss_mul)
    ic(args.prediction_loss_mul)

    args.batch_size = 32

    test_img_trans, test_img_trans_only_resize, test_lbl_trans = init_get_normalize_and_transform() if not IS_EXPLANIEE_CONVNET else init_get_normalize_and_transform(
        mean=CONVENT_NORMALIZATION_MEAN, std=CONVNET_NORMALIZATION_STD)
    ds = Imagenet_Segmentation(IMAGENET_SEG_PATH,
                               batch_size=args.batch_size,
                               transform=test_img_trans,
                               transform_resize=test_img_trans_only_resize, target_transform=test_lbl_trans)

    model_for_classification_image, model_for_mask_generation, feature_extractor = load_explainer_explaniee_models_and_feature_extractor(
        explainee_model_name=EXPLAINEE_MODEL_NAME,
        explainer_model_name=EXPLAINER_MODEL_NAME,
        activation_function=args.activation_function,
        img_size=args.img_size,
    )

    warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(
        n_epochs=args.n_epochs,
        train_samples_length=len(list(Path(IMAGENET_VAL_IMAGES_FOLDER_PATH).iterdir())),
        batch_size=args.batch_size,
    )

    metric = IoU(2, ignore_index=-1)

    model = OptImageClassificationWithTokenClassificationModelSegmentation(
        model_for_classification_image=model_for_classification_image,
        model_for_mask_generation=model_for_mask_generation,
        plot_path='',
        warmup_steps=warmup_steps,
        total_training_steps=total_training_steps,
        best_auc_objects_path='',
        checkpoint_epoch_idx=CHECKPOINT_EPOCH_IDX,
        best_auc_plot_path='',
        run_base_model_only=args.RUN_BASE_MODEL,
        model_runtype='test',
        experiment_path='exp_name',
        is_explainer_convnet=IS_EXPLAINER_CONVNET,
        is_explainee_convnet=IS_EXPLANIEE_CONVNET,
        lr=args.lr,
        n_epochs=args.n_epochs,
        start_epoch_to_evaluate=args.start_epoch_to_evaluate,
        n_batches_to_visualize=args.n_batches_to_visualize,
        mask_loss=args.mask_loss,
        mask_loss_mul=args.mask_loss_mul,
        prediction_loss_mul=args.prediction_loss_mul,
        activation_function=args.activation_function,
        train_model_by_target_gt_class=args.train_model_by_target_gt_class,
        use_logits_only=args.use_logits_only,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        is_ce_neg=args.is_ce_neg,
        verbose=args.verbose,
    )

    model = freeze_multitask_model(
        model=model,
        is_freezing_explaniee_model=args.is_freezing_explaniee_model,
        explainer_model_n_first_layers_to_freeze=args.explainer_model_n_first_layers_to_freeze,
        is_explainer_convnet=IS_EXPLAINER_CONVNET,
    )

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)
    data_module = ImageSegOptDataModuleSegmentation(train_data_loader=dl)
    trainer = pl.Trainer(
        logger=[],
        accelerator='gpu',
        gpus=1,
        devices=1,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=100,
        max_epochs=args.n_epochs,
        resume_from_checkpoint=CKPT_PATH,
        enable_progress_bar=True,
        enable_checkpointing=False,
        default_root_dir=args.default_root_dir,
        weights_summary=None
    )

    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)

    mIoU, pixAcc, mAp, mF1 = model.seg_results['mIoU'], model.seg_results['pixAcc'], model.seg_results['mAp'], \
                             model.seg_results['mF1']
    print_segmentation_results(pixAcc=pixAcc, mAp=mAp, mIoU=mIoU, mF1=mF1)
