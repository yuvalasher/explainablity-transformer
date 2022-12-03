import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from main.seg_classification.model_types_loading import CONVNET_MODELS_BY_NAME, \
    load_explainer_explaniee_models_and_feature_extractor
from icecream import ic
from main.segmentation_eval.segmentation_utils import print_segmentation_results
from pathlib import Path
from main.segmentation_eval.segmentation_model_opt import \
    OptImageClassificationWithTokenClassificationModelSegmentation
import torch
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from PIL import Image
from main.seg_classification.image_token_data_module_opt_segmentation import ImageSegOptDataModuleSegmentation
from config import config
from utils.iou import IoU
from main.segmentation_eval.imagenet import Imagenet_Segmentation
from vit_utils import get_warmup_steps_and_total_training_steps, \
    get_loss_multipliers, freeze_multitask_model, get_params_from_config, suppress_warnings, get_backbone_details
from utils.consts import IMAGENET_SEG_PATH, IMAGENET_VAL_IMAGES_FOLDER_PATH
import pytorch_lightning as pl
import gc
from PIL import ImageFile

suppress_warnings()
seed_everything(config["general"]["seed"])

ImageFile.LOAD_TRUNCATED_IMAGES = True
gc.collect()

batch_size, n_epochs, is_sampled_train_data_uniformly, is_sampled_val_data_uniformly, \
train_model_by_target_gt_class, is_freezing_explaniee_model, \
explainer_model_n_first_layers_to_freeze, is_clamp_between_0_to_1, enable_checkpointing, \
is_competitive_method_transforms, explainer_model_name, explainee_model_name, plot_path, default_root_dir, \
mask_loss, mask_loss_mul, prediction_loss_mul, lr, start_epoch_to_evaluate, \
n_batches_to_visualize, is_ce_neg, activation_function, n_epochs_to_optimize_stage_b, RUN_BASE_MODEL, \
use_logits_only, VERBOSE, IMG_SIZE, PATCH_SIZE, evaluation_experiment_folder_name, train_n_label_sample, \
val_n_label_sample = get_params_from_config(config_vit=config["vit"])

IS_EXPLANIEE_CONVNET = True if explainee_model_name in CONVNET_MODELS_BY_NAME.keys() else False
IS_EXPLAINER_CONVNET = True if explainer_model_name in CONVNET_MODELS_BY_NAME.keys() else False

loss_multipliers = get_loss_multipliers(normalize=False,
                                        mask_loss_mul=mask_loss_mul,
                                        prediction_loss_mul=prediction_loss_mul)


train_model_by_target_gt_class = False
enable_checkpointing = False
target_or_predicted_model = "predicted"

CKPT_PATH, IMG_SIZE, PATCH_SIZE, MASK_LOSS_MUL, CHECKPOINT_EPOCH_IDX, BASE_CKPT_MODEL_AUC = get_backbone_details(
    explainer_model_name=explainer_model_name,
    explainee_model_name=explainee_model_name,
    target_or_predicted_model=target_or_predicted_model,
)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

ic(CKPT_PATH)
ic(mask_loss_mul)
ic(prediction_loss_mul)


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
    batch_size = 32
    test_img_trans, test_img_trans_only_resize, test_lbl_trans = init_get_normalize_and_trns()
    ds = Imagenet_Segmentation(IMAGENET_SEG_PATH,
                               batch_size=batch_size,
                               transform=test_img_trans,
                               transform_resize=test_img_trans_only_resize, target_transform=test_lbl_trans)

    model_for_classification_image, model_for_mask_generation, feature_extractor = load_explainer_explaniee_models_and_feature_extractor(
        explainee_model_name=explainee_model_name,
        explainer_model_name=explainer_model_name,
        activation_function=activation_function,
        img_size=IMG_SIZE,
    )

    warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(
        n_epochs=n_epochs,
        train_samples_length=len(list(Path(IMAGENET_VAL_IMAGES_FOLDER_PATH).iterdir())),
        batch_size=batch_size,
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
        run_base_model_only=RUN_BASE_MODEL,
        model_runtype='test',
        experiment_path='exp_name',
        is_explainer_convnet=IS_EXPLAINER_CONVNET,
        is_explainee_convnet=IS_EXPLANIEE_CONVNET,
        lr=lr,
        n_epochs=n_epochs,
        start_epoch_to_evaluate=start_epoch_to_evaluate,
        n_batches_to_visualize=n_batches_to_visualize,
        mask_loss=mask_loss,
        mask_loss_mul=mask_loss_mul,
        prediction_loss_mul=prediction_loss_mul,
        activation_function=activation_function,
        train_model_by_target_gt_class=train_model_by_target_gt_class,
        use_logits_only=use_logits_only,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        is_ce_neg=is_ce_neg,
        verbose=VERBOSE,
    )

    model = freeze_multitask_model(
        model=model,
        is_freezing_explaniee_model=is_freezing_explaniee_model,
        explainer_model_n_first_layers_to_freeze=explainer_model_n_first_layers_to_freeze,
        is_explainer_convnet=IS_EXPLAINER_CONVNET,
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
        max_epochs=n_epochs,
        resume_from_checkpoint=CKPT_PATH,
        enable_progress_bar=True,
        enable_checkpointing=False,
        default_root_dir=default_root_dir,
        weights_summary=None
    )

    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)

    mIoU, pixAcc, mAp, mF1 = model.seg_results['mIoU'], model.seg_results['pixAcc'], model.seg_results['mAp'], \
                             model.seg_results['mF1']
    print_segmentation_results(pixAcc=pixAcc, mAp=mAp, mIoU=mIoU, mF1=mF1)
