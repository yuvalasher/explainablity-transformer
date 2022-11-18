from pathlib import Path

from utils.consts import PLTE_CHECKPOINTS_PATH

# VIT_BASE_224_PREDICTED_CKPT_PATH = Path(PLTE_CHECKPOINTS_PATH, "vit_base", "pLTE_vit_base_224_predicted_best_auc.ckpt")
# VIT_BASE_224_TARGET_CKPT_PATH = Path(PLTE_CHECKPOINTS_PATH, "vit_base", "pLTE_vit_base_224_target_best_auc_auc.ckpt")
# VIT_SMALL_224_PREDICTED_CKPT_PATH = Path(PLTE_CHECKPOINTS_PATH, "vit_small",
#                                          "pLTE_vit_small_224_predicted_best_auc.ckpt")
# VIT_SMALL_224_TRAGET_CKPT_PATH = Path(PLTE_CHECKPOINTS_PATH, "vit_small", "pLTE_vit_small_target_best_auc.ckpt")

VIT_BASE_224_PREDICTED_CKPT_PATH = "/raid/yuvalas/checkpoints/predicted/train_1000/vit-base-patch16-224_train_uni_True_val_unif_True_activation_sigmoid__norm_by_max_p_False_pred_1_mask_l_bce_50__train_n_samples_1000_lr_0.002_mlp_classifier_True__bs_32/None/checkpoints/epoch=27_val/epoch_auc=18.545.ckpt"
VIT_BASE_224_TARGET_CKPT_PATH = "/raid/yuvalas/checkpoints/target/lightning_logs/600_samples/checkpoints/epoch=55_val/epoch_auc=15.925.ckpt"

VIT_SMALL_224_PREDICTED_CKPT_PATH = "/raid/yuvalas/checkpoints/predicted/train_1000/vit-small-patch16-224_train_uni_True_val_unif_True_activation_sigmoid__norm_by_max_p_False_pred_1_mask_l_bce_30__train_n_samples_6000_lr_0.002_mlp_classifier_True__bs_32/None/checkpoints/epoch=3_val/epoch_auc=16.950.ckpt"
VIT_SMALL_224_TRAGET_CKPT_PATH = "/raid/yuvalas/checkpoints/target/lightning_logs/version_16/checkpoints/epoch=11_val/epoch_auc=14.855.ckpt"

VIT_BACKBONE_DETAILS = {
    "google/vit-base-patch16-224": {
        "ckpt_path": {"target": VIT_BASE_224_TARGET_CKPT_PATH, "predicted": VIT_BASE_224_PREDICTED_CKPT_PATH},
        "img_size": 224,
        "patch_size": 16,
        "mask_loss": 50,
        "experiment_base_path": {
            "target": "/raid/yuvalas/experiments/target/direct_opt_ckpt_56_auc_15.925_model_google_vit-base-patch16-224_train_uni_True_val_unif_True_activation_sigmoid_pred_1_mask_l_bce_50__train_n_samples_1000_lr_0.002__bs_32__layers_freezed_0_by_target_gt__True",
            "predicted": "/raid/yuvalas/experiments/predicted/train_1000/direct_opt_ckpt_27_model_google_vit-base-patch16-224_train_uni_True_val_unif_True_activation_sigmoid__norm_by_max_p_False_pred_1_mask_l_bce_50__train_n_samples_1000_lr_0.002_mlp_classifier_True__bs_32",
        },
    },
    "WinKawaks/vit-small-patch16-224": {
        "ckpt_path": {"target": VIT_SMALL_224_TRAGET_CKPT_PATH, "predicted": VIT_SMALL_224_PREDICTED_CKPT_PATH},
        "img_size": 224,
        "patch_size": 16,
        "mask_loss": 30,
        "experiment_base_path": {
            "target": "/raid/yuvalas/experiments/target/direct_opt_ckpt_12_auc_14.855_model_WinKawaks_vit-small-patch16-224_train_uni_True_val_unif_True_activation_sigmoid_pred_1_mask_l_bce_30__train_n_samples_1000_lr_0.002__bs_32__layers_freezed_0_by_target_gt__True",
            "predicted": "/raid/yuvalas/experiments/predicted/direct_opt_ckpt_4_auc_16.95_model_WinKawaks_vit-small-patch16-224_train_uni_True_val_unif_True_activation_sigmoid_pred_1_mask_l_bce_30__train_n_samples_1000_lr_0.002__bs_32__layers_freezed_0_by_target_gt__False",
        },
    },
}
