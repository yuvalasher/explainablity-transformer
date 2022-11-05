VIT_BASE_224_PREDICTED_CKPT_PATH = "/raid/yuvalas/checkpoints/predicted/train_1000/vit-base-patch16-224_train_uni_True_val_unif_True_activation_sigmoid__norm_by_max_p_False_pred_1_mask_l_bce_50__train_n_samples_1000_lr_0.002_mlp_classifier_True__bs_32/None/checkpoints/epoch=27_val/epoch_auc=18.545.ckpt"
VIT_BASE_224_TARGET_CKPT_PATH = "/raid/yuvalas/checkpoints/target/lightning_logs/600_samples/checkpoints/epoch=55_val/epoch_auc=15.925.ckpt"

VIT_BASE_224_TARGET_CKPT_PATH_16_155_1000_samples = "/raid/yuvalas/checkpoints/target/lightning_logs/version_12/checkpoints/epoch=47_val/epoch_auc=16.155.ckpt"

VIT_SMALL_224_PREDICTED_CKPT_PATH = "/raid/yuvalas/checkpoints/target/lightning_logs/version_14/checkpoints/epoch=9_val/epoch_auc=15.490.ckpt"
VIT_SMALL_224_TRAGET_CKPT_PATH = "/raid/yuvalas/checkpoints/predicted/lightning_logs/version_2/checkpoints/epoch=8_val/epoch_auc=15.620.ckpt"

# VIT_BASE_384_CKPT_PATH = "/home/yuvalas/explainability/research/checkpoints/token_classification/model_google_vit-base-patch16-384_train_uni_True_val_unif_True_activation_sigmoid__norm_by_max_p_False_pred_1_mask_l_bce_70__train_n_samples_1000_lr_0.002_mlp_classifier_True__bs_16/None/checkpoints/epoch=9_val/epoch_auc=21.750.ckpt"

VIT_BACKBONE_DETAILS = {
    "google/vit-base-patch16-224": {
        "ckpt_path": {"target": VIT_BASE_224_TARGET_CKPT_PATH, "predicted": VIT_BASE_224_PREDICTED_CKPT_PATH},
        "img_size": 224,
        "patch_size": 16,
        "mask_loss": 50,
        "experiment_base_path": {
            "target": "/raid/yuvalas/experiments/target/direct_opt_ckpt_56_auc_15.925_model_google_vit-base-patch16-224_train_uni_True_val_unif_True_activation_sigmoid_pred_1_mask_l_bce_50__train_n_samples_1000_lr_0.002__bs_32__layers_freezed_0_by_target_gt__True",
            "potential_target_16_55": "/raid/yuvalas/experiments/target/direct_opt_ckpt_48_auc_16.155_model_google_vit-base-patch16-224_train_uni_True_val_unif_True_activation_sigmoid_pred_1_mask_l_bce_50__train_n_samples_1000_lr_0.002__bs_32__layers_freezed_0_by_target_gt__True",
            "predicted": "/raid/yuvalas/experiments/predicted/train_1000/direct_opt_ckpt_27_model_google_vit-base-patch16-224_train_uni_True_val_unif_True_activation_sigmoid__norm_by_max_p_False_pred_1_mask_l_bce_50__train_n_samples_1000_lr_0.002_mlp_classifier_True__bs_32",
        },
    },
    "WinKawaks/vit-small-patch16-224": {
        "ckpt_path": {"target": VIT_SMALL_224_TRAGET_CKPT_PATH, "predicted": VIT_SMALL_224_PREDICTED_CKPT_PATH},
        "img_size": 224,
        "patch_size": 16,
        "mask_loss": 50,
        "experiment_base_path": {
            "target": "/raid/yuvalas/experiments/target/direct_opt_ckpt_9_auc_15.62_model_WinKawaks_vit-small-patch16-224_train_uni_True_val_unif_True_activation_sigmoid_pred_1_mask_l_bce_50__train_n_samples_1000_lr_0.002__bs_32__layers_freezed_0_by_target_gt__True",
            "predicted": "/raid/yuvalas/experiments/predicted/direct_opt_ckpt_10_auc_15.49_model_WinKawaks_vit-small-patch16-224_train_uni_True_val_unif_True_activation_sigmoid_pred_1_mask_l_bce_50__train_n_samples_1000_lr_0.002__bs_32__layers_freezed_0_by_target_gt__False",
        },
    },
}

# "google/vit-base-patch16-384": {"ckpt_path": "", "img_size": 384, "patch_size": 16, "mask_loss": 70}
