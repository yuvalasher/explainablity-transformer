VIT_BASE_224_PREDICTED_CKPT_PATH = "/raid/yuvalas/checkpoints/predicted/train_1000/vit-base-patch16-224_train_uni_True_val_unif_True_activation_sigmoid__norm_by_max_p_False_pred_1_mask_l_bce_50__train_n_samples_1000_lr_0.002_mlp_classifier_True__bs_32/None/checkpoints/epoch=27_val/epoch_auc=18.545.ckpt"
VIT_BASE_224_TARGET_CKPT_PATH = "/raid/yuvalas/checkpoints/target/lightning_logs/600_samples/checkpoints/epoch=55_val/epoch_auc=15.925.ckpt"

VIT_SMALL_224_PREDICTED_CKPT_PATH = "/raid/yuvalas/checkpoints/target/lightning_logs/version_14/checkpoints/epoch=9_val/epoch_auc=15.490.ckpt"
VIT_SMALL_224_TRAGET_CKPT_PATH = "/raid/yuvalas/checkpoints/predicted/lightning_logs/version_2/checkpoints/epoch=8_val/epoch_auc=15.620.ckpt"

# VIT_BASE_384_CKPT_PATH = "/home/yuvalas/explainability/research/checkpoints/token_classification/model_google_vit-base-patch16-384_train_uni_True_val_unif_True_activation_sigmoid__norm_by_max_p_False_pred_1_mask_l_bce_70__train_n_samples_1000_lr_0.002_mlp_classifier_True__bs_16/None/checkpoints/epoch=9_val/epoch_auc=21.750.ckpt"

VIT_BACKBONE_DETAILS = {
    "google/vit-base-patch16-224": {
        "ckpt_path": {"target": VIT_BASE_224_TARGET_CKPT_PATH, "predicted": VIT_BASE_224_PREDICTED_CKPT_PATH},
        "img_size": 224,
        "patch_size": 16,
        "mask_loss": 50},
    "WinKawaks/vit-small-patch16-224": {
        "ckpt_path": {"target": VIT_SMALL_224_TRAGET_CKPT_PATH, "predicted": VIT_SMALL_224_PREDICTED_CKPT_PATH},
        "img_size": 224,
        "patch_size": 16,
        "mask_loss": 50},
    "google/vit-base-patch16-384": {"ckpt_path": "", "img_size": 384, "patch_size": 16,
                                    "mask_loss": 70}
}
