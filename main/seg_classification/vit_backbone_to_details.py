VIT_BASE_224_CKPT_PATH = "/home/yuvalas/explainability/research/checkpoints/token_classification/model_google/vit-base-patch16-224_train_uni_True_val_unif_True_activation_sigmoid__norm_by_max_p_False_pred_1_mask_l_bce_50__train_n_samples_6000_lr_0.002_mlp_classifier_True__bs_32/None/checkpoints/epoch=27_val/epoch_auc=18.545.ckpt"
VIT_BASE_384_CKPT_PATH = "/home/yuvalas/explainability/research/checkpoints/token_classification/model_WinKawaks/vit-small-patch16-224_train_uni_True_val_unif_True_activation_sigmoid__norm_by_max_p_False_pred_1_mask_l_bce_30__train_n_samples_6000_lr_0.002_mlp_classifier_True__bs_32/None/checkpoints/epoch=3_val/epoch_auc=16.950.ckpt"
VIT_SMALL_224_CKPT_PATH = "/home/yuvalas/explainability/research/checkpoints/token_classification/model_google_vit-base-patch16-384_train_uni_True_val_unif_True_activation_sigmoid__norm_by_max_p_False_pred_1_mask_l_bce_70__train_n_samples_6000_lr_0.002_mlp_classifier_True__bs_16/None/checkpoints/epoch=9_val/epoch_auc=21.750.ckpt"

VIT_BACKBONE_DETAILS = {
    "google/vit-base-patch16-224": {"ckpt_path": VIT_BASE_224_CKPT_PATH, "img_size": 224, "patch_size": 16},
    "google/vit-base-patch16-384": {"ckpt_path": VIT_BASE_384_CKPT_PATH, "img_size": 384, "patch_size": 16},
    "WinKawaks/vit-small-patch16-224": {"ckpt_path": VIT_SMALL_224_CKPT_PATH, "img_size": 224, "patch_size": 16}
}
