from pathlib import Path

from utils.consts import PLTE_CHECKPOINTS_PATH

VIT_BASE_224_VIT_BASE_224_PREDICTED_CKPT_PATH = Path(PLTE_CHECKPOINTS_PATH, "vit_base",
                                                     "pLTE_vit_base_224_predicted_best_auc__epoch=27_val_epoch_auc=0.ckpt")
VIT_BASE_224_VIT_BASE_224_TARGET_CKPT_PATH = Path(PLTE_CHECKPOINTS_PATH, "vit_base",
                                                  "pLTE_vit_base_224_target_best_auc__epoch=55_val_epoch_auc=0.ckpt")

VIT_SMALL_224_VIT_SMALL_224_PREDICTED_CKPT_PATH = Path(PLTE_CHECKPOINTS_PATH, "vit_small",
                                                       "pLTE_vit_small_224_predicted_best_auc__epoch=3_val_epoch_auc=0.ckpt")
VIT_SMALL_224_VIT_SMALL_224_TRAGET_CKPT_PATH = Path(PLTE_CHECKPOINTS_PATH, "vit_small",
                                                    "pLTE_vit_small_224_target_best_auc__epoch=11_val_epoch_auc=0.ckpt")
RESNET_RESNET_TARGET_CKPT_PATH = Path(
    "/raid/yuvalas/checkpoints/target/ARGPARSE_explanier_resnet__explaniee_resnet__train_uni_True_val_unif_True_activation_sigmoid_pred_1_mask_l_bce_50__train_n_samples_1000_lr_0.002__bs_32_by_target_gt__True/epoch=251_val/epoch_auc=18.475.ckpt")
RESNET_RESNET_PREDICTED_CKPT_PATH_200__18_95 = Path(
    "/raid/yuvalas/checkpoints/predicted/ARGPARSE_explanier_resnet__explaniee_resnet__train_uni_True_val_unif_True_activation_sigmoid_pred_1_mask_l_bce_50__train_n_samples_1000_lr_0.002__bs_32_by_target_gt__False/epoch=204_val/epoch_auc=18.950.ckpt")

DENSENET_DENSENET_TARGET_CKPT_PATH = Path(
    "/raid/yuvalas/checkpoints/target/ARGPARSE_explanier_densenet__explaniee_densenet__train_uni_True_val_unif_True_activation_sigmoid_pred_1_mask_l_bce_50__train_n_samples_1000_lr_0.002__bs_32_by_target_gt__True/epoch=286_val/epoch_auc=18.565.ckpt")
DENSENET_DENSENET_PREDICTED_CKPT_PATH_194__18_655 = Path(
    "/raid/yuvalas/checkpoints/predicted/ARGPARSE_explanier_densenet__explaniee_densenet__train_uni_True_val_unif_True_activation_sigmoid_pred_1_mask_l_bce_50__train_n_samples_1000_lr_0.002__bs_32_by_target_gt__False/epoch=194_val/epoch_auc=18.655.ckpt")

BACKBONE_DETAILS = {
    "google/vit-base-patch16-224": {
        "ckpt_path": {"target": VIT_BASE_224_VIT_BASE_224_TARGET_CKPT_PATH,
                      "predicted": VIT_BASE_224_VIT_BASE_224_PREDICTED_CKPT_PATH},
        "img_size": 224,
        "patch_size": 16,
        "mask_loss": 50,
        "experiment_base_path": {
            "target": "/raid/yuvalas/experiments/target/direct_opt_ckpt_56_auc_15.925_model_google_vit-base-patch16-224_train_uni_True_val_unif_True_activation_sigmoid_pred_1_mask_l_bce_50__train_n_samples_1000_lr_0.002__bs_32__layers_freezed_0_by_target_gt__True",
            "predicted": "/raid/yuvalas/experiments/predicted/train_1000/direct_opt_ckpt_27_model_google_vit-base-patch16-224_train_uni_True_val_unif_True_activation_sigmoid__norm_by_max_p_False_pred_1_mask_l_bce_50__train_n_samples_1000_lr_0.002_mlp_classifier_True__bs_32",
        },
    },
    "WinKawaks/vit-small-patch16-224": {
        "ckpt_path": {"target": VIT_SMALL_224_VIT_SMALL_224_TRAGET_CKPT_PATH,
                      "predicted": VIT_SMALL_224_VIT_SMALL_224_PREDICTED_CKPT_PATH},
        "img_size": 224,
        "patch_size": 16,
        "mask_loss": 30,
        "experiment_base_path": {
            "target": "/raid/yuvalas/experiments/target/direct_opt_ckpt_12_auc_14.855_model_WinKawaks_vit-small-patch16-224_train_uni_True_val_unif_True_activation_sigmoid_pred_1_mask_l_bce_30__train_n_samples_1000_lr_0.002__bs_32__layers_freezed_0_by_target_gt__True",
            "predicted": "/raid/yuvalas/experiments/predicted/direct_opt_ckpt_4_auc_16.95_model_WinKawaks_vit-small-patch16-224_train_uni_True_val_unif_True_activation_sigmoid_pred_1_mask_l_bce_30__train_n_samples_1000_lr_0.002__bs_32__layers_freezed_0_by_target_gt__False",
        },
    }
}

EXPLAINER_EXPLAINEE_BACKBONE_DETAILS = {  # key: explainer_name-explainee_name
    "vit_base_224-vit_base_224": {
        "ckpt_path": {"target": VIT_BASE_224_VIT_BASE_224_TARGET_CKPT_PATH,
                      "predicted": VIT_BASE_224_VIT_BASE_224_PREDICTED_CKPT_PATH},
        "img_size": 224,
        "patch_size": 16,
        "mask_loss": 50,
        "explainer": "vit_base_224",
        "explainee": "vit_base_224",
        "experiment_base_path": {
            "target": "/raid/yuvalas/experiments/target/direct_opt_ckpt_56_auc_15.925_model_google_vit-base-patch16-224_train_uni_True_val_unif_True_activation_sigmoid_pred_1_mask_l_bce_50__train_n_samples_1000_lr_0.002__bs_32__layers_freezed_0_by_target_gt__True",
            "predicted": "/raid/yuvalas/experiments/predicted/train_1000/direct_opt_ckpt_27_model_google_vit-base-patch16-224_train_uni_True_val_unif_True_activation_sigmoid__norm_by_max_p_False_pred_1_mask_l_bce_50__train_n_samples_1000_lr_0.002_mlp_classifier_True__bs_32",
        },
    },
    "vit_small_224-vit_small_224": {
        "ckpt_path": {"target": VIT_SMALL_224_VIT_SMALL_224_TRAGET_CKPT_PATH,
                      "predicted": VIT_SMALL_224_VIT_SMALL_224_PREDICTED_CKPT_PATH},
        "img_size": 224,
        "patch_size": 16,
        "mask_loss": 30,
        "explainer": "vit_small_224",
        "explainee": "vit_small_224",
        "experiment_base_path": {
            "target": "/raid/yuvalas/experiments/target/direct_opt_ckpt_12_auc_14.855_model_WinKawaks_vit-small-patch16-224_train_uni_True_val_unif_True_activation_sigmoid_pred_1_mask_l_bce_30__train_n_samples_1000_lr_0.002__bs_32__layers_freezed_0_by_target_gt__True",
            "predicted": "/raid/yuvalas/experiments/predicted/direct_opt_ckpt_4_auc_16.95_model_WinKawaks_vit-small-patch16-224_train_uni_True_val_unif_True_activation_sigmoid_pred_1_mask_l_bce_30__train_n_samples_1000_lr_0.002__bs_32__layers_freezed_0_by_target_gt__False",
        },
    },
    "densenet-densenet": {
        "ckpt_path": {"target": DENSENET_DENSENET_TARGET_CKPT_PATH,
                      "predicted": DENSENET_DENSENET_PREDICTED_CKPT_PATH_234__18_86},
        "img_size": 224,
        "patch_size": None,
        "mask_loss": 50,
        "explainer": "densenet",
        "explainee": "densenet",
        "experiment_base_path": {
            "target": None,
            "predicted": None,
        },
    },
    "resnet-resnet": {
        "ckpt_path": {"target": RESNET_RESNET_TARGET_CKPT_PATH,
                      "predicted": RESNET_RESNET_PREDICTED_CKPT_PATH_200__18_95},
        "img_size": 224,
        "patch_size": None,
        "mask_loss": 50,
        "explainer": "resnet",
        "explainee": "resnet",
        "experiment_base_path": {
            "target": None,
            "predicted": None,
        },
    },
    "vit_base_224-resnet": {
        "ckpt_path": {"target": None, "predicted": None},
        "img_size": 224,
        "patch_size": None,
        "mask_loss": 50,
        "explainer": "vit_base_224",
        "explainee": "resnet",
        "experiment_base_path": {
            "target": None,
            "predicted": None,
        },
    },
    "vit_base_224-densenet": {
        "ckpt_path": {"target": None, "predicted": None},
        "img_size": 224,
        "patch_size": None,
        "mask_loss": 50,
        "explainer": "vit_base_224",
        "explainee": "densenet",
        "experiment_base_path": {
            "target": None,
            "predicted": None,
        },
    },
    "resnet-vit_base_224": {
        "ckpt_path": {"target": None, "predicted": None},
        "img_size": 224,
        "patch_size": None,
        "mask_loss": 50,
        "explainer": "resnet",
        "explainee": "vit_base_224",
        "experiment_base_path": {
            "target": None,
            "predicted": None,
        },
    },
    "densenet-vit_base_224": {
        "ckpt_path": {"target": None, "predicted": None},
        "img_size": 224,
        "patch_size": None,
        "mask_loss": 50,
        "explainer": "densenet",
        "explainee": "vit_base_224",
        "experiment_base_path": {
            "target": None,
            "predicted": None,
        },
    },

}
