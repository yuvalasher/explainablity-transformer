import os

from matplotlib import pyplot as plt
from torch import Tensor

import numpy as np
from icecream import ic
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Callable

import pytorch_lightning as pl

from config import config

from evaluation.perturbation_tests.seg_cls_perturbation_tests import (save_best_auc_objects_to_disk, run_perturbation_test_opt)

from feature_extractor import ViTFeatureExtractor
from main.seg_classification.image_classification_with_token_classification_model import \
    ImageClassificationWithTokenClassificationModel

from main.seg_classification.output_dataclasses.lossloss import LossLoss

from main.seg_classification.seg_cls_consts import AUC_STOP_VALUE

from vit_utils import visu
from models.modeling_vit_patch_classification import ViTForMaskGeneration
from transformers import ViTForImageClassification

pl.seed_everything(config["general"]["seed"])
vit_config = config["vit"]
loss_config = vit_config["seg_cls"]["loss"]


class OptImageClassificationWithTokenClassificationModel(ImageClassificationWithTokenClassificationModel):
    def __init__(
            self,
            vit_for_classification_image: ViTForImageClassification,
            vit_for_patch_classification: ViTForMaskGeneration,
            warmup_steps: int,
            total_training_steps: int,
            feature_extractor: ViTFeatureExtractor,
            plot_path,
            best_auc_objects_path: str,
            best_auc_plot_path: str,
            checkpoint_epoch_idx: int,
            is_clamp_between_0_to_1: bool = True,
            run_base_model_only: bool = False,
            criterion: LossLoss = LossLoss(),
            n_classes: int = 1000,
            batch_size: int = 8,
    ):
        super().__init__(vit_for_classification_image=vit_for_classification_image,
                         vit_for_patch_classification=vit_for_patch_classification,
                         warmup_steps=warmup_steps,
                         total_training_steps=total_training_steps,
                         feature_extractor=feature_extractor,
                         plot_path=plot_path,
                         is_clamp_between_0_to_1=is_clamp_between_0_to_1,
                         criterion=criterion,
                         n_classes=n_classes,
                         batch_size=batch_size,
                         experiment_path=Path(""))
        self.best_auc_objects_path = best_auc_objects_path
        self.best_auc_plot_path = best_auc_plot_path
        self.best_auc = None
        self.best_auc_epoch = None
        self.best_auc_vis = None
        self.checkpoint_epoch_idx = checkpoint_epoch_idx
        self.image_idx = None
        self.auc_by_epoch = None
        self.run_base_model_only = run_base_model_only

    def init_auc(self) -> None:
        self.best_auc = np.inf
        self.best_auc_epoch = 0
        self.best_auc_vis = None
        self.auc_by_epoch = []
        self.image_idx = len(os.listdir(self.best_auc_objects_path))

    def training_step(self, batch, batch_idx):
        inputs = batch["pixel_values"].squeeze(1)
        resized_and_normalized_image = batch["resized_and_normalized_image"]
        image_resized = batch["image"]
        target_class = batch["target_class"]

        if self.current_epoch == self.checkpoint_epoch_idx:
            self.init_auc()
        output = self.forward(inputs=inputs, image_resized=image_resized, target_class=target_class)
        images_mask = self.mask_patches_to_image_scores(output.tokens_mask)

        return {
            "loss": output.lossloss_output.loss,
            "pred_loss": output.lossloss_output.pred_loss,
            "pred_loss_mul": output.lossloss_output.prediction_loss_multiplied,
            "mask_loss": output.lossloss_output.mask_loss,
            "mask_loss_mul": output.lossloss_output.mask_loss_multiplied,
            "resized_and_normalized_image": resized_and_normalized_image,
            "target_class": target_class,
            "image_mask": images_mask,
            "image_resized": image_resized,
            "patches_mask": output.tokens_mask,
            "auc": self.best_auc
        }

    def validation_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs):
        auc = run_perturbation_test_opt(
            model=self.vit_for_classification_image,
            outputs=outputs,
            stage="train",
            epoch_idx=self.current_epoch,
        )
        if self.best_auc is None or auc < self.best_auc:
            self.best_auc = auc
            self.best_auc_epoch = self.current_epoch
            self.best_auc_vis = outputs[0]["image_mask"]
            self.best_auc_image = outputs[0]["image_resized"]

            save_best_auc_objects_to_disk(path=Path(f"{self.best_auc_objects_path}", f"{str(self.image_idx)}.pkl"),
                                          auc=auc,
                                          vis=self.best_auc_vis,
                                          original_image=self.best_auc_image,
                                          epoch_idx=self.current_epoch,
                                          )
            if self.run_base_model_only or auc < AUC_STOP_VALUE:
                outputs[0]['auc'] = auc
                self.trainer.should_stop = True

        if self.current_epoch == vit_config['n_epochs'] - 1:
            self.trainer.should_stop = True


    def validation_epoch_end(self, outputs):
        pass

    def visualize_images_by_outputs(self, outputs):
        image = outputs[0]["resized_and_normalized_image"].detach().cpu()
        mask = outputs[0]["patches_mask"].detach().cpu()
        auc = outputs[0]['auc']
        image = image if len(image.shape) == 3 else image.squeeze(0)
        mask = mask if len(mask.shape) == 3 else mask.squeeze(0)
        visu(
            original_image=image,
            transformer_attribution=mask,
            file_name=Path(self.best_auc_plot_path, f"{str(self.image_idx)}__{self.current_epoch}__AUC_{round(auc,0)}").resolve(),
        )