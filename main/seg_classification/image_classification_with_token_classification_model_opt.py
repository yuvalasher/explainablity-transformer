import os

from matplotlib import pyplot as plt
from torch import Tensor

import numpy as np
from icecream import ic
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config import config
from evaluation.evaluation_utils import patch_score_to_image
from evaluation.perturbation_tests.seg_cls_perturbation_tests import (
    run_perturbation_test, save_best_auc_objects_to_disk, run_perturbation_test_opt,
)
from feature_extractor import ViTFeatureExtractor
from main.seg_classification.image_classification_with_token_classification_model import \
    ImageClassificationWithTokenClassificationModel
from main.seg_classification.output_dataclasses.image_classification_with_token_classification_model_output import \
    ImageClassificationWithTokenClassificationModelOutput
from main.seg_classification.output_dataclasses.lossloss import LossLoss
from main.seg_classification.output_dataclasses.lossloss_output import LossLossOutput
from main.seg_classification.seg_cls_consts import AUC_STOP_VALUE
from main.seg_classification.seg_cls_utils import l1_loss, prediction_loss, encourage_token_mask_to_prior_loss
from utils import save_obj_to_disk
from utils.metrices import batch_pix_accuracy, batch_intersection_union, get_ap_scores, get_f1_scores
from vit_utils import visu, get_loss_multipliers
from models.modeling_vit_patch_classification import ViTForMaskGeneration
from transformers import ViTForImageClassification
from transformers.modeling_outputs import SequenceClassifierOutput

pl.seed_everything(config["general"]["seed"])
vit_config = config["vit"]
loss_config = vit_config["seg_cls"]["loss"]
device = torch.device(type='cuda', index=config["general"]["gpu_index"])


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
                         batch_size=batch_size)
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

        if self.current_epoch == self.checkpoint_epoch_idx:

            self.init_auc()
            output = self.forward(inputs=inputs, image_resized=image_resized)
            images_mask = self.mask_patches_to_image_scores(output.tokens_mask)
            outputs = [{"image_resized": image_resized, "image_mask": images_mask,
                        "resized_and_normalized_image": resized_and_normalized_image,
                        "patches_mask": output.tokens_mask}]
            auc = run_perturbation_test_opt(
                model=self.vit_for_classification_image,
                outputs=outputs,
                stage="train_step",
                epoch_idx=self.current_epoch,
            )
            # print(f'Basemodel - AUC: {round(auc, 3)} !')
            self.best_auc = auc
            self.best_auc_epoch = self.current_epoch
            self.best_auc_vis = outputs[0]["image_mask"]
            self.best_auc_image = outputs[0]["image_resized"]  # need original as will run perturbation tests on it
            # self.auc_by_epoch.append(auc)

            save_best_auc_objects_to_disk(path=Path(f"{self.best_auc_objects_path}", f"{str(self.image_idx)}.pkl"),
                                          auc=auc,
                                          vis=self.best_auc_vis,
                                          original_image=self.best_auc_image,
                                          epoch_idx=self.current_epoch,
                                          )

            if self.run_base_model_only or auc < AUC_STOP_VALUE:
                # self.visualize_images_by_outputs(outputs=outputs)
                self.trainer.should_stop = True

        else:
            output = self.forward(inputs=inputs, image_resized=image_resized)
            images_mask = self.mask_patches_to_image_scores(output.tokens_mask)

        return {
            "loss": output.lossloss_output.loss,
            "pred_loss": output.lossloss_output.pred_loss,
            "pred_loss_mul": output.lossloss_output.prediction_loss_multiplied,
            "mask_loss": output.lossloss_output.mask_loss,
            "mask_loss_mul": output.lossloss_output.mask_loss_multiplied,
            "resized_and_normalized_image": resized_and_normalized_image,
            "image_mask": images_mask,
            "image_resized": image_resized,
            "patches_mask": output.tokens_mask,
            "auc": self.best_auc
        }

    def validation_step(self, batch, batch_idx):
        inputs = batch["pixel_values"].squeeze(1)
        resized_and_normalized_image = batch["resized_and_normalized_image"]
        image_resized = batch["image"]
        output = self.forward(inputs=inputs, image_resized=image_resized)

        images_mask = self.mask_patches_to_image_scores(output.tokens_mask)
        return {
            "loss": output.lossloss_output.loss,
            "pred_loss": output.lossloss_output.pred_loss,
            "pred_loss_mul": output.lossloss_output.prediction_loss_multiplied,
            "mask_loss": output.lossloss_output.mask_loss,
            "mask_loss_mul": output.lossloss_output.mask_loss_multiplied,
            "resized_and_normalized_image": resized_and_normalized_image,
            "image_mask": images_mask,
            "image_resized": image_resized,
            "patches_mask": output.tokens_mask,
        }

    def training_epoch_end(self, outputs):
        # loss = torch.mean(torch.stack([output["loss"] for output in outputs]))
        # pred_loss = torch.mean(torch.stack([output["pred_loss"] for output in outputs]))
        # mask_loss = torch.mean(torch.stack([output["mask_loss"] for output in outputs]))
        # pred_loss_mul = torch.mean(torch.stack([output["pred_loss_mul"] for output in outputs]))
        # mask_loss_mul = torch.mean(torch.stack([output["mask_loss_mul"] for output in outputs]))
        # self.log("train/loss", loss, prog_bar=True, logger=True)
        # self.log("train/prediction_loss", pred_loss, prog_bar=True, logger=True)
        # self.log("train/mask_loss", mask_loss, prog_bar=True, logger=True)
        # self.log("train/prediction_loss_mul", pred_loss_mul, prog_bar=True, logger=True)
        # self.log("train/mask_loss_mul", mask_loss_mul, prog_bar=True, logger=True)
        # self._visualize_outputs(
        #     outputs, stage="train", n_batches=vit_config["n_batches_to_visualize"], epoch_idx=self.current_epoch
        # )
        auc = run_perturbation_test_opt(
            model=self.vit_for_classification_image,
            outputs=outputs,
            stage="train",
            epoch_idx=self.current_epoch,
        )
        # self.auc_by_epoch.append(auc)
        # print(f"EPOCHEEEE: {self.current_epoch}")
        if self.best_auc is None or auc < self.best_auc:
            # print(f'New Best AUC: {round(auc, 3)} !')
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
                # self.visualize_images_by_outputs(outputs=outputs)
                self.trainer.should_stop = True

        if self.current_epoch == vit_config['n_epochs'] - 1:
            # print(f"AUC by Epoch: {self.auc_by_epoch}")
            # print(f"Best auc: {self.best_auc} by epoch {self.best_auc_epoch}")
            # self.visualize_images_by_outputs(outputs=outputs)
            self.trainer.should_stop = True

    def visualize_images_by_outputs(self, outputs):
        image = outputs[0]["resized_and_normalized_image"].detach().cpu()
        mask = outputs[0]["patches_mask"].detach().cpu()
        auc = outputs[0]['auc']
        image = image if len(image.shape) == 3 else image.squeeze(0)
        mask = mask if len(mask.shape) == 3 else mask.squeeze(0)
        visu(
            original_image=image,
            transformer_attribution=mask,
            file_name=Path(self.best_auc_plot_path, f"{str(self.image_idx)}__AUC_{auc}").resolve(),
        )

    def validation_epoch_end(self, outputs):
        """
        loss = torch.mean(torch.stack([output["loss"] for output in outputs]))
        pred_loss = torch.mean(torch.stack([output["pred_loss"] for output in outputs]))
        mask_loss = torch.mean(torch.stack([output["mask_loss"] for output in outputs]))
        pred_loss_mul = torch.mean(torch.stack([output["pred_loss_mul"] for output in outputs]))
        mask_loss_mul = torch.mean(torch.stack([output["mask_loss_mul"] for output in outputs]))

        self.log("val/loss", loss, prog_bar=True, logger=True)
        self.log("val/prediction_loss", pred_loss, prog_bar=True, logger=True)
        self.log("val/mask_loss", mask_loss, prog_bar=True, logger=True)
        self.log("val/prediction_loss_mul", pred_loss_mul, prog_bar=True, logger=True)
        self.log("val/mask_loss_mul", mask_loss_mul, prog_bar=True, logger=True)
        self._visualize_outputs(
            outputs, stage="val", n_batches=vit_config["n_batches_to_visualize"], epoch_idx=self.current_epoch
        )
        if self.current_epoch >= vit_config["start_epoch_to_evaluate"]:
            run_perturbation_test(
                model=self.vit_for_classification_image,
                outputs=outputs,
                stage="val",
                epoch_idx=self.current_epoch,
            )
        """
        return {"loss": torch.tensor(1)}
