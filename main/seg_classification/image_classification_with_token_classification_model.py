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
from transformers import get_linear_schedule_with_warmup

from config import config
from evaluation.evaluation_utils import patch_score_to_image
from evaluation.perturbation_tests.seg_cls_perturbation_tests import (
    run_perturbation_test,
)
from feature_extractor import ViTFeatureExtractor
from main.seg_classification.output_dataclasses.image_classification_with_token_classification_model_output import \
    ImageClassificationWithTokenClassificationModelOutput
from main.seg_classification.output_dataclasses.lossloss import LossLoss
from main.seg_classification.output_dataclasses.lossloss_output import LossLossOutput
from main.seg_classification.seg_cls_utils import prediction_loss, l1_loss, encourage_token_mask_to_prior_loss
from utils import save_obj_to_disk
from vit_utils import visu, get_loss_multipliers
from models.modeling_vit_patch_classification import ViTForMaskGeneration
from transformers import ViTForImageClassification
from transformers.modeling_outputs import SequenceClassifierOutput

pl.seed_everything(config["general"]["seed"])
vit_config = config["vit"]
loss_config = vit_config["seg_cls"]["loss"]


class ImageClassificationWithTokenClassificationModel(pl.LightningModule):
    def __init__(
            self,
            vit_for_classification_image: ViTForImageClassification,
            vit_for_patch_classification: ViTForMaskGeneration,
            warmup_steps: int,
            total_training_steps: int,
            feature_extractor: ViTFeatureExtractor,
            plot_path,
            experiment_path,
            is_clamp_between_0_to_1: bool = True,
            criterion: LossLoss = LossLoss(),
            n_classes: int = 1000,
            batch_size: int = 8,
            p: int = 1
    ):
        super().__init__()
        self.vit_for_classification_image = vit_for_classification_image
        self.vit_for_patch_classification = vit_for_patch_classification
        self.criterion = criterion
        self.n_classes = n_classes
        self.n_warmup_steps = warmup_steps
        self.n_training_steps = total_training_steps
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        self.is_clamp_between_0_to_1 = is_clamp_between_0_to_1
        self.plot_path = plot_path
        self.p = p
        self.experiment_path = experiment_path

    def normalize_mask_values(self, mask, is_clamp_between_0_to_1: bool):
        if is_clamp_between_0_to_1:
            norm_mask = torch.clamp(mask, min=0, max=1)
        else:
            norm_mask = (mask - mask.min()) / (mask.max() - mask.min())
        return norm_mask

    def normalize_image(self, tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        return tensor

    def forward(self, inputs, image_resized, target_class=None) -> ImageClassificationWithTokenClassificationModelOutput:
        vit_cls_output = self.vit_for_classification_image(inputs)
        interpolated_mask, tokens_mask = self.vit_for_patch_classification(inputs)
        if vit_config["activation_function"]:
            interpolated_mask_normalized = interpolated_mask
        else:
            interpolated_mask_normalized = self.normalize_mask_values(mask=interpolated_mask.clone(),
                                                                      is_clamp_between_0_to_1=self.is_clamp_between_0_to_1)

        masked_image = image_resized * interpolated_mask_normalized
        masked_image_inputs = self.normalize_image(masked_image)
        vit_masked_output: SequenceClassifierOutput = self.vit_for_classification_image(masked_image_inputs)
        vit_masked_output_logits = vit_masked_output.logits

        if loss_config['is_ce_neg']:
            masked_neg_image = image_resized * (1 - interpolated_mask_normalized)
            masked_neg_image_inputs = self.normalize_image(masked_neg_image)
            vit_masked_neg_output: SequenceClassifierOutput = self.vit_for_classification_image(masked_neg_image_inputs)

        vit_masked_neg_output_logits = None if not loss_config['is_ce_neg'] else vit_masked_neg_output.logits

        lossloss_output = self.criterion(
            output=vit_masked_output_logits, neg_output=vit_masked_neg_output_logits, target=vit_cls_output.logits,
            tokens_mask=tokens_mask, target_class=target_class
        )

        return ImageClassificationWithTokenClassificationModelOutput(
            lossloss_output=lossloss_output,
            vit_masked_output=vit_masked_output,
            interpolated_mask=interpolated_mask,
            masked_image=masked_image,
            tokens_mask=tokens_mask,
        )

    def training_step(self, batch, batch_idx):
        inputs = batch["pixel_values"].squeeze(1)
        resized_and_normalized_image = batch["resized_and_normalized_image"]
        image_resized = batch["image"]
        target_class = batch["target_class"]
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
        }

    def validation_step(self, batch, batch_idx):
        inputs = batch["pixel_values"].squeeze(1)
        resized_and_normalized_image = batch["resized_and_normalized_image"]
        image_resized = batch["image"]
        target_class = batch["target_class"]
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
        }

    def training_epoch_end(self, outputs):
        loss = torch.mean(torch.stack([output["loss"] for output in outputs]))
        pred_loss = torch.mean(torch.stack([output["pred_loss"] for output in outputs]))
        mask_loss = torch.mean(torch.stack([output["mask_loss"] for output in outputs]))
        pred_loss_mul = torch.mean(torch.stack([output["pred_loss_mul"] for output in outputs]))
        mask_loss_mul = torch.mean(torch.stack([output["mask_loss_mul"] for output in outputs]))

        self.log("train/loss", loss, prog_bar=True, logger=True)
        self.log("train/prediction_loss", pred_loss, prog_bar=True, logger=True)
        self.log("train/mask_loss", mask_loss, prog_bar=True, logger=True)
        self.log("train/prediction_loss_mul", pred_loss_mul, prog_bar=True, logger=True)
        self.log("train/mask_loss_mul", mask_loss_mul, prog_bar=True, logger=True)
        self._visualize_outputs(
            outputs, stage="train", n_batches=vit_config["n_batches_to_visualize"], epoch_idx=self.current_epoch
        )

    def validation_epoch_end(self, outputs):
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
        epoch_auc = -1
        if self.current_epoch >= vit_config["start_epoch_to_evaluate"]:
            epoch_auc = run_perturbation_test(
                model=self.vit_for_classification_image,
                outputs=outputs,
                stage="val",
                epoch_idx=self.current_epoch,
                experiment_path=self.experiment_path,
            )

        self.log("val/epoch_auc", epoch_auc, prog_bar=True, logger=True)
        return {"loss": loss}

    def mask_patches_to_image_scores(self, patches_mask):
        images_mask = []
        for mask in patches_mask:
            images_mask.append(patch_score_to_image(transformer_attribution=mask, output_2d_tensor=False))
        images_mask = torch.stack(images_mask).squeeze(1)
        return images_mask

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=vit_config["lr"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step"))

    def _visualize_outputs(self, outputs, stage: str, epoch_idx: int, n_batches: int = None):
        """
        :param outputs: batched outputs
        """
        n_batches = n_batches if n_batches is not None else len(outputs)
        epoch_path = Path(self.plot_path, stage, f"epoch_{str(epoch_idx)}")
        print(epoch_path)
        if not epoch_path.exists():
            epoch_path.mkdir(exist_ok=True, parents=True)
        for batch_idx, output in enumerate(outputs[:n_batches]):
            for idx, (image, mask) in enumerate(
                    zip(output["resized_and_normalized_image"].detach().cpu(), output["patches_mask"].detach().cpu())):
                visu(
                    original_image=image,
                    transformer_attribution=mask,
                    file_name=Path(epoch_path, f"{str(batch_idx)}_{str(idx)}").resolve(),
                )


from matplotlib import pyplot as plt
from torch import Tensor


def show_mask(mask):  # [1, 1, 224, 224]
    mask = mask if len(mask.shape) == 3 else mask.squeeze(0)
    _ = plt.imshow(mask.squeeze(0).cpu().detach())
    plt.show()


def plot_image(image) -> None:  # [1,3,224,224] or [3,224,224]
    image = image if len(image.shape) == 3 else image.squeeze(0)
    plt.imshow(image.cpu().detach().permute(1, 2, 0))
    plt.show();
