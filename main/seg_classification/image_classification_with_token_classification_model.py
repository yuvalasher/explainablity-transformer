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
from utils import save_obj_to_disk
from vit_utils import visu
from models.modeling_vit_patch_classification import ViTForMaskGeneration
from transformers import ViTForImageClassification
from transformers.modeling_outputs import SequenceClassifierOutput

pl.seed_everything(config["general"]["seed"])
vit_config = config["vit"]
loss_config = vit_config["seg_cls"]["loss"]

bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction="mean")
ce_loss = nn.CrossEntropyLoss(reduction="mean")

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda and vit_config["gpus"] > 0 else "cpu")


def l1_loss(tokens_mask) -> Tensor:
    # return torch.abs(tokens_mask).sum() # should be limited by batch_size * num_tokens
    return torch.abs(tokens_mask).mean()


def prediction_loss(output, target):
    argmax_target = torch.argmax(target, dim=1)
    return ce_loss(output, argmax_target)  # maximize the pred to original model


def encourage_token_mask_to_prior_loss(tokens_mask: Tensor, prior: int = 0):
    if prior == 0:
        target_encourage_patches = torch.zeros_like(tokens_mask)
    elif prior == 1:
        target_encourage_patches = torch.ones_like(tokens_mask)
    else:
        raise NotImplementedError
    bce_encourage_prior_patches_loss = bce_with_logits_loss(
        tokens_mask, target_encourage_patches
    )  # turn off token masks
    return bce_encourage_prior_patches_loss


@dataclass
class LossLossOutput:
    loss: Tensor
    prediction_loss_multiplied: Tensor
    mask_loss_multiplied: Tensor
    pred_loss: Tensor
    mask_loss: Tensor


@dataclass
class LossLoss:
    mask_loss: str = loss_config["mask_loss"]
    prediction_loss_mul: float = loss_config["prediction_loss_mul"]
    mask_loss_mul: float = loss_config["mask_loss_mul"]

    def __call__(self, output: Tensor, target: Tensor, tokens_mask: Tensor) -> LossLossOutput:
        """
        Objective 1 - Keep the classification as original with as much as dark tokens
        This will be applied on the token classification by encourage the sigmoid to go to zero & CE with the original

        Loss between the original prediction's distribution of the model and the prediction's distribution of the new model
        + average of the BCE of the x * self-attention
        """
        if self.mask_loss == "bce":
            mask_loss = encourage_token_mask_to_prior_loss(tokens_mask=tokens_mask, prior=0)
        else:
            mask_loss = l1_loss(tokens_mask)

        pred_loss = prediction_loss(output=output, target=target)

        prediction_loss_multiplied = self.prediction_loss_mul * pred_loss
        mask_loss_multiplied = self.mask_loss_mul * mask_loss
        loss = prediction_loss_multiplied + mask_loss_multiplied
        # print(
        #     f'prediction_loss: {pred_loss} * {self.prediction_loss_mul}, mask_loss: {mask_loss} * {self.mask_loss_mul}')
        return LossLossOutput(
            loss=loss,
            prediction_loss_multiplied=prediction_loss_multiplied,
            mask_loss_multiplied=mask_loss_multiplied,
            pred_loss=pred_loss,
            mask_loss=mask_loss,
        )


@dataclass
class ImageClassificationWithTokenClassificationModelOutput:
    lossloss_output: LossLossOutput
    vit_masked_output: SequenceClassifierOutput
    masked_image: Tensor
    interpolated_mask: Tensor
    tokens_mask: Tensor


class ImageClassificationWithTokenClassificationModel(pl.LightningModule):
    """
    If fine-tuned with non frozen vit, the model should continue training with the original objective (classification)
    TODO - talk to Oren after trying to fine-tune with freezed model
    TODO - read about segmentation in ViT - https://arxiv.org/pdf/2105.05633.pdf
    """

    def __init__(
            self,
            vit_for_classification_image: ViTForImageClassification,
            vit_for_patch_classification: ViTForMaskGeneration,
            warmup_steps: int,
            total_training_steps: int,
            feature_extractor: ViTFeatureExtractor,
            plot_path,
            criterion: LossLoss = LossLoss(),
            emb_size: int = 768,
            n_classes: int = 1000,
            n_patches: int = 196,
            batch_size: int = 8,
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
        self.plot_path = plot_path

    def forward(self, inputs) -> ImageClassificationWithTokenClassificationModelOutput:
        vit_cls_output = self.vit_for_classification_image(inputs)
        interpolated_mask, tokens_mask = self.vit_for_patch_classification(inputs)
        masked_image = inputs * interpolated_mask

        vit_masked_output: SequenceClassifierOutput = self.vit_for_classification_image(masked_image)
        lossloss_output = self.criterion(
            output=vit_masked_output.logits, target=vit_cls_output.logits, tokens_mask=tokens_mask
        )  # TODO - if won't be regularized, the mask will be all full - sanity check
        return ImageClassificationWithTokenClassificationModelOutput(
            lossloss_output=lossloss_output,
            vit_masked_output=vit_masked_output,
            interpolated_mask=interpolated_mask,
            masked_image=masked_image,
            tokens_mask=tokens_mask,
        )

    def training_step(self, batch, batch_idx):
        inputs = batch["pixel_values"].squeeze(1)
        original_image = batch["original_transformed_image"]
        output = self.forward(inputs)

        images_mask = self.mask_patches_to_image_scores(output.tokens_mask)
        return {
            "loss": output.lossloss_output.loss,
            "pred_loss": output.lossloss_output.pred_loss,
            "pred_loss_mul": output.lossloss_output.prediction_loss_multiplied,
            "mask_loss": output.lossloss_output.mask_loss,
            "mask_loss_mul": output.lossloss_output.mask_loss_multiplied,
            "original_image": original_image,
            "image_mask": images_mask,
            "patches_mask": output.tokens_mask,
        }

    def validation_step(self, batch, batch_idx):
        inputs = batch["pixel_values"].squeeze(1)
        original_image = batch["original_transformed_image"]
        output = self.forward(inputs)

        images_mask = self.mask_patches_to_image_scores(output.tokens_mask)
        return {
            "loss": output.lossloss_output.loss,
            "pred_loss": output.lossloss_output.pred_loss,
            "pred_loss_mul": output.lossloss_output.prediction_loss_multiplied,
            "mask_loss": output.lossloss_output.mask_loss,
            "mask_loss_mul": output.lossloss_output.mask_loss_multiplied,
            "original_image": original_image,
            "image_mask": images_mask,
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
        if self.current_epoch >= vit_config["start_epoch_to_evaluate"] and self.current_epoch % 4 == 0:
            run_perturbation_test(
                feature_extractor=self.feature_extractor,
                model=self.vit_for_classification_image,
                outputs=outputs,
                stage="train",
                epoch_idx=self.current_epoch,
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
        if self.current_epoch >= vit_config["start_epoch_to_evaluate"]:
            run_perturbation_test(
                feature_extractor=self.feature_extractor,
                model=self.vit_for_classification_image,
                outputs=outputs,
                stage="val",
                epoch_idx=self.current_epoch,
            )
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
                    zip(output["original_image"].detach().cpu(), output["patches_mask"].detach().cpu())
            ):
                visu(
                    original_image=image,
                    transformer_attribution=mask,
                    file_name=Path(epoch_path, f"{str(batch_idx)}_{str(idx)}").resolve(),
                )


from matplotlib import pyplot as plt
from torch import Tensor


def show_image_inputs(inputs: Tensor):  # [1, 3, 224, 224]
    inputs = inputs[0] if len(inputs.shape) == 4 else inputs
    _ = plt.imshow(inputs.permute(1, 2, 0))
    plt.show()


def show_mask(mask: Tensor):  # [1, 1, 224, 224]
    _ = plt.imshow(mask[0][0])
    plt.show()
