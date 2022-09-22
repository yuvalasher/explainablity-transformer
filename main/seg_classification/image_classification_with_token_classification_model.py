import numpy as np
from icecream import ic
from pathlib import Path
from typing import Tuple, Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from config import config
from evaluation.evaluation_utils import patch_score_to_image
from evaluation.perturbation_tests.seg_cls_perturbation_tests import eval_perturbation_test, run_perturbation_test
from feature_extractor import ViTFeatureExtractor
from utils import save_obj_to_disk
from vit_utils import visu

pl.seed_everything(config["general"]["seed"])
vit_config = config["vit"]
loss_config = vit_config["seg_cls"]["loss"]

bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction="mean")
ce_loss = nn.CrossEntropyLoss(reduction="mean")

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda and vit_config["gpus"] > 0 else "cpu")


def l1_loss(tokens_mask) -> Tensor:
    return torch.abs(tokens_mask).sum()


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
    bce_encourage_prior_patches_loss = bce_with_logits_loss(tokens_mask,
                                                            target_encourage_patches)  # turn off token masks
    return bce_encourage_prior_patches_loss


def prediction_loss_plus_bce_turn_off_patches_loss(output: Tensor, target: Tensor, tokens_mask: Tensor) -> Tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Objective 1 - Keep the classification as original with as much as dark tokens
    This will be applied on the token classification by encourage the sigmoid to go to zero & CE with the original

    Loss between the original prediction's distribution of the model and the prediction's distribution of the new model
    + average of the BCE of the x * self-attention
    """
    if loss_config['mask_loss'] == 'bce':
        mask_loss = encourage_token_mask_to_prior_loss(tokens_mask=tokens_mask, prior=0)
    else:
        mask_loss = l1_loss(tokens_mask)
    pred_loss = prediction_loss(output=output, target=target)
    prediction_loss_multiplied = loss_config['prediction_loss_mul'] * pred_loss
    mask_loss_multiplied = loss_config['mask_loss_mul'] * mask_loss
    print(
        f'prediction_loss: {pred_loss} * {loss_config["prediction_loss_mul"]}, mask_loss: {mask_loss} * {loss_config["mask_loss_mul"]}')
    loss = prediction_loss_multiplied + mask_loss_multiplied
    return loss, prediction_loss_multiplied, mask_loss_multiplied, pred_loss, mask_loss


class ImageClassificationWithTokenClassificationModel(pl.LightningModule):
    """
    If fine-tuned with non frozen vit, the model should continue training with the original objective (classification)
    TODO - talk to Oren after trying to fine-tune with freezed model
    TODO - read about segmentation in ViT - https://arxiv.org/pdf/2105.05633.pdf
    """

    def __init__(self, vit_for_classification_image, vit_for_patch_classification, warmup_steps: int,
                 total_training_steps: int, feature_extractor: ViTFeatureExtractor, plot_path,
                 criterion: Callable = prediction_loss_plus_bce_turn_off_patches_loss, emb_size: int = 768,
                 n_classes: int = 1000, n_patches: int = 196, batch_size: int = 8, max_perturbation_stage: int = 5):
        super().__init__()
        self.vit_for_classification_image = vit_for_classification_image
        self.vit_for_patch_classification = vit_for_patch_classification
        self.criterion = criterion
        self.n_classes = n_classes
        # self.token_classification = nn.Linear(emb_size, 1)  # regression to a number 0-1 or classification to 2 units
        self.n_warmup_steps = warmup_steps
        self.n_training_steps = total_training_steps
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        self.plot_path = plot_path
        self.max_perturbation_stage = max_perturbation_stage

    def forward(self, inputs, original_image) -> Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        vit_cls_output = self.vit_for_classification_image(inputs)
        patches_mask = self.vit_for_patch_classification(inputs).logits  # [batch_size, n_patches]
        patches_mask_reshaped = patches_mask.reshape(inputs.shape[0], 1, 14, 14)
        interpolated_mask = torch.nn.functional.interpolate(patches_mask_reshaped, scale_factor=16,
                                                            mode='bilinear').cpu().detach()
        masked_image = inputs.cpu().detach() * interpolated_mask
        masked_image_inputs = torch.stack(
            [self.feature_extractor(images=img, return_tensors="pt")['pixel_values'] for img in masked_image]).squeeze(
            1).to(device)
        vit_masked_output = self.vit_for_classification_image(masked_image_inputs)
        loss, prediction_loss_multiplied, mask_loss_multiplied, pred_loss, mask_loss = self.criterion(
            output=vit_masked_output.logits,
            target=vit_cls_output.logits,
            tokens_mask=patches_mask)  # TODO - if won't be regularized, the mask will be all full - sanity check
        return loss, prediction_loss_multiplied, mask_loss_multiplied, vit_masked_output, masked_image_inputs, original_image, patches_mask, pred_loss, mask_loss

    def training_step(self, batch, batch_idx):
        # print(f'Train. batch_idx: {batch_idx}, len_batch: {len(batch["image_name"])}')
        inputs = batch['pixel_values'].squeeze(1)
        original_image = batch['original_transformed_image']
        loss, prediction_loss_multiplied, mask_loss_multiplied, vit_masked_output, masked_image_inputs, original_image, \
        patches_mask, pred_loss, mask_loss = self(inputs, original_image)
        # ic(prediction_loss_multiplied)
        # print(f'******************** epoch_idx: {epoch_idx} ***********')
        # ic(loss, prediction_loss_multiplied, mask_loss_multiplied)
        # print(patches_mask)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_prediction_loss_multiplied", prediction_loss_multiplied, prog_bar=True, logger=True)
        self.log("train_mask_loss_multiplied", mask_loss_multiplied, prog_bar=True, logger=True)
        images_mask = self.mask_patches_to_image_scores(patches_mask)
        return {"loss": loss, "prediction_loss_multiplied": prediction_loss_multiplied,
                "mask_loss_multiplied": mask_loss_multiplied, "predictions": vit_masked_output,
                'masked_image': masked_image_inputs,
                "original_image": original_image, 'patches_mask': patches_mask,
                'image_mask': images_mask
                }

    def mask_patches_to_image_scores(self, patches_mask):
        images_mask = []
        for mask in patches_mask:
            images_mask.append(patch_score_to_image(transformer_attribution=mask, output_2d_tensor=False))
        images_mask = torch.stack(images_mask).squeeze(1)
        return images_mask

    def validation_step(self, batch, batch_idx):
        # print(f'Val. batch_idx: {batch_idx}, len_batch: {len(batch["image_name"])}')
        inputs = batch['pixel_values'].squeeze(1)
        original_image = batch['original_transformed_image']
        loss, prediction_loss_multiplied, mask_loss_multiplied, vit_masked_output, masked_image_inputs, original_image, \
        patches_mask, pred_loss, mask_loss = self(inputs, original_image)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_prediction_loss_multiplied", prediction_loss_multiplied, prog_bar=True, logger=True)
        self.log("val_mask_loss_multiplied", mask_loss_multiplied, prog_bar=True, logger=True)
        images_mask = self.mask_patches_to_image_scores(patches_mask)

        return {"loss": loss, "prediction_loss_multiplied": prediction_loss_multiplied,
                "mask_loss_multiplied": mask_loss_multiplied, "predictions": vit_masked_output,
                'masked_image': masked_image_inputs,
                "original_image": original_image, 'patches_mask': patches_mask, 'image_mask': images_mask}

    def _visualize_outputs(self, outputs, stage: str, epoch_idx: int, n_batches: int = None):
        """
        :param outputs: batched outputs
        """
        n_batches = n_batches if n_batches is not None else len(outputs)
        # print(f'Stage: {stage}; Num of batches visualizing: {n_batches}, len(outputs): {len(outputs)}')
        epoch_path = Path(self.plot_path, stage, f"epoch_{str(epoch_idx)}")
        if not epoch_path.exists():
            epoch_path.mkdir(exist_ok=True, parents=True)
        for batch_idx, output in enumerate(outputs[:n_batches]):
            for idx, (image, mask) in enumerate(
                    zip(output["original_image"].detach().cpu(), output["patches_mask"].detach().cpu())):
                if epoch_idx >= 0 and stage == 'train':
                    # print(idx)
                    # print(f'******************** epoch_idx: {epoch_idx} ***********')
                    # print(mask)
                    # print(1)
                    pass
                visu(original_image=image, transformer_attribution=mask,
                     file_name=Path(epoch_path, f"{str(batch_idx)}_{str(idx)}").resolve())

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=vit_config['lr'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )

    def training_epoch_end(self, outputs):
        print('training_epoch_end')
        run_perturbation_test(feature_extractor=self.feature_extractor, model=self.vit_with_classification_head,
                              max_perturbation_stage=self.max_perturbation_stage, outputs=outputs, stage='train')

    def validation_epoch_end(self, outputs) -> None:
        print('val_epoch_end')
        run_perturbation_test(feature_extractor=self.feature_extractor, model=self.vit_with_classification_head,
                              max_perturbation_stage=self.max_perturbation_stage, outputs=outputs, stage='val')
        # self._visualize_outputs(outputs, stage='val', n_batches=10, epoch_idx=self.current_epoch)
        pass