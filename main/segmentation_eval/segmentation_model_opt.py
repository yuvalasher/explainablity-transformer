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
from utils import save_obj_to_disk
from utils.metrices import batch_pix_accuracy, batch_intersection_union, get_ap_scores, get_f1_scores
from vit_utils import visu, get_loss_multipliers
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
AUC_STOP_VALUE = 6.0


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

    def __post_init__(self):
        loss_multipliers = get_loss_multipliers(loss_config=loss_config)
        self.mask_loss = loss_config["mask_loss"]
        self.prediction_loss_mul = loss_multipliers["prediction_loss_mul"]
        self.mask_loss_mul = loss_multipliers["mask_loss_mul"]
        print(f"loss multipliers: {self.mask_loss_mul}; {self.prediction_loss_mul}")

    def __call__(self, output: Tensor, target: Tensor, tokens_mask: Tensor,
                 neg_output: Tensor = None) -> LossLossOutput:
        """
        Objective 1 - Keep the classification as original with as much as dark tokens
        This will be applied on the token classification by encourage the sigmoid to go to zero & CE with the original

        Loss between the original prediction's distribution of the model and the prediction's distribution of the new model
        + average of the BCE of the x * self-attention
        """
        if self.mask_loss == "bce":
            mask_loss = encourage_token_mask_to_prior_loss(tokens_mask=tokens_mask, prior=0)
        if self.mask_loss == "l1":
            mask_loss = l1_loss(tokens_mask)
        if self.mask_loss == "entropy_softmax":
            assert vit_config['activation_function'] == 'softmax', \
                "The activation_function must be softmax!!"
            mask_loss = self.entropy_loss(mask_loss, tokens_mask)

        pred_pos_loss = prediction_loss(output=output, target=target)
        pred_loss = pred_pos_loss
        if loss_config['is_ce_neg']:
            pred_neg_loss = -1 * prediction_loss(output=neg_output, target=target)
            pred_loss = (pred_pos_loss + pred_neg_loss) / 2

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

    def entropy_loss(self, mask_loss, tokens_mask):
        tokens_mask_reshape = tokens_mask.reshape(tokens_mask.shape[0],
                                                  -1)  # From (32,1,14,14) --> (32,196) - easy for compute entropy.
        d = torch.distributions.Categorical(tokens_mask_reshape + 10e-8)
        normalized_entropy = d.entropy() / np.log(d.param_shape[-1])
        mask_loss = normalized_entropy.mean()
        return mask_loss


@dataclass
class ImageClassificationWithTokenClassificationModelOutput:
    lossloss_output: LossLossOutput
    vit_masked_output: SequenceClassifierOutput
    masked_image: Tensor
    interpolated_mask: Tensor
    tokens_mask: Tensor

class OptImageClassificationWithTokenClassificationModel_Segmentation(pl.LightningModule):
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
            best_auc_objects_path: str,
            best_auc_plot_path: str,
            checkpoint_epoch_idx: int,
            model_runtype: str,
            criterion: LossLoss = LossLoss(),
            emb_size: int = 768,
            n_classes: int = 1000,
            n_patches: int = 196,
            batch_size: int = 8,
            run_base_model_only: bool = False,

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
        self.best_auc_objects_path = best_auc_objects_path
        self.best_auc_plot_path = best_auc_plot_path
        self.best_auc = None
        self.best_auc_epoch = None
        self.best_auc_vis = None
        self.checkpoint_epoch_idx = checkpoint_epoch_idx
        self.image_idx = None
        self.auc_by_epoch = None
        self.save_best_auc_to_disk = False,
        self.target = None
        self.image_resized = None
        self.run_base_model_only = run_base_model_only
        self.seg_results = None
        self.model_runtype = model_runtype

    def init_auc(self) -> None:
        self.best_auc = np.inf
        self.best_auc_epoch = 0
        self.best_auc_vis = None
        self.auc_by_epoch = []
        # self.image_idx = len(os.listdir(self.best_auc_objects_path))

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

    def forward(self, inputs, image_resized) -> ImageClassificationWithTokenClassificationModelOutput:
        vit_cls_output = self.vit_for_classification_image(inputs)
        interpolated_mask, tokens_mask = self.vit_for_patch_classification(inputs)
        # TODO -
        if vit_config["activation_function"]:
            interpolated_mask_normalized = interpolated_mask
        else:
            interpolated_mask_normalized = self.normalize_mask_values(mask=interpolated_mask.clone(),
                                                                      is_clamp_between_0_to_1=self.is_clamp_between_0_to_1)

        masked_image = image_resized * interpolated_mask_normalized
        masked_image_inputs = self.normalize_image(masked_image)
        vit_masked_output: SequenceClassifierOutput = self.vit_for_classification_image(masked_image_inputs)
        vit_masked_output_logits = vit_masked_output.logits

        ### NEGATIVE MASK
        if loss_config['is_ce_neg']:
            masked_neg_image = image_resized * (1 - interpolated_mask_normalized)
            masked_neg_image_inputs = self.normalize_image(masked_neg_image)
            vit_masked_neg_output: SequenceClassifierOutput = self.vit_for_classification_image(masked_neg_image_inputs)

        vit_masked_neg_output_logits = vit_masked_neg_output.logits if loss_config['is_ce_neg'] else None

        lossloss_output = self.criterion(
            output=vit_masked_output_logits, neg_output=vit_masked_neg_output_logits, target=vit_cls_output.logits,
            tokens_mask=tokens_mask
        )
        return ImageClassificationWithTokenClassificationModelOutput(
            lossloss_output=lossloss_output,
            vit_masked_output=vit_masked_output,
            interpolated_mask=interpolated_mask,
            masked_image=masked_image,
            tokens_mask=tokens_mask,
        )

    def training_step(self, batch, batch_idx):

        if self.model_runtype == 'test':
            self.trainer.should_stop = True
            return

        inputs, target, image_resized = batch
        self.target = target
        original_image = inputs
        self.image_resized = image_resized

        if self.current_epoch == self.checkpoint_epoch_idx:
            self.init_auc()

        output = self.forward(inputs, image_resized=image_resized)
        images_mask = self.mask_patches_to_image_scores(output.tokens_mask)

        return {
            "loss": output.lossloss_output.loss,
            "pred_loss": output.lossloss_output.pred_loss,
            "pred_loss_mul": output.lossloss_output.prediction_loss_multiplied,
            "mask_loss": output.lossloss_output.mask_loss,
            "mask_loss_mul": output.lossloss_output.mask_loss_multiplied,
            "original_image": original_image,
            "image_mask": images_mask,
            "image_resized": image_resized,
            "patches_mask": output.tokens_mask,
        }

    def training_epoch_end(self, outputs):
        if self.model_runtype == 'test':
            self.trainer.should_stop = True
            return

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

            if self.save_best_auc_to_disk:
                save_best_auc_objects_to_disk(path=Path(f"{self.best_auc_objects_path}", f"{str(self.image_idx)}.pkl"),
                                              auc=auc,
                                              vis=self.best_auc_vis,
                                              original_image=self.best_auc_image,
                                              epoch_idx=self.current_epoch,
                                              )
            if self.run_base_model_only or auc < AUC_STOP_VALUE:
                self.trainer.should_stop = True

        if self.current_epoch == vit_config['n_epochs'] - 1:
            self.trainer.should_stop = True

    def test_step(self, batch, batch_idx):
        inputs, target, image_resized = batch

        self.target = target

        self.image_resized = image_resized

        output = self.forward(inputs, image_resized)

        images_mask = self.mask_patches_to_image_scores(output.tokens_mask)
        # plt.imshow(images_mask.squeeze(0).squeeze(0).cpu().detach())
        # plt.title('TEST')
        # plt.show()
        # plt.imshow(image_resized.squeeze(0).cpu().detach().permute(1, 2, 0))
        # plt.title('TEST')
        # plt.show()

        outputs = {'images_mask': images_mask,
                   'target': target,
                   'image_resized': image_resized}  # If i want to save more things like orignal_img and etc.

        return outputs

    def test_epoch_end(self, outputs):
        total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
        total_ap, total_f1 = [], []
        predictions, targets = [], []

        for idx, val in tqdm(enumerate(outputs), position=0, leave=True, total=len(outputs)):
            Res_batch, target_batch, image_resized_batch = val['images_mask'], val['target'], val['image_resized']

            correct, labeled, inter, union, ap, f1, pred, target = self.eval_results_per_bacth(Res_batch,
                                                                                               q=-1,
                                                                                               labels=target_batch,
                                                                                               image=image_resized_batch)
            predictions.append(pred)
            targets.append(target)
            total_correct += correct.astype('int64')
            total_label += labeled.astype('int64')
            total_inter += inter.astype('int64')
            total_union += union.astype('int64')
            ap = np.pad(ap, (0, self.batch_size - len(ap)), 'constant')
            f1 = np.pad(f1, (0, self.batch_size - len(ap)), 'constant')
            total_ap += [ap]
            total_f1 += [f1]
            pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
            IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
            mIoU = IoU.mean()
            mAp = np.mean(total_ap)
            mF1 = np.mean(total_f1)

        self.seg_results = {"mIoU": mIoU, 'pixAcc': pixAcc, 'mAp': mAp, 'mF1': mF1}
        return

    def eval_results_per_res(self, Res, image, labels, q=-1):

        Res = (Res - Res.min()) / (Res.max() - Res.min())

        if q == -1:
            ret = Res.mean()
        else:
            ret = torch.quantile(Res, q=q)

        Res_1 = Res.gt(ret).type(Res.type())
        Res_0 = Res.le(ret).type(Res.type())

        Res_1_AP = Res
        Res_0_AP = 1 - Res

        Res_1[Res_1 != Res_1] = 0
        Res_0[Res_0 != Res_0] = 0
        Res_1_AP[Res_1_AP != Res_1_AP] = 0
        Res_0_AP[Res_0_AP != Res_0_AP] = 0

        # TEST
        pred = Res.clamp(min=0.0) / Res.max()  # args.thr instead of 0.0
        pred = pred.view(-1).data.cpu().numpy()
        target = labels.view(-1).data.cpu().numpy()
        # print("target", target.shape)

        output = torch.cat((Res_0, Res_1), 1)
        output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)

        # Evaluate Segmentation
        batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
        batch_ap, batch_f1 = 0, 0

        # Segmentation resutls
        correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels)  # labels should be [224,224]
        inter, union = batch_intersection_union(output[0].data.cpu(), labels, 2)
        batch_correct += correct
        batch_label += labeled
        batch_inter += inter
        batch_union += union
        # print("output", output.shape)
        # print("ap labels", labels.shape)
        # ap = np.nan_to_num(get_ap_scores(output, labels))
        ap = np.nan_to_num(get_ap_scores(output_AP, labels))
        f1 = np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0]))
        batch_ap += ap
        batch_f1 += f1

        return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, pred, target

    def eval_results_per_bacth(self, Res, image, labels, q=-1):

        # vit_output: SequenceClassifierOutput = self.vit_for_classification_image(self.normalize_image(image))
        # gt_idx = vit_output.logits.argmax(dim=1)
        # q_arr = np.arange(0, 1, 0.05)
        # th_torch = torch.zeros_like(gt_idx)
        # for idx in range(self.batch_size):
        #     Res_flat = torch.flatten(Res[idx]).sort().values
        #     for q in q_arr:
        #
        #         th = torch.quantile(Res_flat, q=q)
        #         th_mask = Res[idx] > th
        #         new_mask = Res[idx] * th_mask
        #         # new_mask = Res[idx].clamp(min=th)
        #         # plt.imshow(new_mask.squeeze().data.cpu().numpy())
        #         # plt.title(f'th = {th} and q= {q}')
        #         # plt.show()
        #         masked_image = image[idx] * new_mask
        #         masked_image_norm = (masked_image - masked_image.min()) / (masked_image.max() - masked_image.min())
        #         # plt.imshow(masked_image_norm.permute(1, 2, 0).cpu().detach())
        #         # plt.title(f'image with mask = {q}')
        #         # plt.show()
        #
        #         masked_image = masked_image.unsqueeze(0)
        #         ####### CHECK WHAT IS GT _ AND IF NORMALIZE CHANGED THE INPUT!
        #         masked_image_inputs = self.normalize_image(masked_image)
        #         vit_masked_output: SequenceClassifierOutput = self.vit_for_classification_image(masked_image_inputs)
        #         pred_class = torch.softmax(vit_masked_output.logits, dim=1).argmax().item()
        #         if pred_class != gt_idx[idx]:
        #             th_torch[idx] = th
        #             # TODO - check taking the prev th (before breaking the class).
        #             break

        Res = (Res - Res.min()) / (Res.max() - Res.min())

        # ret = th_torch
        if q == -1:
            ret = Res.mean()
        else:
            ret = torch.quantile(Res, q=q)

        Res_1 = Res.gt(ret).type(Res.type())
        Res_0 = Res.le(ret).type(Res.type())
        Res_1_AP = Res
        Res_0_AP = 1 - Res
        Res_1[Res_1 != Res_1] = 0
        Res_0[Res_0 != Res_0] = 0
        Res_1_AP[Res_1_AP != Res_1_AP] = 0
        Res_0_AP[Res_0_AP != Res_0_AP] = 0
        # TEST
        pred = Res.clamp(min=0.0) / Res.max()
        pred = pred.view(-1).data.cpu().numpy()
        target = labels.view(-1).data.cpu().numpy()
        output = torch.cat((Res_0, Res_1), 1)
        output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)
        # Evaluate Segmentation
        batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
        batch_ap, batch_f1 = 0, 0
        # Segmentation resutls
        correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
        inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)
        batch_correct += correct
        batch_label += labeled
        batch_inter += inter
        batch_union += union
        # print("output", output.shape)
        # print("ap labels", labels.shape)
        # ap = np.nan_to_num(get_ap_scores(output, labels))
        ap = np.nan_to_num(get_ap_scores(output_AP, labels))
        f1 = np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0]))
        batch_ap += ap
        batch_f1 += f1

        return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, pred, target

    def visualize_images_by_outputs(self, outputs):
        image = outputs[0]["resized_and_normalized_image"].detach().cpu()
        mask = outputs[0]["patches_mask"].detach().cpu()
        image = image if len(image.shape) == 3 else image.squeeze(0)
        mask = mask if len(mask.shape) == 3 else mask.squeeze(0)
        visu(
            original_image=image,
            transformer_attribution=mask,
            file_name=Path(self.best_auc_plot_path, f"{str(self.image_idx)}").resolve(),
        )

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


def show_image_inputs(inputs: Tensor):  # [1, 3, 224, 224]
    inputs = inputs[0] if len(inputs.shape) == 4 else inputs
    _ = plt.imshow(inputs.permute(1, 2, 0))
    plt.show()


def show_mask(mask: Tensor):  # [1, 1, 224, 224]
    _ = plt.imshow(mask[0][0])
    plt.show()
