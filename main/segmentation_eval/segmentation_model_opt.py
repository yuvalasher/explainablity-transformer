import os
import numpy as np
from pathlib import Path
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from config import config
from evaluation.perturbation_tests.seg_cls_perturbation_tests import run_perturbation_test_opt
from feature_extractor import ViTFeatureExtractor
from main.seg_classification.image_classification_with_token_classification_model import \
    ImageClassificationWithTokenClassificationModel
from main.seg_classification.output_dataclasses.lossloss import LossLoss
from utils.metrices import batch_pix_accuracy, batch_intersection_union, get_ap_scores, get_f1_scores
from vit_utils import visu
from models.modeling_vit_patch_classification import ViTForMaskGeneration
from transformers import ViTForImageClassification
from matplotlib import pyplot as plt
from torch import Tensor
from main.seg_classification.seg_cls_consts import AUC_STOP_VALUE

pl.seed_everything(config["general"]["seed"])
vit_config = config["vit"]
loss_config = vit_config["seg_cls"]["loss"]


class OptImageClassificationWithTokenClassificationModel_Segmentation(ImageClassificationWithTokenClassificationModel):
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
            experiment_path: str,
            checkpoint_epoch_idx: int,
            run_base_model_only: bool = False,
            is_clamp_between_0_to_1: bool = True,
            model_runtype: str = 'N/A',
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
                         experiment_path=experiment_path)
        self.best_auc_objects_path = best_auc_objects_path
        self.best_auc_plot_path = best_auc_plot_path
        self.best_auc = None
        self.best_auc_epoch = None
        self.best_auc_vis = None
        self.checkpoint_epoch_idx = checkpoint_epoch_idx
        self.image_idx = None
        self.auc_by_epoch = None
        self.run_base_model_only = run_base_model_only
        self.model_runtype = model_runtype
        self.target = None
        self.image_resized = None
        self.seg_results = None

    def init_auc(self) -> None:
        self.best_auc = np.inf
        self.best_auc_epoch = 0
        self.best_auc_vis = None
        self.auc_by_epoch = []
        self.image_idx = len(os.listdir(self.best_auc_objects_path))

    def training_step(self, batch, batch_idx):
        if self.model_runtype == 'test':
            self.trainer.should_stop = True
            return

        inputs, target, image_resized = batch
        self.target = target
        original_image = inputs.clone()
        self.image_resized = image_resized

        if self.current_epoch == self.checkpoint_epoch_idx:
            self.init_auc()

        vit_cls_output = self.vit_for_classification_image(inputs)
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
            "target_class": torch.argmax(vit_cls_output.logits, dim=1),
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


    def eval_results_per_bacth(self, Res, image, labels, q=-1):
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
        # Segmentation results
        correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
        inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)
        batch_correct += correct
        batch_label += labeled
        batch_inter += inter
        batch_union += union
        ap = np.nan_to_num(get_ap_scores(output_AP, labels))
        f1 = np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0]))
        batch_ap += ap
        batch_f1 += f1

        return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, pred, target

    def visualize_images_by_outputs(self, outputs):
        image = outputs[0]["resized_and_normalized_image"].detach().cpu()
        mask = outputs[0]["patches_mask"].detach().cpu()
        auc = outputs[0]['auc']
        image = image if len(image.shape) == 3 else image.squeeze(0)
        mask = mask if len(mask.shape) == 3 else mask.squeeze(0)
        visu(
            original_image=image,
            transformer_attribution=mask,
            file_name=Path(self.best_auc_plot_path, f"{str(self.image_idx)}__AUC_{round(auc, 0)}").resolve(),
        )


def show_image_inputs(inputs: Tensor):  # [1, 3, 224, 224]
    inputs = inputs[0] if len(inputs.shape) == 4 else inputs
    _ = plt.imshow(inputs.permute(1, 2, 0))
    plt.show()


def show_mask(mask: Tensor):  # [1, 1, 224, 224]
    _ = plt.imshow(mask[0][0])
    plt.show()
