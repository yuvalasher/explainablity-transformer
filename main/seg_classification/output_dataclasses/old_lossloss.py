from dataclasses import dataclass
from main.seg_classification.output_dataclasses.lossloss_output import LossLossOutput
import torch
from torch import Tensor
from main.seg_classification.seg_cls_utils import encourage_token_mask_to_prior_loss, l1_loss, prediction_loss
from utils.vit_utils import get_loss_multipliers
import numpy as np


@dataclass
class LossLoss:
    mask_loss: str
    prediction_loss_mul: int
    mask_loss_mul: int

    def __post_init__(self):
        loss_multipliers = get_loss_multipliers(normalize=False,
                                                mask_loss_mul=self.mask_loss_mul,
                                                prediction_loss_mul=self.prediction_loss_mul)
        self.prediction_loss_mul = loss_multipliers["prediction_loss_mul"]
        self.mask_loss_mul = loss_multipliers["mask_loss_mul"]
        print(f"loss multipliers: {self.mask_loss_mul}; {self.prediction_loss_mul}")

    def __call__(self, output: Tensor,
                 target: Tensor,
                 tokens_mask: Tensor,
                 target_class: Tensor,
                 activation_function: str,
                 train_model_by_target_gt_class: bool,
                 use_logits_only: bool,
                 is_ce_neg: bool,
                 neg_output: Tensor = None,
                 use_kl_div: bool = False,
                 confidence_drop_loss: bool = False) -> LossLossOutput:
        # print what use
        print(f"Using mask_loss: {self.mask_loss}, prediction_loss_mul: {self.prediction_loss_mul}, "
              f"mask_loss_mul: {self.mask_loss_mul}, activation_function: {activation_function}, "
              f"train_model_by_target_gt_class: {train_model_by_target_gt_class}, use_logits_only: {use_logits_only}, "
              f"is_ce_neg: {is_ce_neg}, use_kl_div: {use_kl_div}, confidence_drop_loss: {confidence_drop_loss}")
        if self.mask_loss == "bce":
            mask_loss = encourage_token_mask_to_prior_loss(tokens_mask=tokens_mask, prior=0)
        elif self.mask_loss == "l1":
            mask_loss = l1_loss(tokens_mask)
        elif self.mask_loss == "entropy_softmax":
            assert activation_function == 'softmax', \
                "The activation_function must be softmax!!"
            mask_loss = self.entropy_loss(tokens_mask)
        else:
            raise (f"Value of self.mask_loss is not recognized")

        if confidence_drop_loss:
            probs_inpainted = torch.nn.functional.softmax(output, dim=-1)
            probs_original = torch.nn.functional.softmax(neg_output, dim=-1)
            original_conf = probs_original[torch.arange(probs_original.size(0)), target_class]
            inpainted_conf = probs_inpainted[torch.arange(probs_inpainted.size(0)), target_class]
            confidence_drop = original_conf - inpainted_conf
            pred_loss = -confidence_drop.mean()  # encourage bigger drop
        elif use_kl_div:
            log_probs_inpainted = torch.nn.functional.log_softmax(output, dim=-1)
            probs_original = torch.nn.functional.softmax(target, dim=-1)
            kl_div_per_token = torch.nn.functional.kl_div(log_probs_inpainted, probs_original,
                                                          reduction='none')  # shape: [B, C]
            kl_div_per_example = kl_div_per_token.sum(dim=-1)  # shape: [B]
            importance_mask = tokens_mask.detach().float().reshape_as(kl_div_per_example)  # shape: [B]
            pred_pos_loss = (kl_div_per_example * importance_mask).sum() / (importance_mask.sum() + 1e-8)
            pred_loss = pred_pos_loss
        else:
            pred_pos_loss = prediction_loss(output=output,
                                            target=target,
                                            target_class=target_class,
                                            train_model_by_target_gt_class=train_model_by_target_gt_class,
                                            use_logits_only=use_logits_only)
            pred_loss = pred_pos_loss
            if is_ce_neg:
                pred_neg_loss = -1 * prediction_loss(output=neg_output,
                                                     target=target,
                                                     target_class=target_class,
                                                     train_model_by_target_gt_class=train_model_by_target_gt_class,
                                                     use_logits_only=use_logits_only)
                pred_loss = (pred_pos_loss + pred_neg_loss) / 2

        prediction_loss_multiplied = self.prediction_loss_mul * pred_loss
        mask_loss_multiplied = self.mask_loss_mul * mask_loss
        loss = prediction_loss_multiplied + mask_loss_multiplied
        return LossLossOutput(
            loss=loss,
            prediction_loss_multiplied=prediction_loss_multiplied,
            mask_loss_multiplied=mask_loss_multiplied,
            pred_loss=pred_loss,
            mask_loss=mask_loss,
        )

    def entropy_loss(self, tokens_mask: Tensor):
        tokens_mask_reshape = tokens_mask.reshape(tokens_mask.shape[0], -1)
        d = torch.distributions.Categorical(tokens_mask_reshape + 10e-8)
        normalized_entropy = d.entropy() / np.log(d.param_shape[-1])
        mask_loss = normalized_entropy.mean()
        return mask_loss
