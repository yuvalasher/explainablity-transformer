from dataclasses import dataclass

from main.seg_classification.output_dataclasses.lossloss_output import LossLossOutput
import torch
from torch import Tensor

from main.seg_classification.seg_cls_utils import encourage_token_mask_to_prior_loss, l1_loss, prediction_loss
from config import config
from vit_utils import get_loss_multipliers
import numpy as np

vit_config = config["vit"]
loss_config = vit_config["seg_cls"]["loss"]


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

    def __call__(self, output: Tensor, target: Tensor, tokens_mask: Tensor, target_class: Tensor,
                 neg_output: Tensor = None) -> LossLossOutput:
        """
        Objective 1 - Keep the classification as original with as much as dark tokens
        This will be applied on the token classification by encourage the sigmoid to go to zero & CE with the original

        Loss between the original prediction's distribution of the model and the prediction's distribution of the new model
        + average of the BCE of the x * self-attention
        """
        if self.mask_loss == "bce":
            mask_loss = encourage_token_mask_to_prior_loss(tokens_mask=tokens_mask, prior=0)
        elif self.mask_loss == "l1":
            mask_loss = l1_loss(tokens_mask)
        elif self.mask_loss == "entropy_softmax":
            assert vit_config['activation_function'] == 'softmax', \
                "The activation_function must be softmax!!"
            mask_loss = self.entropy_loss(tokens_mask)
        else:
            raise (f"Value of self.mask_loss is not recognized")

        pred_pos_loss = prediction_loss(output=output, target=target, target_class=target_class)
        pred_loss = pred_pos_loss
        if loss_config['is_ce_neg']:
            pred_neg_loss = -1 * prediction_loss(output=neg_output, target=target, target_class=target_class)
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

    def entropy_loss(self, tokens_mask: Tensor):
        tokens_mask_reshape = tokens_mask.reshape(tokens_mask.shape[0],
                                                  -1)  # From (32,1,14,14) --> (32,196) - easy for compute entropy.
        d = torch.distributions.Categorical(tokens_mask_reshape + 10e-8)
        normalized_entropy = d.entropy() / np.log(d.param_shape[-1])
        mask_loss = normalized_entropy.mean()
        return mask_loss
