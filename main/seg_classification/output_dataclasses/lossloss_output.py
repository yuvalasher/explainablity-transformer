from dataclasses import dataclass

from torch import Tensor


@dataclass
class LossLossOutput:
    loss: Tensor
    prediction_loss_multiplied: Tensor
    prediction_neg_loss_multiplied: Tensor
    mask_loss_multiplied: Tensor
    pred_loss: Tensor
    pred_neg_loss: Tensor
    mask_loss: Tensor