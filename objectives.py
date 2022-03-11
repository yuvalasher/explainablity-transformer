from loss_utils import ce_loss, entropy, log
import torch
from torch import Tensor
from torch.functional import F
from config import config
vit_config = config['vit']
loss_config = vit_config['loss']


def objective_temp_softmax(output: Tensor, target: Tensor, temp: Tensor) -> Tensor:
    prediction_loss = ce_loss(output, torch.argmax(target).unsqueeze(0)) * loss_config['pred_loss_multiplier']
    entropy_loss = entropy(F.softmax(temp, dim=-1)) * loss_config['entropy_loss_multiplier']
    l1_loss = torch.mean(torch.abs(temp)) * loss_config['l1_loss_multiplier']
    print(f'entropy: {entropy_loss}, prediction_loss: {prediction_loss}, l1_loss: {l1_loss}')
    # print(f'entropy: {entropy_loss}, prediction_loss: {prediction_loss}')
    loss = entropy_loss + prediction_loss
    print(f'loss: {loss}, max_temp: {torch.max(temp)}, min_temp: {torch.min(temp)}')

    log(loss=loss, entropy_loss=entropy_loss, l1_loss=l1_loss, prediction_loss=prediction_loss, x_attention=temp, output=output,
        target=target)
    return loss