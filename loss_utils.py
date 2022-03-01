import wandb
from torch.functional import F
from vit_utils import *
from config import config
from icecream import ic

vit_config = config['vit']
bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='mean')
ce_loss = nn.CrossEntropyLoss(reduction='mean')


def prediction_loss_plus_bce_turn_off_patches_loss(output: Tensor, target: Tensor, x_attention: Tensor,
                                                   iteration_idx: int) -> Tensor:
    """
    Loss between the original prediction's distribution of the model and the prediction's distribution of the new model
    + average of the BCE of the x * self-attention
    """
    target_off_patches = torch.zeros_like(x_attention)
    prediction_loss = ce_loss(output, torch.argmax(target).unsqueeze(0))  # maximize the pred to original model
    bce_turn_off_patches_loss = bce_with_logits_loss(x_attention, target_off_patches)  # turn off patches
    print(f'prediction_loss: {prediction_loss}, bce_turn_off_patches_loss: {bce_turn_off_patches_loss}')
    loss = prediction_loss + bce_turn_off_patches_loss
    return loss

def entropy(p_dist: Tensor) -> Tensor:
    # return sum([-p * torch.log2(p) if p > 0 else 0 for p in p_dist])
    return torch.sum(torch.nan_to_num_(-torch.log2(p_dist) * p_dist, nan=0.0))


def log(loss, x_attention, output, target, sampled_binary_patches=None, kl_loss=None, l1_loss=None, entropy_loss=None,
        prediction_loss=None, other_loss=None, contrastive_class_idx: int=None) -> None:
    target_class_idx = contrastive_class_idx if contrastive_class_idx is not None else torch.argmax(F.softmax(target, dim=-1)).item()
    if vit_config['log']:
        wandb.log({"loss": loss,
                   "other_loss": other_loss if not None else 0,
                   "kl_loss": kl_loss if not None else 0,
                   "l1_loss": l1_loss if not None else 0,
                   "entropy_loss": entropy_loss if not None else 0,
                   "prediction_loss": prediction_loss if not None else 0,
                   'correct_class_pred': F.softmax(output)[0][target_class_idx],
                   'correct_class_logit': output[0][target_class_idx]
                   # 'num_of_non-zero_x_sampled_values': len(torch.where(sampled_binary_patches)[0]) if sampled_binary_patches is not None else None,
                   # 'num_of_non-negative-x_attention_values': len(torch.where(nn.functional.relu(x_attention))[0])
                   })
    print(
        f'pred_loss: {prediction_loss}, kl_loss: {kl_loss}, l1_loss: {l1_loss}, entropy_loss: {entropy_loss}, correct_class_logit: {output[0][target_class_idx]}, other_loss: {other_loss}')
        # f'pred_loss: {prediction_loss}, kl_loss: {kl_loss}, l1_loss: {l1_loss}, entropy_loss: {entropy_loss}, correct_class_logit: {output[0][torch.argmax(F.softmax(target)).item()]}, num_of_non-zero_x_sampled_values: {len(torch.where(sampled_binary_patches)[0]) if sampled_binary_patches is not None else None}, num_of_non-negative_x_attention_values: {len(torch.where(nn.functional.relu(x_attention))[0])}')




def is_iteration_to_action(iteration_idx: int, action:str= 'print') -> bool:
    """
    :param action: 'print' / 'save'
    """
    return vit_config['verbose'] and (
            iteration_idx % vit_config[f'{action}_every'] == 0 or iteration_idx == vit_config['num_steps'] - 1)


def print_objective_every(a: Tensor, b: Tensor, iteration_idx: int, output: Tensor, target: Tensor) -> None:
    if is_iteration_to_action(iteration_idx=iteration_idx, action='print'):
        ic(iteration_idx, a.item(), b.item())
        ic(F.softmax(output)[0][torch.argmax(F.softmax(target)).item()].item())
