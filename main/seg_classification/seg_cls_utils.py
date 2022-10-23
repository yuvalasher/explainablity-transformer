import pickle
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import Tensor, nn
from config import config

vit_config = config["vit"]
loss_config = vit_config["seg_cls"]["loss"]

bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction="mean")
ce_loss = nn.CrossEntropyLoss(reduction="mean")


def l1_loss(tokens_mask) -> Tensor:
    # return torch.abs(tokens_mask).sum() # should be limited by batch_size * num_tokens
    return torch.abs(tokens_mask).mean()


def prediction_loss(output, target):
    argmax_target = torch.argmax(target, dim=1)
    if loss_config["use_logits_only"]:
        # return -torch.gather(output, 1, argmax_target.unsqueeze(1)).squeeze(1).sum()
        return -torch.gather(output, 1, argmax_target.unsqueeze(1)).squeeze(1).mean()
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


def load_obj(path) -> Any:
    with open(Path(path), 'rb') as f:
        return pickle.load(f)


def load_pickles_and_calculate_auc(path):
    aucs = []
    listdir = sorted(list(Path(path).iterdir()))
    for pkl_path in listdir:
        # print(pkl_path)
        loaded_obj = load_obj(pkl_path)
        auc = loaded_obj['auc']
        aucs.append(auc)
    # print(f'AUCS: {aucs}')
    print(f"{len(aucs)} samples")
    return np.mean(aucs)