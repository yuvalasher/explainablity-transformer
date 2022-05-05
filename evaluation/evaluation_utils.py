import pickle
from pathlib import Path, WindowsPath
from typing import Tuple, Union, Any
import torch
from torch import Tensor
from sklearn.metrics import auc
import os
import numpy as np


def get_iteration_idx_of_minimum_loss(path) -> int:
    losses = load_obj_from_path(path=Path(path, 'objects' 'losses'))
    return torch.argmin(torch.tensor(losses)).item()


def get_tokens_mask_by_iteration_idx(path, iteration_idx: int) -> Tensor:
    return load_obj_from_path(path=Path(path, 'objects', 'tokens_mask'))[iteration_idx]


def load_tokens_mask(path, iteration_idx: int = None) -> Tuple[int, Tensor]:
    if iteration_idx is None:
        iteration_idx = get_iteration_idx_of_minimum_loss(path=path)
        print(f'Minimum prediction loss at iteration: {iteration_idx}')
    else:
        print(f'Get tokens mask of iteration: {iteration_idx}')
    tokens_mask = get_tokens_mask_by_iteration_idx(path=path, iteration_idx=iteration_idx)
    return iteration_idx, tokens_mask


def load_obj_from_path(path: Union[str, WindowsPath, Path]) -> Any:
    if type(path) == str and path[-4:] != '.pkl':
        path += '.pkl'
    elif type(path) == WindowsPath and path.suffix != '.pkl':
        path = path.with_suffix('.pkl')

    with open(path, 'rb') as f:
        return pickle.load(f)


def patch_score_to_image(transformer_attribution: Tensor, output_2d_tensor: bool = True) -> Tensor:
    """
    Convert Patch scores ([196]) to image size tesnor [224, 224]
    :param transformer_attribution: Tensor with score of each patch in the picture
    :return:
    """
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    if output_2d_tensor:
        transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
            transformer_attribution.max() - transformer_attribution.min())
    return transformer_attribution


def normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def calculate_auc(mean_accuracy_by_step: np.ndarray) -> float:
    return auc(x=np.arange(0, 1, 0.1), y=mean_accuracy_by_step)


def _remove_file_if_exists(path: Path) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
