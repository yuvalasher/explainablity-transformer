from typing import Union, Any, List
from pathlib import Path, WindowsPath
import os
import pickle
import torch


def load_obj(path: Union[str, WindowsPath, Path]) -> Any:
    if type(path) == str and path[-4:] != '.pkl':
        path += '.pkl'
    elif type(path) == WindowsPath and path.suffix != '.pkl':
        path = path.with_suffix('.pkl')

    with open(path, 'rb') as f:
        return pickle.load(f)


def get_iteration_idx_of_minimum_loss(losses: List) -> int:
    return torch.argmin(torch.tensor(losses)).item()

def get_top_k_mimimum_values_indices(array: List[float], k: int = 5):
    return torch.topk(torch.tensor(array), k=min(k, len(array)), largest=False)[1]

path = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\Tesis\Explainability NLP\explainablity-transformer\research\plots\head_layer_temp_softmax_lr0_003_temp_1+l1_0+kl_loss_0+entropy_loss_1000+pred_loss_10"
k = 50

for idx, image_name in enumerate(os.listdir(path)):
    p = Path(path, image_name)
    print(f'Image: {image_name}')
    losses = load_obj(path=Path(p, 'losses'))
    total_losses = load_obj(path=Path(p, 'total_losses'))
    print(f'Minimum Pred Losses k iterations at: {get_top_k_mimimum_values_indices(array=losses, k=k)}')
    print(f'Minimum Total Losses k iterations at: {get_top_k_mimimum_values_indices(array=total_losses, k=k)}')
    if 'contrastive' in os.listdir(Path(path, image_name)):
        contrastive_path = Path(path, image_name, 'contrastive')
        losses = load_obj(path=Path(contrastive_path, 'losses'))
        total_losses = load_obj(path=Path(contrastive_path, 'total_losses'))
    print(f'Contrastive Minimum Pred Losses k iterations at: {get_top_k_mimimum_values_indices(array=losses, k=k)}')
    print(f'Contrastive Minimum Total Losses k iterations at: {get_top_k_mimimum_values_indices(array=total_losses, k=k)}')
    print('---------------------------------------------------------------------------------------')

