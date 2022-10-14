import pickle
from pathlib import Path

import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


def load_obj(path):
    with open(Path(path), 'rb') as f:
        return pickle.load(f)


def plot_img_with_mask(idx,img_idx):
    path_pkl = f"/home/amiteshel1/Projects/explainablity-transformer-cv/research/experiments/seg_cls/del_fixed_ft_50000_new_model_seg_only_base_new/opt_model/objects_pkl/{idx}.pkl"
    img_path = f"/home/amiteshel1/Projects/explainablity-transformer-cv/run/imagenet/token_to_learn_vgg/experiment_27/results/input/{img_idx}/{img_idx}_input.png"
    image = Image.open(img_path)
    img_tensor = transform(image)
    loaded = load_obj(path_pkl)
    mask = loaded['vis']
    mask = mask.squeeze(0)
    new = img_tensor * mask.cpu()
    plt.imshow(mask.squeeze(0).cpu().detach())
    plt.show()
    plt.imshow(transforms.ToPILImage()(new), interpolation="bicubic")
    auc = loaded['auc']
    plt.title(f'AUC = {auc}')
    plt.show()
    return


plot_img_with_mask(idx=69,img_idx=32)
