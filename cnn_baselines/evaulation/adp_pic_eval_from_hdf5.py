import os
from distutils.util import strtobool

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"

from utils.vit_utils import suppress_warnings
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import models
from cnn_baselines.evaulation.imangenet_results_cnn_baselines import ImagenetResults
from main.seg_classification.cnns.cnn_utils import CONVENT_NORMALIZATION_MEAN, CONVNET_NORMALIZATION_STD
import torch
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils.consts import CNN_BASELINES_RESULTS_PATH

suppress_warnings()


def normalize(tensor,
              mean=[0.5, 0.5, 0.5],
              std=[0.5, 0.5, 0.5],
              ):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def eval(imagenet_ds, sample_loader, model, method: str):
    prob_correct_model = np.zeros((len(imagenet_ds, )))
    prob_correct_model_mask = np.zeros((len(imagenet_ds, )))
    model_index = 0

    for batch_idx, (data, vis, target) in enumerate(tqdm(sample_loader)):
        data = data.to(device)
        # plot_image(data[0])

        vis = vis.to(device)
        # show_mask(vis[0])
        # show_mask(data[0] * vis[0])

        target = target.to(device)
        norm_data = normalize(data.clone(),
                              mean=CONVENT_NORMALIZATION_MEAN,
                              std=CONVNET_NORMALIZATION_STD,
                              )

        # Compute model accuracy
        pred = model(norm_data)
        probs = torch.softmax(pred, dim=1)
        target_probs = torch.gather(probs, 1, target[:, None])[:, 0]
        # proba of original image
        prob_correct_model[model_index:model_index + len(target_probs)] = target_probs.data.cpu().numpy()

        # #### ADP PIC
        img_with_mask = data.clone() * vis.clone()
        norm_data_mask = normalize(img_with_mask.clone(),
                                   mean=CONVENT_NORMALIZATION_MEAN,
                                   std=CONVNET_NORMALIZATION_STD,
                                   )
        pred_mask = model(norm_data_mask)
        probs_mask = torch.softmax(pred_mask, dim=1)
        # if batch_idx in [23541//32 - 1, 23541//32 , 23541//32 + 1]:
        #     print(1)
        target_probs_mask = torch.gather(probs_mask, 1, target[:, None])[:, 0]
        # for i in range(20):
        #     show_mask(vis[i])
        #     plot_image(img_with_mask[i], title=f"{method} - Original:{round(target_probs[i].item(),3)}, masked: {round(target_probs_mask[i].item(),3)}")
        prob_correct_model_mask[model_index:model_index + len(target_probs)] = target_probs_mask.data.cpu().numpy()

        model_index += len(target)
        # if batch_idx in [733, 734, 735]:
        #     print(1)

    x = torch.tensor(prob_correct_model)
    y = torch.tensor(prob_correct_model_mask)
    adp = (torch.maximum(x - y, torch.zeros_like(x)) / x).mean() * 100
    pic = torch.where(x < y, 1.0, 0.0).mean() * 100
    print(f"PIC = {pic.item()}")
    print(f"ADP = {adp.item()}")


def preprocess(backbone: str, method: str, is_target: bool):
    runs_dir = Path(CNN_BASELINES_RESULTS_PATH, "visualizations", backbone, method,
                    "target" if is_target else "predicted")
    print(runs_dir)
    imagenet_ds = ImagenetResults(runs_dir)
    if backbone == "resnet101":
        model = models.resnet101(pretrained=True).cuda()
    elif backbone == "densenet":
        model = models.densenet201(pretrained=True).cuda()
    else:
        raise ("Backbone not implemented")
    model.eval()
    sample_loader = DataLoader(
        imagenet_ds,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False,
    )
    return imagenet_ds, sample_loader, model


def plot_image(image, title: str = None):
    plt.imshow(image.cpu().detach().permute(1, 2, 0))
    plt.axis('off');
    if title is not None:
        plt.title(title)
    plt.show();


def show_mask(mask):
    plt.imshow(mask[0].cpu().detach())
    plt.axis('off');
    plt.show();


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    parser = argparse.ArgumentParser(description="Infer")
    parser.add_argument("--batch-size", type=int,
                        default=32,
                        )
    parser.add_argument("--method", type=str,
                        default="lift-cam",
                        choices=["gradcam", "gradcampp", "lift-cam", "fullgrad", "ablation-cam"],
                        )
    parser.add_argument('--backbone', type=str,
                        default='densenet',
                        choices=["resnet101", "densenet"],
                        )
    parser.add_argument("--is-target",
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=False,
                        )

    args = parser.parse_args()
    print(args)
    torch.multiprocessing.set_start_method('spawn')
    imagenet_ds, sample_loader, model = preprocess(backbone=args.backbone, method=args.method, is_target=args.is_target)
    eval(imagenet_ds=imagenet_ds, sample_loader=sample_loader, model=model, method=args.method)
