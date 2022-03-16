from pathlib import Path

import numpy as np
import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import h5py
from feature_extractor import ViTFeatureExtractor
import argparse
from torchvision.datasets import ImageNet

from evaluation.evaluation_utils import patch_score_to_image, normalize, _remove_file_if_exists
from main.temp_softmax_opt import temp_softmax_optimization
from utils.consts import DATA_PATH, EXPERIMENTS_FOLDER_PATH
from config import config
from vit_utils import load_feature_extractor_and_vit_model, create_folder
from torch import nn

vit_config = config['vit']
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


def compute_saliency_and_save(results_path: Path, feature_extractor: ViTFeatureExtractor, vit_model: nn.Module):
    first = True
    _remove_file_if_exists(path=results_path)

    with h5py.File(results_path, 'a') as f:
        data_cam = f.create_dataset('vis',
                                    (1, 1, 224, 224),
                                    maxshape=(None, 1, 224, 224),
                                    dtype=np.float32,
                                    compression="gzip")
        data_image = f.create_dataset('image',
                                      (1, 3, 224, 224),
                                      maxshape=(None, 3, 224, 224),
                                      dtype=np.float32,
                                      compression="gzip")
        data_target = f.create_dataset('target',
                                       (1,),
                                       maxshape=(None,),
                                       dtype=np.int32,
                                       compression="gzip")
        for batch_idx, (data, target) in enumerate(tqdm(sample_loader)):
            if first:
                first = False
                data_cam.resize(data_cam.shape[0] + data.shape[0] - 1, axis=0)
                data_image.resize(data_image.shape[0] + data.shape[0] - 1, axis=0)
                data_target.resize(data_target.shape[0] + data.shape[0] - 1, axis=0)
            else:
                data_cam.resize(data_cam.shape[0] + data.shape[0], axis=0)
                data_image.resize(data_image.shape[0] + data.shape[0], axis=0)
                data_target.resize(data_target.shape[0] + data.shape[0], axis=0)

            # Add data
            data_image[-data.shape[0]:] = data.data.cpu().numpy()
            data_target[-data.shape[0]:] = target.data.cpu().numpy()

            # target = target.to(device)

            # data = normalize(data)
            data = data.to(device)
            # data.requires_grad_()
            cls_attentions_probs = temp_softmax_optimization(vit_model=model, feature_extractor=feature_extractor,
                                                            image=transforms.ToPILImage()(data.reshape(3, vit_config['img_size'],
                                                                               vit_config['img_size'])),
                                                            num_steps=vit_config['num_steps'])
            Res = patch_score_to_image(transformer_attribution=cls_attentions_probs.median(dim=0)[0],
                                       output_2d_tensor=False)  # [1, 1, 224, 224]
            data_cam[-data.shape[0]:] = Res.cpu().detach().numpy()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Train a segmentation')
    # parser.add_argument('--batch-size', type=int,
    #                     default=1,
    #                     help='')
    # parser.add_argument('--method', type=str,
    #                     default='grad_rollout',
    #                     choices=['rollout', 'lrp', 'transformer_attribution', 'full_lrp', 'lrp_last_layer',
    #                              'attn_last_layer', 'attn_gradcam'],
    #                     help='')
    # parser.add_argument('--lmd', type=float,
    #                     default=10,
    #                     help='')
    # parser.add_argument('--vis-class', type=str,
    #                     default='top',
    #                     choices=['top', 'target', 'index'],
    #                     help='')
    # parser.add_argument('--class-id', type=int,
    #                     default=0,
    #                     help='')
    # parser.add_argument('--imagenet-validation-path', type=str,
    #                     required=True,
    #                     help='')
    # args = parser.parse_args()
    #
    # # PATH variables
    # PATH = os.path.dirname(os.path.abspath(__file__)) + '/' # evaluation folder
    # os.makedirs(os.path.join(PATH, 'visualizations'), exist_ok=True)
    #
    # try:
    #     os.remove(os.path.join(PATH, 'visualizations/{}/{}/results.hdf5'.format(args.method,
    #                                                                             args.vis_class)))
    # except OSError:
    #     pass
    #
    # os.makedirs(os.path.join(PATH, 'visualizations/{}'.format(args.method)), exist_ok=True)
    # os.makedirs(os.path.join(PATH, 'visualizations/{}/{}_{}'.format(args.method,
    #                                                                 args.vis_class,
    #                                                                 args.class_id)), exist_ok=True)
    # args.method_dir = os.path.join(PATH, 'visualizations/{}/{}_{}'.format(args.method,
    #                                                                       args.vis_class,
    #                                                                       args.class_id))
    #

    #
    # # Model
    experiment_path = create_folder(Path(EXPERIMENTS_FOLDER_PATH, 'test'))
    results_path = Path(experiment_path, 'results.hdf5')
    feature_extractor, model = load_feature_extractor_and_vit_model(vit_config=vit_config, model_type='vit-for-dino')

    # Dataset loader for sample images
    transform = transforms.Compose([
        transforms.Resize((vit_config['img_size'], vit_config['img_size'])),
        transforms.ToTensor(),
    ])
    val_imagenet_ds = ImageNet(str(DATA_PATH), split='val', transform=transform)
    imagenet_ds = torch.utils.data.Subset(val_imagenet_ds,
                                          list(range(vit_config['evaluation']['num_samples_to_evaluate'])))
    sample_loader = DataLoader(
        imagenet_ds,
        batch_size=vit_config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    compute_saliency_and_save(results_path=results_path, feature_extractor=feature_extractor, vit_model=model)
