from pathlib import Path

import numpy as np
import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import h5py

import argparse
from torchvision.datasets import ImageNet

from evaluation.evaluation_utils import patch_score_to_image, normalize
from utils.consts import DATA_PATH, EXPERIMENTS_FOLDER_PATH
from config import config
from vit_utils import load_feature_extractor_and_vit_model, get_attention_probs_by_layer_of_the_CLS

vit_config = config['vit']
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


def compute_saliency_and_save(results_path):
    first = True
    try:
        os.remove(results_path)
    except FileNotFoundError:
        pass

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

            target = target.to(device)

            data = normalize(data)
            data = data.to(device)
            # data.requires_grad_()

            inputs = feature_extractor(images=data.reshape(3, vit_config['img_size'], vit_config['img_size']),
                                       return_tensors="pt")
            output = model(**inputs)
            cls_attention_probs = get_attention_probs_by_layer_of_the_CLS(model=model, layer=-1)  # TODO
            """
            # TODO - Make the optimization here and take the relevant iteration !
            """
            Res = patch_score_to_image(transformer_attribution=cls_attention_probs.median(dim=0)[0],
                                       output_2d_tensor=False)  # [1, 1, 224, 224]
            data_cam[-data.shape[0]:] = Res.cpu().numpy()


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
    experiment_path = Path(EXPERIMENTS_FOLDER_PATH, 'test')
    results_path = Path(experiment_path, 'results.hdf5')
    feature_extractor, model = load_feature_extractor_and_vit_model(vit_config=vit_config, model_type='vit-for-dino')

    # Dataset loader for sample images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    val_imagenet_ds = ImageNet(str(DATA_PATH), split='val', transform=transform)
    imagenet_ds = torch.utils.data.Subset(val_imagenet_ds, list(range(0, 20)))
    sample_loader = DataLoader(
        imagenet_ds,
        batch_size=vit_config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=0
    )

    compute_saliency_and_save(results_path)
