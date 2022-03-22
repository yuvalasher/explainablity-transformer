from pathlib import Path
from typing import Tuple

import numpy as np
import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import h5py

from datasets.imagenet_dataset import ImageNetDataset
from feature_extractor import ViTFeatureExtractor
import argparse
from torchvision.datasets import ImageNet

from evaluation.evaluation_utils import patch_score_to_image, normalize, _remove_file_if_exists
from main.temp_softmax_opt import temp_softmax_optimization
from utils.consts import DATA_PATH, EXPERIMENTS_FOLDER_PATH, EVALUATION_FOLDER_PATH
from config import config
from vit_utils import load_feature_extractor_and_vit_model, create_folder, read_file
from torch import nn

vit_config = config['vit']
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

VIS_SHAPE: Tuple[int, int, int, int] = (1, 1, 224, 224)
VIS_MAX_SHAPE: Tuple[None, int, int, int] = (None, 1, 224, 224)


def compute_saliency_and_save(results_path: Path, feature_extractor: ViTFeatureExtractor, vit_model: nn.Module):
    _remove_file_if_exists(path=results_path)

    with h5py.File(results_path, 'a') as f:
        data_cam_min_pred_loss = f.create_dataset('vis_min_pred_loss',
                                                  VIS_SHAPE,
                                                  maxshape=VIS_MAX_SHAPE,
                                                  dtype=np.float32,
                                                  compression="gzip")
        data_cam_max_logits = f.create_dataset('vis_max_logits',
                                               VIS_SHAPE,
                                               maxshape=VIS_MAX_SHAPE,
                                               dtype=np.float32,
                                               compression="gzip")
        data_cam_90 = f.create_dataset('vis_90',
                                       VIS_SHAPE,
                                       maxshape=VIS_MAX_SHAPE,
                                       dtype=np.float32,
                                       compression="gzip")
        data_cam_100 = f.create_dataset('vis_100',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_110 = f.create_dataset('vis_110',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_120 = f.create_dataset('vis_120',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_130 = f.create_dataset('vis_130',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")

        data_cam_140 = f.create_dataset('vis_140',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_150 = f.create_dataset('vis_150',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_160 = f.create_dataset('vis_160',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_165 = f.create_dataset('vis_165',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_170 = f.create_dataset('vis_170',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_175 = f.create_dataset('vis_175',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")

        data_cam_180 = f.create_dataset('vis_180',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_185 = f.create_dataset('vis_185',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_190 = f.create_dataset('vis_190',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_target = f.create_dataset('target',
                                       (1,),
                                       maxshape=(None,),
                                       dtype=np.int32,
                                       compression="gzip")
        data_image = f.create_dataset('image',
                                      (1, 3, 224, 224),
                                      maxshape=(None, 3, 224, 224),
                                      dtype=np.float32,
                                      compression="gzip")

        for batch_idx, (data, target) in enumerate(tqdm(sample_loader)):
            first = True if batch_idx == 0 else False
            resize_array_src_to_dst_shape(src_array=data_cam_min_pred_loss, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_max_logits, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_90, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_100, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_110, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_120, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_130, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_140, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_150, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_160, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_165, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_170, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_175, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_180, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_185, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_190, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_image, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_target, dst_array_shape=data.shape, is_first=first)

            # Add data
            data_image[-data.shape[0]:] = data.data.cpu().numpy()
            data_target[-data.shape[0]:] = target.data.cpu().numpy()

            # data = normalize(data)
            data = data.to(device)
            # data.requires_grad_()
            res_by_iter = {}
            d_cls_attentions_probs = temp_softmax_optimization(vit_model=vit_model, feature_extractor=feature_extractor,
                                                               image=transforms.ToPILImage()(
                                                                   data.reshape(3, vit_config['img_size'],
                                                                                vit_config['img_size'])),
                                                               num_steps=vit_config['num_steps'])

            for iter_desc, cls_attn in d_cls_attentions_probs.items():
                res_by_iter[iter_desc] = patch_score_to_image(transformer_attribution=cls_attn.median(dim=0)[0],
                                                              output_2d_tensor=False)  # [1, 1, 224, 224]
            insert_result_to_array(res_by_iter['max_logits'], array=data_cam_max_logits, data_shape=data.shape)
            insert_result_to_array(res_by_iter['min_pred_loss'], array=data_cam_min_pred_loss, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_90'], array=data_cam_90, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_100'], array=data_cam_100, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_110'], array=data_cam_110, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_120'], array=data_cam_120, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_130'], array=data_cam_130, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_140'], array=data_cam_140, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_150'], array=data_cam_150, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_160'], array=data_cam_160, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_165'], array=data_cam_165, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_170'], array=data_cam_170, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_175'], array=data_cam_175, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_180'], array=data_cam_180, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_185'], array=data_cam_185, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_190'], array=data_cam_190, data_shape=data.shape)


def insert_result_to_array(result, array, data_shape: Tuple):
    array[-data_shape[0]:] = result.cpu().detach().numpy()


def resize_array_src_to_dst_shape(src_array, dst_array_shape, is_first: bool):
    if is_first:
        src_array.resize(src_array.shape[0] + dst_array_shape[0] - 1, axis=0)
    else:
        src_array.resize(src_array.shape[0] + dst_array_shape[0], axis=0)
    return src_array


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

    experiment_path = create_folder(Path(EXPERIMENTS_FOLDER_PATH, vit_config['evaluation']['experiment_folder_name']))
    results_path = Path(experiment_path, 'results.hdf5')
    feature_extractor, model = load_feature_extractor_and_vit_model(vit_config=vit_config, model_type='vit-for-dino')

    # Dataset loader for sample images
    transform = transforms.Compose([
        transforms.Resize((vit_config['img_size'], vit_config['img_size'])),
        transforms.ToTensor(),
    ])
    images_indices = eval(read_file(path=Path(EVALUATION_FOLDER_PATH, 'images_to_test.txt')))
    print(images_indices)
    # val_imagenet_ds = ImageNet(str(DATA_PATH), split='val', transform=transform)
    val_imagenet_ds = ImageNetDataset(transform=transform)
    imagenet_ds = torch.utils.data.Subset(val_imagenet_ds, images_indices)
    sample_loader = DataLoader(
        imagenet_ds,
        batch_size=vit_config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    compute_saliency_and_save(results_path=results_path, feature_extractor=feature_extractor, vit_model=model)
