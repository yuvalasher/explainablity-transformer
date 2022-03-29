from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import h5py

from datasets.imagenet_dataset import ImageNetDataset
from feature_extractor import ViTFeatureExtractor

from evaluation.evaluation_utils import patch_score_to_image, _remove_file_if_exists
from main.temp_softmax_grad_rollout_opt import temp_softmax_grad_rollout_optimization
from utils.consts import EXPERIMENTS_FOLDER_PATH, EVALUATION_FOLDER_PATH
from config import config
from vit_utils import load_feature_extractor_and_vit_model, create_folder, read_file, setup_model_and_optimizer
from torch import nn

vit_config = config['vit']
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

VIS_SHAPE: Tuple[int, int, int, int] = (1, 1, 224, 224)
VIS_MAX_SHAPE: Tuple[None, int, int, int] = (None, 1, 224, 224)


def compute_saliency_and_save(results_path: Path, feature_extractor: ViTFeatureExtractor, vit_model: nn.Module):
    _remove_file_if_exists(path=results_path)

    with h5py.File(results_path, 'a') as f:
        data_cam_min_pred_loss_rollout_grad_max = f.create_dataset('vis_min_pred_loss_rollout_grad_max',
                                                  VIS_SHAPE,
                                                  maxshape=VIS_MAX_SHAPE,
                                                  dtype=np.float32,
                                                  compression="gzip")
        data_cam_max_logits_rollout_grad_max = f.create_dataset('vis_max_logits_rollout_grad_max',
                                               VIS_SHAPE,
                                               maxshape=VIS_MAX_SHAPE,
                                               dtype=np.float32,
                                               compression="gzip")

        data_cam_90_rollout_grad_max = f.create_dataset('vis_90_rollout_grad_max',
                                                         VIS_SHAPE,
                                                         maxshape=VIS_MAX_SHAPE,
                                                         dtype=np.float32,
                                                         compression="gzip")
        data_cam_100_rollout_grad_max = f.create_dataset('vis_100_rollout_grad_max',
                                                         VIS_SHAPE,
                                                         maxshape=VIS_MAX_SHAPE,
                                                         dtype=np.float32,
                                                         compression="gzip")
        data_cam_110_rollout_grad_max = f.create_dataset('vis_110_rollout_grad_max',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_120_rollout_grad_max = f.create_dataset('vis_120_rollout_grad_max',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_130_rollout_grad_max = f.create_dataset('vis_130_rollout_grad_max',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")

        data_cam_140_rollout_grad_max = f.create_dataset('vis_140_rollout_grad_max',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_150_rollout_grad_max = f.create_dataset('vis_150_rollout_grad_max',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_160_rollout_grad_max = f.create_dataset('vis_160_rollout_grad_max',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_165_rollout_grad_max = f.create_dataset('vis_165_rollout_grad_max',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_170_rollout_grad_max = f.create_dataset('vis_170_rollout_grad_max',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_175_rollout_grad_max = f.create_dataset('vis_175_rollout_grad_max',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")

        data_cam_180_rollout_grad_max = f.create_dataset('vis_180_rollout_grad_max',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_185_rollout_grad_max = f.create_dataset('vis_185_rollout_grad_max',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_190_rollout_grad_max = f.create_dataset('vis_190_rollout_grad_max',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")

        data_cam_min_pred_loss_rollout_mean_relu_grad = f.create_dataset('vis_min_pred_loss_rollout_mean_relu_grad',
                                                  VIS_SHAPE,
                                                  maxshape=VIS_MAX_SHAPE,
                                                  dtype=np.float32,
                                                  compression="gzip")
        data_cam_max_logits_rollout_mean_relu_grad = f.create_dataset('vis_max_logits_rollout_mean_relu_grad',
                                               VIS_SHAPE,
                                               maxshape=VIS_MAX_SHAPE,
                                               dtype=np.float32,
                                               compression="gzip")

        data_cam_90_rollout_mean_relu_grad = f.create_dataset('vis_90_rollout_mean_relu_grad',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_100_rollout_mean_relu_grad = f.create_dataset('vis_100_rollout_mean_relu_grad',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")

        data_cam_110_rollout_mean_relu_grad = f.create_dataset('vis_110_rollout_mean_relu_grad',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_120_rollout_mean_relu_grad = f.create_dataset('vis_120_rollout_mean_relu_grad',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_130_rollout_mean_relu_grad = f.create_dataset('vis_130_rollout_mean_relu_grad',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")

        data_cam_140_rollout_mean_relu_grad = f.create_dataset('vis_140_rollout_mean_relu_grad',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_150_rollout_mean_relu_grad = f.create_dataset('vis_150_rollout_mean_relu_grad',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_160_rollout_mean_relu_grad = f.create_dataset('vis_160_rollout_mean_relu_grad',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_165_rollout_mean_relu_grad = f.create_dataset('vis_165_rollout_mean_relu_grad',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_170_rollout_mean_relu_grad = f.create_dataset('vis_170_rollout_mean_relu_grad',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_175_rollout_mean_relu_grad = f.create_dataset('vis_175_rollout_mean_relu_grad',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")

        data_cam_180_rollout_mean_relu_grad = f.create_dataset('vis_180_rollout_mean_relu_grad',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_185_rollout_mean_relu_grad = f.create_dataset('vis_185_rollout_mean_relu_grad',
                                        VIS_SHAPE,
                                        maxshape=VIS_MAX_SHAPE,
                                        dtype=np.float32,
                                        compression="gzip")
        data_cam_190_rollout_mean_relu_grad = f.create_dataset('vis_190_rollout_mean_relu_grad',
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
        vit_ours_model, _ = setup_model_and_optimizer(model_name='softmax_temp')
        for batch_idx, (data, target) in enumerate(tqdm(sample_loader)):
            first = True if batch_idx == 0 else False
            resize_array_src_to_dst_shape(src_array=data_cam_min_pred_loss_rollout_grad_max, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_max_logits_rollout_grad_max, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_90_rollout_grad_max, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_100_rollout_grad_max, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_110_rollout_grad_max, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_120_rollout_grad_max, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_130_rollout_grad_max, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_140_rollout_grad_max, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_150_rollout_grad_max, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_160_rollout_grad_max, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_165_rollout_grad_max, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_170_rollout_grad_max, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_175_rollout_grad_max, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_180_rollout_grad_max, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_185_rollout_grad_max, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_190_rollout_grad_max, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_min_pred_loss_rollout_mean_relu_grad, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_max_logits_rollout_mean_relu_grad, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_90_rollout_mean_relu_grad, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_100_rollout_mean_relu_grad, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_110_rollout_mean_relu_grad, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_120_rollout_mean_relu_grad, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_130_rollout_mean_relu_grad, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_140_rollout_mean_relu_grad, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_150_rollout_mean_relu_grad, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_160_rollout_mean_relu_grad, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_165_rollout_mean_relu_grad, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_170_rollout_mean_relu_grad, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_175_rollout_mean_relu_grad, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_180_rollout_mean_relu_grad, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_185_rollout_mean_relu_grad, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_cam_190_rollout_mean_relu_grad, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_image, dst_array_shape=data.shape, is_first=first)
            resize_array_src_to_dst_shape(src_array=data_target, dst_array_shape=data.shape, is_first=first)

            # Add data
            data_image[-data.shape[0]:] = data.data.cpu().numpy()
            data_target[-data.shape[0]:] = target.data.cpu().numpy()

            # data = normalize(data)
            data = data.to(device)
            # data.requires_grad_()
            res_by_iter = {}
            d_cls_attentions_probs = temp_softmax_grad_rollout_optimization(vit_ours_model=vit_ours_model, vit_model=vit_model, # TODO - change!!!!
                                                               feature_extractor=feature_extractor,
                                                               image=transforms.ToPILImage()(
                                                                   data.reshape(3, vit_config['img_size'],
                                                                                vit_config['img_size'])),
                                                               num_steps=vit_config['num_steps'])

            for iter_desc, cls_attn in d_cls_attentions_probs.items():
                res_by_iter[iter_desc] = patch_score_to_image(transformer_attribution=cls_attn.median(dim=0)[0],
                                                              output_2d_tensor=False)  # [1, 1, 224, 224]
            insert_result_to_array(res_by_iter['max_logits_rollout_grad_max'], array=data_cam_min_pred_loss_rollout_grad_max, data_shape=data.shape)
            insert_result_to_array(res_by_iter['min_pred_loss_rollout_grad_max'], array=data_cam_max_logits_rollout_grad_max, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_90_rollout_grad_max'], array=data_cam_90_rollout_grad_max, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_100_rollout_grad_max'], array=data_cam_100_rollout_grad_max, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_110_rollout_grad_max'], array=data_cam_110_rollout_grad_max, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_120_rollout_grad_max'], array=data_cam_120_rollout_grad_max, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_130_rollout_grad_max'], array=data_cam_130_rollout_grad_max, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_140_rollout_grad_max'], array=data_cam_140_rollout_grad_max, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_150_rollout_grad_max'], array=data_cam_150_rollout_grad_max, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_160_rollout_grad_max'], array=data_cam_160_rollout_grad_max, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_165_rollout_grad_max'], array=data_cam_165_rollout_grad_max, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_170_rollout_grad_max'], array=data_cam_170_rollout_grad_max, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_175_rollout_grad_max'], array=data_cam_175_rollout_grad_max, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_180_rollout_grad_max'], array=data_cam_180_rollout_grad_max, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_185_rollout_grad_max'], array=data_cam_185_rollout_grad_max, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_190_rollout_grad_max'], array=data_cam_190_rollout_grad_max, data_shape=data.shape)

            insert_result_to_array(res_by_iter['max_logits_rollout_mean_relu_grad'], array=data_cam_min_pred_loss_rollout_mean_relu_grad, data_shape=data.shape)
            insert_result_to_array(res_by_iter['min_pred_loss_rollout_mean_relu_grad'], array=data_cam_max_logits_rollout_mean_relu_grad, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_90_rollout_mean_relu_grad'], array=data_cam_90_rollout_mean_relu_grad, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_100_rollout_mean_relu_grad'], array=data_cam_100_rollout_mean_relu_grad, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_110_rollout_mean_relu_grad'], array=data_cam_110_rollout_mean_relu_grad, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_120_rollout_mean_relu_grad'], array=data_cam_120_rollout_mean_relu_grad, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_130_rollout_mean_relu_grad'], array=data_cam_130_rollout_mean_relu_grad, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_140_rollout_mean_relu_grad'], array=data_cam_140_rollout_mean_relu_grad, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_150_rollout_mean_relu_grad'], array=data_cam_150_rollout_mean_relu_grad, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_160_rollout_mean_relu_grad'], array=data_cam_160_rollout_mean_relu_grad, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_165_rollout_mean_relu_grad'], array=data_cam_165_rollout_mean_relu_grad, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_170_rollout_mean_relu_grad'], array=data_cam_170_rollout_mean_relu_grad, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_175_rollout_mean_relu_grad'], array=data_cam_175_rollout_mean_relu_grad, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_180_rollout_mean_relu_grad'], array=data_cam_180_rollout_mean_relu_grad, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_185_rollout_mean_relu_grad'], array=data_cam_185_rollout_mean_relu_grad, data_shape=data.shape)
            insert_result_to_array(res_by_iter['iter_190_rollout_mean_relu_grad'], array=data_cam_190_rollout_mean_relu_grad, data_shape=data.shape)



def insert_result_to_array(result, array, data_shape: Tuple):
    array[-data_shape[0]:] = result.cpu().detach().numpy()


def resize_array_src_to_dst_shape(src_array, dst_array_shape, is_first: bool):
    if is_first:
        src_array.resize(src_array.shape[0] + dst_array_shape[0] - 1, axis=0)
    else:
        src_array.resize(src_array.shape[0] + dst_array_shape[0], axis=0)
    return src_array


if __name__ == "__main__":
    experiment_path = create_folder(
        Path(EXPERIMENTS_FOLDER_PATH, 'temp', vit_config['evaluation']['experiment_folder_name']))
    results_path = Path(experiment_path, 'results.hdf5')
    feature_extractor, model = load_feature_extractor_and_vit_model(vit_config=vit_config, model_type='vit-for-dino',
                                                                    is_wolf_transforms=vit_config['is_wolf_transforms'])

    # Dataset loader for sample images
    transform = transforms.Compose([
        transforms.Resize((vit_config['img_size'], vit_config['img_size'])),
        transforms.ToTensor(),
    ])
    images_indices = eval(read_file(path=Path(EVALUATION_FOLDER_PATH, 'images_to_test.txt')))
    print(images_indices)
    val_imagenet_ds = ImageNetDataset(transform=transform)
    imagenet_ds = torch.utils.data.Subset(val_imagenet_ds, images_indices)
    sample_loader = DataLoader(
        imagenet_ds,
        batch_size=vit_config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    compute_saliency_and_save(results_path=results_path, feature_extractor=feature_extractor, vit_model=model)
