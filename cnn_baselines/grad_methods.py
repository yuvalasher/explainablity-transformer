    import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from pathlib import Path
import torch
from torch import Tensor
from tqdm import tqdm
from matplotlib import pyplot as plt
from cnn_baselines.grad_methods_utils import run_all_operations
from cnn_baselines.saliency_models import GradModel, ReLU, lift_cam, ig_captum, generic_torchcam
from utils import get_gt_classes, get_preprocessed_image, show_image
from utils.consts import GT_VALIDATION_PATH_LABELS, IMAGENET_VAL_IMAGES_FOLDER_PATH
from cnn_baselines.torchgc.pytorch_grad_cam.fullgrad_cam import FullGrad
from cnn_baselines.torchgc.pytorch_grad_cam.layer_cam import LayerCAM
from cnn_baselines.torchgc.pytorch_grad_cam.score_cam import ScoreCAM
from cnn_baselines.torchgc.pytorch_grad_cam.ablation_cam import AblationCAM

device = torch.device('cuda')

backbones = ['densenet', 'resnet101']
FEATURE_LAYER_NUMBER_BY_BACKBONE = {'resnet101': 8, 'densenet': 12}

if __name__ == '__main__':
    images_listdir = sorted(list(Path(IMAGENET_VAL_IMAGES_FOLDER_PATH).iterdir()))[1:2]
    targets = get_gt_classes(path=GT_VALIDATION_PATH_LABELS)[1:2]

    results = []
    image = None

    backbone_name = backbones[1]
    FEATURE_LAYER_NUMBER = FEATURE_LAYER_NUMBER_BY_BACKBONE[backbone_name]

    PREV_LAYER = FEATURE_LAYER_NUMBER - 1
    num_layers_options = [1]

    BY_MAX_CLASS = False  # predicted / TARGET
    USE_MASK = True
    save_img = False
    save_heatmaps_masks = False
    to_write_results = False

    operations = ['fullgrad', 'ablation-cam', 'lift-cam', 'layercam',
                  'ig']  # TODO Yuval - the methods of grad2heatmaps should be in the end of the list
    gradient_operations = ['gradcam', 'gradcampp']
    torch.nn.modules.activation.ReLU.forward = ReLU.forward
    if backbone_name.__contains__('vgg'):
        torch.nn.modules.activation.ReLU.forward = ReLU.forward

    model = GradModel(backbone_name, feature_layer=FEATURE_LAYER_NUMBER)
    model.to(device)
    model.eval()
    model.zero_grad()

    for index, (image_path, target) in tqdm(enumerate(zip(images_listdir, targets)),
                                            position=0,
                                            leave=True,
                                            total=len(images_listdir)):

        current_image_results = {}
        label = target
        target_label = target
        inputs = get_preprocessed_image(image_path=image_path)

        input_predictions = model(inputs.to(device), hook=False).detach()
        predicted_label = torch.max(input_predictions, 1).indices[0].item()

        if BY_MAX_CLASS:
            label = predicted_label

        res_class_saliency = run_all_operations(model=model,
                                                image_preprocessed=inputs.squeeze(0),
                                                label=label,
                                                backbone_name=backbone_name,
                                                device=device,
                                                features_layer=FEATURE_LAYER_NUMBER,
                                                operations=gradient_operations,
                                                use_mask=USE_MASK,
                                                )
        show_image(res_class_saliency[0][1], title='gradcam')
        show_image(res_class_saliency[1][1], title='gradcampp')
        """
        Another baselines beside GC & GC++
        """
        operation_index = 0
        score_original_image = 0
        img_dict = []
        for operation in operations:
            print(operation)
            if operation == 'lift-cam':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = lift_cam(
                    model=model,
                    inputs=inputs,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
                show_image(blended_im, title=operation)

            elif operation == 'score-cam':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = generic_torchcam(
                    modelCAM=ScoreCAM,
                    backbone_name=backbone_name,
                    inputs=inputs,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
                show_image(blended_im, title=operation)
            elif operation == 'ablation-cam':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = generic_torchcam(
                    modelCAM=AblationCAM,
                    backbone_name=backbone_name,
                    inputs=inputs,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
                show_image(blended_im, title=operation)

            elif operation == 'ig':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = ig_captum(
                    model=model,
                    inputs=inputs,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
                show_image(blended_im, title=operation)

            elif operation == 'layercam':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = generic_torchcam(
                    modelCAM=LayerCAM,
                    backbone_name=backbone_name,
                    inputs=inputs,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
                show_image(blended_im, title=operation)

            elif operation == 'fullgrad':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = generic_torchcam(
                    modelCAM=FullGrad,
                    backbone_name=backbone_name,
                    inputs=inputs,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
                show_image(blended_im, title=operation)

            # else:
            #     t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = res_class_saliency[
            #         operation_index]
            #     operation_index = operation_index + 1
