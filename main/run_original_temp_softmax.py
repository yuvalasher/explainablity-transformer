from tqdm import tqdm
# from utils.utils import *
from loss_utils import *
import wandb
from log_utils import get_wandb_config
from utils.consts import *
from pytorch_lightning import seed_everything
from typing import Callable
from objectives import objective_temp_softmax
from time import time

vit_config = config['vit']
loss_config = vit_config['loss']

seed_everything(config['general']['seed'])
feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config, model_type='vit-for-dino',
                                                                    is_wolf_transforms=vit_config['is_wolf_transforms'])


def optimize_params(vit_model: ViTForImageClassification, criterion: Callable):
    for idx, image_dict in enumerate(vit_config['images']):
        image_name, correct_class_idx, contrastive_class_idx = get_image_spec(image_dict)
        # wandb_config = get_wandb_config(vit_config=vit_config, experiment_name=experiment_name, image_name=image_name)
        start_time = time()
        # with wandb.init(project=config['general']['wandb_project'], entity=config['general']['wandb_entity'],
        #                 config=wandb_config) as run:
        vit_ours_model, optimizer = setup_model_and_optimizer(model_name='softmax_temp')

        image_plot_folder_path = get_and_create_image_plot_folder_path(images_folder_path=IMAGES_FOLDER_PATH,
                                                                       experiment_name=experiment_name,
                                                                       image_name=image_name)

        inputs, original_transformed_image = get_image_and_inputs_and_transformed_image(image_name=image_name,
                                                                                        feature_extractor=feature_extractor)
        target = vit_model(**inputs)
        target_class_idx = torch.argmax(target.logits[0])

        total_losses, prediction_losses, correct_class_logits, correct_class_probs, tokens_mask, temps = [], [], [], [], [], []

        attention_probs = get_attention_probs(model=vit_model)
        h_rollout_max = rollout(attnetions=attention_probs, head_fusion='max')
        h_rollout_mean = rollout(attnetions=attention_probs, head_fusion='mean')
        image_rollout_plots_folder = create_folder(Path(image_plot_folder_path, 'rollout'))
        visu(original_image=original_transformed_image, transformer_attribution=h_rollout_max, file_name=Path(image_rollout_plots_folder, 'h_rollout_max'))
        visu(original_image=original_transformed_image, transformer_attribution=h_rollout_mean, file_name=Path(image_rollout_plots_folder, 'h_rollout_mean'))
        plot_attention_rollout(attention_probs=attention_probs, path=image_plot_folder_path,
                               patch_size=16, iteration_idx=0, head_fusion='max',
                               original_image=original_transformed_image)
        plot_attention_rollout(attention_probs=attention_probs, path=image_plot_folder_path,
                               patch_size=16, iteration_idx=0, head_fusion='mean',
                               original_image=original_transformed_image)


if __name__ == '__main__':
    experiment_name = f"hila_rollouts"
    print(experiment_name)
    _ = create_folder(Path(PLOTS_PATH, experiment_name))
    optimize_params(vit_model=vit_model, criterion=objective_temp_softmax)