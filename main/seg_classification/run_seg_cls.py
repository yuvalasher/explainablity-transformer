import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from pathlib import Path
from typing import Tuple

import wandb

from main.seg_classification.image_classification_with_token_classification_model import \
    ImageClassificationWithTokenClassificationModel
from main.seg_classification.image_token_data_module import ImageSegDataModule
from main.seg_classification.image_token_dataset import ImageSegDataset
import pytorch_lightning as pl
from config import config
from utils.consts import VAL_IMAGES_FOLDER_PATH, TRAIN_IMAGES_FOLDER_PATH
from vit_utils import load_feature_extractor_and_vit_model, get_warmup_steps_and_total_training_steps, \
    freeze_multitask_model, print_number_of_trainable_and_not_trainable_params
from transformers import AutoModel, ViTForImageClassification
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch

# torch.cuda.set_device(3)
print(torch.cuda.current_device())
vit_config = config['vit']
seed_everything(config['general']['seed'])
import gc
import torch

torch.cuda.empty_cache()
gc.collect()
exp_name = f'pred_loss_{vit_config["seg_cls"]["loss"]["prediction_loss_mul"]}_mask_loss_{vit_config["seg_cls"]["loss"]["mask_loss_mul"]}'

feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config, model_type='vit-basic',
                                                                    is_wolf_transforms=vit_config[
                                                                        'is_wolf_transforms'])  # TODO if vit-for-dino is relevant

vit_with_classification_head = ViTForImageClassification.from_pretrained(vit_config['model_name'])
vit_without_classification_head = AutoModel.from_pretrained(vit_config['model_name'])
data_module = ImageSegDataModule(feature_extractor=feature_extractor, train_images_path=str(TRAIN_IMAGES_FOLDER_PATH),
                                 val_images_path=str(VAL_IMAGES_FOLDER_PATH), batch_size=vit_config['batch_size'])

warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(n_epochs=vit_config['n_epochs'],
                                                                               train_samples_length=len(list(Path(
                                                                                   TRAIN_IMAGES_FOLDER_PATH).iterdir())),
                                                                               batch_size=vit_config['batch_size'])
plot_path = Path(vit_config['plot_path'], exp_name)
model = ImageClassificationWithTokenClassificationModel(vit_with_classification_head=vit_with_classification_head,
                                                        vit_without_classification_head=vit_without_classification_head,
                                                        feature_extractor=feature_extractor,
                                                        plot_path=plot_path,
                                                        warmup_steps=warmup_steps,
                                                        total_training_steps=total_training_steps,
                                                        batch_size=vit_config['batch_size'])

wandb.login()
wandb_logger = WandbLogger(
    name=f'seg_cls; {exp_name}',
    project='run_seg_cls', )

# early_stop_callback = EarlyStopping(
#    monitor='val_loss',
#    min_delta=0.0001,
#    patience=3,
#    verbose=False,
#    mode='min')

model = freeze_multitask_model(model=model, freezing_transformer=vit_config['freezing_transformer'],
                               freeze_classification_head=vit_config['freeze_classification_head'])
print_number_of_trainable_and_not_trainable_params(model)
trainer = pl.Trainer(
    callbacks=[ModelCheckpoint(monitor="val_loss", mode="min", filename="{epoch}--{val_loss:.1f}", save_top_k=3)],
    # , early_stop_callback],
    logger=[wandb_logger],
    max_epochs=vit_config['n_epochs'],
    gpus=vit_config['gpus'],
    progress_bar_refresh_rate=30,
    default_root_dir=vit_config['default_root_dir']
)

trainer.fit(model=model, datamodule=data_module)
