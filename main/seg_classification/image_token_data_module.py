import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from main.seg_classification.image_token_dataset import ImageSegDataset


class ImageSegDataModule(pl.LightningDataModule):
    def __init__(
            self,
            feature_extractor,
            batch_size: int,
            train_images_path: str,
            val_images_path: str,
            is_sampled_data_uniformly: bool,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        self.train_images_path = train_images_path
        self.val_images_path = val_images_path
        self.is_sampled_data_uniformly = is_sampled_data_uniformly

    def setup(self, stage=None):
        self.train_dataset = ImageSegDataset(
            images_path=self.train_images_path,
            feature_extractor=self.feature_extractor,
            is_val=False,
            is_sampled_data_uniformly=self.is_sampled_data_uniformly,
        )
        self.val_dataset = ImageSegDataset(
            images_path=self.val_images_path,
            feature_extractor=self.feature_extractor,
            is_val=True,
            is_sampled_data_uniformly=self.is_sampled_data_uniformly,
        )

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
                          #generator=torch.Generator(device='cuda'))

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False),
                          # generator=torch.Generator(device='cuda'))
