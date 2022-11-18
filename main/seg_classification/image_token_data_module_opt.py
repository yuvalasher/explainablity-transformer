import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from main.seg_classification.image_token_opt_dataset import ImageSegOptDataset


class ImageSegOptDataModule(pl.LightningDataModule):
    def __init__(
            self,
            feature_extractor,
            batch_size: int,
            train_image_path: str,
            val_image_path: str,
            target: int,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        self.train_image_path = train_image_path
        self.val_image_path = val_image_path
        self.target = target

    def setup(self, stage=None):
        self.train_dataset = ImageSegOptDataset(
            image_path=self.train_image_path,
            feature_extractor=self.feature_extractor,
            target=self.target
        )
        self.val_dataset = ImageSegOptDataset(
            image_path=self.val_image_path,
            feature_extractor=self.feature_extractor,
            target=self.target,
        )

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False)
