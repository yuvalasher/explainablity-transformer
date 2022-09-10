import pytorch_lightning as pl
from torch.utils.data import DataLoader

from main.seg_classification.image_token_dataset import ImageSegDataset


class ImageSegDataModule(pl.LightningDataModule):
    def __init__(self, feature_extractor, train_images_path: str, val_images_path: str, batch_size: int, train_n_samples: int, val_n_samples:int) -> object:
        super().__init__()
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        self.train_images_path = train_images_path
        self.val_images_path = val_images_path
        self.train_n_samples = train_n_samples
        self.val_n_samples = val_n_samples

    def setup(self, stage=None):
        self.train_dataset = ImageSegDataset(
            images_path=self.train_images_path,
            feature_extractor=self.feature_extractor,
            n_samples=self.train_n_samples
        )
        self.val_dataset = ImageSegDataset(
            images_path=self.val_images_path,
            feature_extractor=self.feature_extractor,
            n_samples=self.val_n_samples
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False, #TODO - change to True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
        )
