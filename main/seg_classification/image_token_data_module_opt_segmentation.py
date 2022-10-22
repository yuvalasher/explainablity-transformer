import pytorch_lightning as pl
from torch.utils.data import DataLoader

from main.seg_classification.image_token_dataset import ImageSegDataset
from main.seg_classification.image_token_opt_dataset import ImageSegOptDataset


class ImageSegOptDataModuleSegmentation(pl.LightningDataModule):
    def __init__(
            self,
            train_data_loader: DataLoader
    ):
        super().__init__()
        self.train_data_loader = train_data_loader
        self.val_data_loader = train_data_loader
        self.test_data_loader = train_data_loader

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.train_data_loader

    def val_dataloader(self):
        return self.val_data_loader

    def test_dataloader(self):
        return self.test_data_loader
