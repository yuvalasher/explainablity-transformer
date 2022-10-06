import pytorch_lightning as pl
from torch.utils.data import DataLoader

from main.seg_classification.image_token_dataset import ImageSegDataset
from main.seg_classification.image_token_opt_dataset import ImageSegOptDataset


class ImageSegOptDataModuleSegmentation(pl.LightningDataModule):
    def __init__(
            self,
            feature_extractor,
            batch_idx: int,
            train_data_loader: DataLoader,
    ):
        super().__init__()
        self.batch_idx = batch_idx
        self.feature_extractor = feature_extractor
        self.train_data_loader = train_data_loader

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.train_data_loader

    # def val_dataloader(self):
    #     return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False)
