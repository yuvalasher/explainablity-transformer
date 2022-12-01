import torch
from pytorch_lightning import LightningModule
from torch import nn


class CNNForMaskGeneration(LightningModule):
    def __init__(self, cnn_model, activation_function: str = "sigmoid", img_size: int = 224):
        super().__init__()
        self.img_size = img_size
        self.activation_function = activation_function
        backbone_children = list(cnn_model.children())
        self.encoder = nn.Sequential(*backbone_children[:-2])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=backbone_children[-1].in_features, out_channels=1, kernel_size=1, stride=1),
        )

    def forward(self, inputs):  # inputs.shape: [batch_size, 3, 224, 224]
        batch_size = inputs.shape[0]
        # ic(inputs.shape)
        self.encoder.eval()
        enc_rep = self.encoder(inputs)  # [batch_size, 2048, 7, 7]
        # ic(representations.shape)
        mask = self.bottleneck(enc_rep)
        # ic(mask.shape)

        # ic(mask.shape)
        if self.activation_function == 'sigmoid':
            mask = torch.sigmoid(mask)

        mask = torch.nn.functional.interpolate(mask, scale_factor=32, mode="bilinear")
        interpolated_mask = mask.view(batch_size, 1, self.img_size, self.img_size)
        return interpolated_mask, mask  # [batch_size, img_size, img_size] , [batch_size, 1, n_tokens]
