import torch
from pytorch_lightning import LightningModule
from torch import nn
from config import config

vit_config = config["vit"]


class CNNForMaskGeneration(LightningModule):
    def __init__(self, cnn_model, img_size: int = 224):
        super().__init__()
        self.img_size = img_size
        backbone_children = list(cnn_model.children())
        self.cnn_model = nn.Sequential(*backbone_children[:-1])
        backbone_pretrained_classifier = backbone_children[-1]
        self.pooler = nn.Linear(backbone_pretrained_classifier.in_features, backbone_pretrained_classifier.in_features)
        self.score_classifier = nn.Sequential(
            nn.Linear(backbone_pretrained_classifier.in_features, self.img_size * self.img_size))
        self.activation = nn.Tanh()

    def forward(self, inputs):  # inputs.shape: [batch_size, 3, 224, 224]
        batch_size = inputs.shape[0]
        self.cnn_model.eval()
        representations = self.cnn_model(inputs).flatten(1)  # [batch_size, 2048]
        mask = self.pooler(representations)
        mask = self.activation(mask)
        mask = self.score_classifier(mask)
        if vit_config["activation_function"] == 'sigmoid':
            mask = torch.sigmoid(mask)
        if vit_config["normalize_by_max_patch"]:
            mask = mask / mask.max(dim=1, keepdim=True)[0]
        interpolated_mask = mask.view(batch_size, self.img_size, self.img_size)
        return interpolated_mask, mask  # [batch_size, img_size, img_size] , [batch_size, 1, n_tokens]
