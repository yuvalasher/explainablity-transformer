from typing import Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.vit import ViTPreTrainedModel, ViTModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from config import config

vit_config = config["vit"]


class ViTForMaskGeneration(ViTPreTrainedModel):
    vit: ViTModel
    patch_classifier: nn.Linear

    def __init__(self, config):
        super().__init__(config)

        # self.num_labels = config.num_labels
        self.vit = ViTModel(config, add_pooling_layer=False)

        # Classifier head
        self.patch_pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_classifier = nn.Linear(config.hidden_size, 1)  # regression to one number

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            pixel_values=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            interpolate_pos_encoding=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs: BaseModelOutputWithPooling = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state

        tokens_output = sequence_output[:, 1:, :]
        # tokens_output - [batch_size, tokens_count, hidden_size]
        # truncating the hidden states to remove the CLS token, which is the first

        batch_size = tokens_output.shape[0]
        hidden_size = tokens_output.shape[2]

        tokens_output_reshaped = tokens_output.reshape(-1, hidden_size)
        if vit_config["is_mlp_on_segmentation"]:
            tokens_output_reshaped = self.patch_pooler(tokens_output_reshaped)
            tokens_output_reshaped = self.activation(tokens_output_reshaped)
            # tokens_output_reshaped = self.dropout(tokens_output_reshaped)
        logits = self.patch_classifier(tokens_output_reshaped)
        mask = logits.view(batch_size, -1, 1) # logits - [batch_size, tokens_count]


        if vit_config["is_relu_segmentation"]:
            mask = torch.relu(mask)
        else:
            if vit_config["is_sigmoid_segmentation"]:
                mask = torch.sigmoid(mask)

        mask = mask.view(batch_size, 1, 14, 14)

        interpolated_mask = torch.nn.functional.interpolate(mask, scale_factor=16, mode='bilinear')

        return interpolated_mask, mask
