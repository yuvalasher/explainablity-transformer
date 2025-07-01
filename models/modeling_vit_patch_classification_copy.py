from typing import Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.vit import ViTPreTrainedModel, ViTModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.vit.modeling_vit import ViTEncoder

from config import config
from config.imagenet1000_clsidx_to_labels import class_idx_to_label

vit_config = config["vit"]
EPSILON = 0.05


class ViTForMaskGenerationInpaint(ViTPreTrainedModel):
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
            inpaint_model=None,  # Optional inpainting model
            use_inpainting_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        tokens_output_reshaped = self.patch_pooler(tokens_output_reshaped)
        tokens_output_reshaped = self.activation(tokens_output_reshaped)
        # tokens_output_reshaped = self.dropout(tokens_output_reshaped)
        logits = self.patch_classifier(tokens_output_reshaped)

        if use_inpainting_mask:
            prob_mask = torch.sigmoid(logits)  # Convert logits to probabilities

            mask_bernoulli = torch.bernoulli(prob_mask)

            sampled_mask =mask_bernoulli.view(batch_size, 1,
                                                           int(vit_config["img_size"] / vit_config["patch_size"]),
                                                           int(vit_config["img_size"] / vit_config["patch_size"]))

            interpolated_mask = torch.nn.functional.interpolate(sampled_mask, scale_factor=vit_config["patch_size"],
                                                                mode='bilinear')
            inpainted_image = None
            from torchvision.transforms.functional import to_pil_image, to_tensor
            from PIL import Image
            from diffusers import StableDiffusionInpaintPipeline
            inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                revision="fp16",
                torch_dtype=torch.float16,
            ).to("cuda")
            target_class = torch.argmax(logits, dim=1)
            if inpaint_model is not None:

                binary_mask = (interpolated_mask > 0.5).float()
                inpainted_image = []

                for i in range(batch_size):
                    # Resize image and mask to 512x512 for inpainting
                    image_pil = to_pil_image(pixel_values[i].cpu()).resize((512, 512), Image.BICUBIC)
                    mask_pil = to_pil_image(binary_mask[i][0].cpu()).resize((512, 512), Image.NEAREST)
                    prompt = class_idx_to_label[(target_class[i]+ 1) % len(class_idx_to_label)],  # get next class label
                    # Run Stable Diffusion Inpainting
                    result = inpaint_model(
                        prompt="",  # or optionally add class prompt
                        image=image_pil,
                        mask_image=mask_pil,
                        guidance_scale=7.5
                    ).images[0]

                    # Resize result back to 224x224 before classification
                    result_resized = result.resize((224, 224), Image.BICUBIC)
                    inpainted_image.append(to_tensor(result_resized))

                inpainted_image = torch.stack(inpainted_image).to(pixel_values.device)


        else:
            mask = logits.view(batch_size, -1, 1)  # logits - [batch_size, tokens_count]

            if vit_config["activation_function"] == 'relu':
                mask = torch.relu(mask)
            if vit_config["activation_function"] == 'sigmoid':
                mask = torch.sigmoid(mask)
            if vit_config["activation_function"] == 'softmax':
                mask = torch.softmax(mask, dim=1)

            sampled_mask = mask.view(batch_size, 1, int(vit_config["img_size"] / vit_config["patch_size"]),
                                     int(vit_config["img_size"] / vit_config["patch_size"]))
            interpolated_mask = torch.nn.functional.interpolate(sampled_mask, scale_factor=vit_config["patch_size"],
                                                                mode='bilinear')
            inpainted_image = None

        return interpolated_mask, sampled_mask, inpainted_image
