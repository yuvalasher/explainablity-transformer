from typing import Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.models.vit import ViTPreTrainedModel, ViTModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from config import config
from config.imagenet1000_clsidx_to_labels import class_idx_to_label

# Additional imports for inpainting functionality
from torchvision.transforms.functional import to_pil_image, to_tensor
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torchvision.transforms as T

vit_config = config["vit"]


class ViTForMaskGeneration(ViTPreTrainedModel):
    vit: ViTModel
    patch_classifier: nn.Linear

    def __init__(self, config):
        super().__init__(config)

        # Vision Transformer backbone without pooling layer
        self.vit = ViTModel(config, add_pooling_layer=False)

        # Patch-level processing layers
        self.patch_pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.patch_classifier = nn.Linear(config.hidden_size, 1)  # Regression to a single value per patch

        # Initialize weights and apply final processing
        self.post_init()

        # Flag and initialization for optional inpainting model
        self.use_inpaint_model = vit_config['use_inpaint_model']
        if self.use_inpaint_model:
            print('Initializing inpainting model')
            self.inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                revision="fp16",
                torch_dtype=torch.float16,
            ).to("cuda")

    def extract_patch_logits(self, tokens_output):
        logits = self.patch_classifier(
            self.activation(self.patch_pooler(tokens_output.reshape(-1, tokens_output.shape[-1]))))
        return logits, tokens_output.shape[0], tokens_output.shape[1]

    def generate_mask(self, logits, batch_size):
        # Convert logits to probabilities with sigmoid, threshold, and sample in one step
        relevance_threshold = 0.3
        sampled_mask_probs = torch.bernoulli(torch.clamp(torch.sigmoid(logits), min=relevance_threshold))

        patch_dim = int(vit_config["img_size"] / vit_config["patch_size"])
        sampled_mask = sampled_mask_probs.view(batch_size, 1, patch_dim, patch_dim)

        interpolated_mask = torch.nn.functional.interpolate(
            sampled_mask, scale_factor=vit_config["patch_size"], mode='bilinear'
        )
        return sampled_mask, interpolated_mask

    def binarize_mask(self, interpolated_mask):
        # Binarize the mask after interpolation
        binary_mask = (interpolated_mask > 0.5).float()
        return binary_mask

    def prepare_inpainting_inputs(self, pixel_values, binary_mask):
        # Use list comprehension for preparing images and masks
        images_pil = [to_pil_image(pixel_values[i].cpu()).resize((512, 512), Image.BICUBIC) for i in
                      range(pixel_values.shape[0])]
        masks_pil = [to_pil_image(binary_mask[i][0].cpu()).resize((512, 512), Image.NEAREST) for i in
                     range(binary_mask.shape[0])]
        return images_pil, masks_pil

    def get_class_prompt(self, top2_classes, i):
        original_idx, second_idx = (
            top2_classes.indices[i][0].item(), top2_classes.indices[i][1].item()) if top2_classes is not None else (
            0, 0)
        original_class, target_class = class_idx_to_label[original_idx], class_idx_to_label[second_idx]
        prompt = f"A photo of a {target_class}"
        return prompt, original_class, target_class

    def run_inpainting(self, image_pil, mask_pil, prompt):
        with torch.no_grad():
            result = self.inpaint_model(
                prompt=prompt,
                image=image_pil,
                mask_image=mask_pil,
                guidance_scale=7.5
            ).images[0]
            # Resize back to 224x224 for downstream tasks
            result_resized = result.resize((224, 224), Image.BICUBIC)
            return to_tensor(result_resized)

    def generate_mask_without_inpainting(self, logits, batch_size, tokens_count):
        mask = logits.view(batch_size, tokens_count, 1)

        if vit_config["activation_function"] == 'relu':
            mask = torch.relu(mask)
        elif vit_config["activation_function"] == 'sigmoid':
            mask = torch.sigmoid(mask)
        elif vit_config["activation_function"] == 'softmax':
            mask = torch.softmax(mask, dim=1)

        patch_dim = int(vit_config["img_size"] / vit_config["patch_size"])
        sampled_mask = mask.view(batch_size, patch_dim, patch_dim, 1).permute(0, 3, 1, 2)
        interpolated_mask = torch.nn.functional.interpolate(
            sampled_mask, scale_factor=vit_config["patch_size"], mode='bilinear'
        )
        inpainted_image = None
        return sampled_mask, interpolated_mask, inpainted_image

    def perform_inpainting_batch(self, pixel_values, logits, batch_size, top2_classes):
        sampled_mask, interpolated_mask = self.generate_mask(logits, batch_size)
        binary_mask = self.binarize_mask(interpolated_mask)

        images_pil, masks_pil = self.prepare_inpainting_inputs(pixel_values, binary_mask)

        inpainted_images = [self.run_inpainting(images_pil[i], masks_pil[i], self.get_class_prompt(top2_classes, i)[0])
                            for i in range(batch_size)]

        inpainted_image = torch.stack(inpainted_images).to(pixel_values.device)
        return sampled_mask, interpolated_mask, inpainted_image

    def forward(
            self,
            pixel_values=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            interpolate_pos_encoding=None,
            top2_classes=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # --- Feature Extraction ---
        outputs: BaseModelOutputWithPooling = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state  # [batch_size, tokens, hidden_size]
        tokens_output = sequence_output[:, 1:, :]  # Remove CLS token

        logits, batch_size, tokens_count = self.extract_patch_logits(tokens_output)

        if self.use_inpaint_model:
            sampled_mask, interpolated_mask, inpainted_image = self.perform_inpainting_batch(pixel_values, logits,
                                                                                             batch_size, top2_classes)
        else:
            sampled_mask, interpolated_mask, inpainted_image = self.generate_mask_without_inpainting(logits, batch_size,
                                                                                                     tokens_count)

        # Return tuple: (upsampled mask, patch-level mask, inpainted images if any)
        return interpolated_mask, sampled_mask, inpainted_image


if __name__ == "__main__":

    image = Image.open('/home/amitesh/Projects/explainablity-transformer-cv/vit_data/ILSVRC2012_val_00000001.JPEG')

    # Transform to tensor
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).cuda()

    from transformers import ViTConfig

    # Minimal ViT config
    vit_cfg = ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    model = ViTForMaskGeneration(vit_cfg).cuda()
    model.eval()

    from transformers import ViTForImageClassification

    classifier = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").cuda()
    classifier.eval()

    with torch.no_grad():
        outputs = classifier(image_tensor)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1)

    original_idx = top2.indices[0][0].item()
    second_idx = top2.indices[0][1].item()
    original_class = class_idx_to_label[original_idx]
    second_class = class_idx_to_label[second_idx]

    original_proba = probs[0, original_idx].item()

    print(f"Original predicted class: '{original_class}' (index {original_idx})")
    print(f"Second predicted class: '{second_class}' (index {second_idx})")

    with torch.no_grad():
        interpolated_mask, sampled_mask, inpainted_image = model(pixel_values=image_tensor, top2_classes=top2)

    # Compute inpainted image classification probabilities if inpainted_image is not None
    if inpainted_image is not None:
        with torch.no_grad():
            inpaint_outputs = classifier(inpainted_image)
            inpaint_probs = torch.nn.functional.softmax(inpaint_outputs.logits, dim=-1)
        inpaint_target_proba = inpaint_probs[0, original_idx].item()
    else:
        inpaint_target_proba = None

    print("Interpolated mask shape:", interpolated_mask.shape)
    print("Sampled mask shape:", sampled_mask.shape)
    if inpainted_image is not None:
        print("Inpainted image shape:", inpainted_image.shape)

    import matplotlib.pyplot as plt
    import numpy as np

    # Setup subplots: 5 if inpainted_image exists, else 3
    n_subplots = 5 if inpainted_image is not None else 3
    fig, axs = plt.subplots(1, n_subplots, figsize=(4 * n_subplots, 4))

    # Prepare original image as numpy array
    original_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

    # Prepare mask as numpy array, resize to 224x224, and normalize
    mask_np = interpolated_mask[0][0].cpu().numpy()
    # Normalize mask between 0 and 1 if not already
    mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-8)

    # Create a black background image
    black_background = np.zeros_like(original_np)

    # Use mask as window: show original pixels where mask is 1, else black
    overlay_np = original_np * mask_np[..., None] + black_background * (1 - mask_np[..., None])

    if inpainted_image is not None:
        inpainted_np = inpainted_image[0].permute(1, 2, 0).cpu().numpy()
        # Ensure float32 and normalized between 0 and 1 if needed
        if inpainted_np.max() > 1.0:
            inpainted_np = inpainted_np / 255.0
        inpainted_overlay_np = inpainted_np * mask_np[..., None] + black_background * (1 - mask_np[..., None])

    # Plot original image
    axs[0].imshow(original_np)
    axs[0].set_title(f"Original Image\nClass: {original_class} ({original_proba:.2%})")
    axs[0].axis("off")

    # Plot overlay image (original + mask)
    axs[1].imshow(overlay_np)
    axs[1].set_title("Original Image + Mask Overlay")
    axs[1].axis("off")

    # Plot mask alone
    axs[2].imshow(interpolated_mask[0][0].cpu(), cmap="gray")
    axs[2].set_title("Interpolated Mask")
    axs[2].axis("off")

    # Plot inpainted image if exists
    if inpainted_image is not None:
        axs[3].imshow(inpainted_image[0].permute(1, 2, 0).cpu())
        if inpaint_target_proba is not None:
            axs[3].set_title(f"Inpainted Image\nTarget: {second_class} ({inpaint_target_proba:.2%})")
        else:
            axs[3].set_title(f"Inpainted Image\nTarget: {second_class}")
        axs[3].axis("off")

    # Plot inpainted image + mask overlay if exists
    if inpainted_image is not None:
        axs[4].imshow(inpainted_overlay_np)
        axs[4].set_title("Inpainted Image + Mask Overlay")
        axs[4].axis("off")

    plt.tight_layout()
    plt.show()
