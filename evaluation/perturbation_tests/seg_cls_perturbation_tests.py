import os
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision import transforms
from transformers.modeling_outputs import ImageClassifierOutput
from diffusers import StableDiffusionInpaintPipeline

from config import config
from config.imagenet1000_clsidx_to_labels import class_idx_to_label
from evaluation.evaluation_utils import calculate_auc
from main.seg_classification.cnns.cnn_utils import CONVENT_NORMALIZATION_MEAN, CONVNET_NORMALIZATION_STD
from config.imagenet1000_clsidx_to_labels import class_idx_to_label
# Global configuration
vit_config = config['vit']
EXPERIMENTS_FOLDER_PATH = vit_config["experiments_path"]

# Device setup
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Initialize inpainting model if configured
if vit_config['use_inpaint_model']:
    print('Initializing inpainting model...')
    inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")


def normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """Normalize image tensor with given mean and std."""
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def eval_perturbation_test(
        model,
        outputs: List[Dict],
        img_size: int = 224,
        perturbation_type: str = "POS",
        perturbation_method: str = "black",
        max_images_to_plot: int = 5,
        save_dir: Optional[Path] = None
) -> Tuple[float, float, List[Dict]]:
    """Evaluate perturbation test for model and outputs.

    Args:
        model: Classification model to evaluate
        outputs: List of batched outputs with images, masks, and targets
        img_size: Size of input images (assumes square)
        perturbation_type: "POS" or "NEG" for positive or negative perturbation
        perturbation_method: Method for perturbation ("black", "mean", "blur", "inpaint")
        max_images_to_plot: Maximum number of images to plot results for
        save_dir: Directory to save plots to (None for no saving)

    Returns:
        Tuple of (class_change_auc, target_prob_auc, sample_results)
        where sample_results is a list of result dictionaries per sample

    Note:
        For a good explanation mask, we expect to see class changes as important
        regions are perturbed. Class changes are recorded as 0, no changes as 1.
        Lower AUC indicates better mask quality.
    """
    # Setup and constants
    model.eval()
    base_size = img_size * img_size
    perturbation_steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Collect all samples for processing
    all_samples = []
    for batch in outputs:
        for data, vis, target in zip(batch["image_resized"], batch["image_mask"], batch["target_class"]):
            all_samples.append({
                "image": data.unsqueeze(0).to(device),
                "mask": vis.unsqueeze(0).to(device),
                "target": target.unsqueeze(0).to(device)
            })

    results = []

    # Process each image
    for sample_idx, sample in enumerate(all_samples):
        image = sample["image"]
        mask = sample["mask"]
        target = sample["target"]

        # Run model on original image
        with torch.no_grad():
            # Normalize the image - important for model prediction
            norm_image = normalize(image.clone())

            pred = model(norm_image)
            pred_logits = pred.logits if hasattr(pred, 'logits') else pred
            pred_probabilities = torch.softmax(pred_logits, dim=1)

            # Get top prediction and probability of target class
            original_top_class = pred_logits.argmax(dim=1).item()
            target_prob = pred_probabilities[0, target.item()].item()

            # Initialize results for this sample
            sample_results = {
                "steps": perturbation_steps,
                "class_changes": [1.0],  # 1 at 0% perturbation (no change from original)
                "target_probs": [target_prob],  # Original probability at 0%
                "top_classes": [original_top_class],
                "predicted_probs": [pred_probabilities.max().item()],
                "perturbed_images": [image],  # Store original image as first step
            }

        # Process mask direction
        vis = mask.clone()
        if perturbation_type == "NEG":
            vis = -vis

        # Flatten mask for processing
        vis = vis.reshape(1, -1)

        # Apply perturbation at different levels
        for step in perturbation_steps[1:]:  # Skip 0% as we already have it
            # Generate perturbed image
            if perturbation_method == "inpaint" and "inpaint_model" in globals():
                perturbed = get_perturbated_inpaint_data(
                    image=image.clone(),
                    mask=vis,
                    perturbation_step=step,
                    base_size=base_size,
                    img_size=img_size
                )
            else:
                perturbed = get_perturbated_data(
                    vis=vis,
                    image=image.clone(),
                    perturbation_step=step,
                    base_size=base_size,
                    img_size=img_size,
                    perturbation_type=perturbation_method
                )

            # Run model on perturbed image
            with torch.no_grad():
                # Normalize the perturbed image for prediction
                norm_perturbed = normalize(perturbed.clone())

                perturbed_pred = model(norm_perturbed)
                perturbed_logits = perturbed_pred.logits if hasattr(perturbed_pred, 'logits') else perturbed_pred
                perturbed_probs = torch.softmax(perturbed_logits, dim=1)

                # Get top prediction for perturbed image
                perturbed_top_class = perturbed_logits.argmax(dim=1).item()
                perturbed_target_prob = perturbed_probs[0, target.item()].item()

                # Record class change (0 if different from original top class, 1 if same)
                # Good masks should lead to class changes as important regions are perturbed
                class_changed = 0.0 if perturbed_top_class != original_top_class else 1.0

                # Add results
                sample_results["class_changes"].append(class_changed)
                sample_results["target_probs"].append(perturbed_target_prob)
                sample_results["top_classes"].append(perturbed_top_class)
                sample_results["predicted_probs"].append(perturbed_probs.max().item())
                sample_results["perturbed_images"].append(perturbed)

        # Plot results if this is one of the first max_images_to_plot samples
        if sample_idx < max_images_to_plot:
            plot_perturbation_results(
                sample_results=sample_results,
                target_class=target.item(),
                original_class=original_top_class,
                method=perturbation_method,
                sample_idx=sample_idx,
                save_dir=save_dir
            )

        results.append(sample_results)

    # Calculate AUC scores
    class_changes_auc = calculate_perturbation_auc(results, "class_changes")
    target_probs_auc = calculate_perturbation_auc(results, "target_probs")

    # Plot the aggregate results across all samples
    if len(results) > 1:
        plot_aggregate_perturbation_results(
            results=results,
            method=perturbation_method,
            save_dir=save_dir
        )

    return class_changes_auc, target_probs_auc, results


def calculate_perturbation_auc(results, key="class_changes"):
    """Calculate AUC for perturbation results.

    Args:
        results: List of result dictionaries per sample
        key: Which metric to use ('class_changes' or 'target_probs')

    Returns:
        AUC score (0-100)

    Note:
        Lower AUC values indicate better explanation masks:
        - For class_changes: 0 means class changed (good), 1 means no change (bad)
        - For target_probs: Lower values as perturbation increases means good masks

    Process:
        1. For each image, create a curve of values across perturbation steps
        2. Average these curves across all images at each perturbation step
        3. Calculate the area under the averaged curve (AUC)
    """
    # Extract values for each step across all samples
    steps = results[0]["steps"]  # Perturbation steps are the same for all samples

    # For target_probs, we expect decreasing values as perturbation increases
    # For class_changes, we want to see more values of 0 (class changed)
    if key == "target_probs":
        # For target probs, higher is better, so we take 1-value for AUC calculation
        # Lower AUC means better masks
        values = [[1.0 - val for val in sample[key]] for sample in results]
    else:
        # For class changes, values are already set: 0 for changed, 1 for unchanged
        # So keep as-is - lower AUC is better (more class changes)
        values = [sample[key] for sample in results]

    # Print individual curves for debugging
    print(f"\nCalculating AUC for {key} across {len(values)} images:")
    for i, sample_values in enumerate(values):
        print(f"  Image {i + 1}: {[round(v, 2) for v in sample_values]}")

    # Average across samples for each step
    avg_values = [
        sum(sample[i] for sample in values) / len(values)
        for i in range(len(steps))
    ]

    print(f"  Average: {[round(v, 2) for v in avg_values]}")

    # Calculate AUC using trapezoidal rule
    auc = np.trapz(y=avg_values, x=steps) * 100

    print(f"  AUC: {auc:.2f}")

    return auc


def plot_perturbation_results(sample_results, target_class, original_class, method, sample_idx, save_dir=None):
    """Plot perturbation results for a single sample.

    Args:
        sample_results: Results dictionary for the sample
        target_class: Target class index
        original_class: Original predicted class index
        method: Perturbation method used
        sample_idx: Sample index for saving
        save_dir: Directory to save plot to (None for no saving)
    """
    steps = sample_results["steps"]
    class_changes = sample_results["class_changes"]
    target_probs = sample_results["target_probs"]
    top_classes = sample_results["top_classes"]
    predicted_probs = sample_results["predicted_probs"]
    perturbed_images = sample_results["perturbed_images"]

    # Create figure with two rows - plots and images
    fig = plt.figure(figsize=(15, 10))

    # First row: Metrics plots
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax1.plot(steps, class_changes, 'ro-', label='Class Unchanged (0=class changed=good, 1=unchanged=bad)')
    ax1.plot(steps, target_probs, 'bo-', label='Target Class Probability')
    ax1.set_xlabel('Perturbation Amount')
    ax1.set_ylabel('Value')
    ax1.set_title(f'Perturbation Results - {method.title()} (lower AUC is better)')
    ax1.grid(True)
    ax1.legend()

    # Second row: Image grid
    num_images = min(len(perturbed_images), 6)  # Show at most 6 images
    image_indices = np.linspace(0, len(perturbed_images) - 1, num_images).astype(int)

    for i, img_idx in enumerate(image_indices):
        ax = plt.subplot2grid((2, num_images), (1, i))
        step = steps[img_idx]
        img = perturbed_images[img_idx]
        cls = top_classes[img_idx]
        prob = predicted_probs[img_idx]

        # Determine if class has changed from original
        class_status = "Changed" if cls != original_class else "Same"

        # Show image
        ax.imshow(img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
        ax.set_title(f"{int(step * 100)}%\nClass: {cls} ({class_status})\nProb: {prob:.2f}")
        ax.axis('off')

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir / f"sample_{sample_idx}_{method}.png")

    plt.show()
    plt.close()


def _normalize_input(data, is_convnet=False):
    """Helper to normalize model inputs based on model type."""
    if is_convnet:
        return normalize(data, mean=CONVENT_NORMALIZATION_MEAN, std=CONVNET_NORMALIZATION_STD)
    else:
        return normalize(data)


def _print_prediction_details(logits, probabilities, target, description):
    """Helper to print model prediction details."""
    top_class = logits[0].argmax(dim=0).item()
    max_logit = round(logits[0].max(dim=0)[0].item(), 2)
    max_prob = round(probabilities[0].max(dim=0)[0].item(), 5)
    target_logit = round(logits[0][target].item(), 2)
    target_prob = round(probabilities[0][target].item(), 5)

    print(f'\n{description}. Target: {target.item()}. Top Class: {top_class}, '
          f'Max logits: {max_logit}, Max prob: {max_prob}; '
          f'Correct class logit: {target_logit} Correct class prob: {target_prob}')


def get_auc(num_correct_pertub, num_correct_model) -> float:
    """Calculate AUC for perturbation test.

    Args:
        num_correct_pertub: Array of correct predictions for perturbed images (steps x samples)
        num_correct_model: Array of correct predictions for original images (samples)

    Returns:
        AUC score (0-100)
    """
    # Get mean accuracy for each perturbation level
    mean_accuracy_by_step = np.mean(num_correct_pertub, axis=1)

    # Insert original model accuracy at beginning
    mean_accuracy_original = np.mean(num_correct_model)
    mean_accuracy_by_step = np.insert(mean_accuracy_by_step, 0, mean_accuracy_original)

    # Create x-axis values (perturbation levels from 0 to 1)
    perturbation_levels = np.linspace(0, 1, len(mean_accuracy_by_step))

    try:
        # Calculate AUC using sklearn's auc function
        auc = calculate_auc(mean_accuracy_by_step=mean_accuracy_by_step) * 100
    except ValueError as e:
        print(f"AUC calculation error: {e}")
        print(f"Shape of accuracy array: {mean_accuracy_by_step.shape}")
        print(f"Values: {mean_accuracy_by_step}")

        # Handle case where arrays have inconsistent shapes
        if "inconsistent" in str(e) and "samples" in str(e):
            print("Handling inconsistent samples by using trapezoidal rule directly")
            # Manually calculate AUC using trapezoidal rule
            auc = np.trapz(y=mean_accuracy_by_step, x=perturbation_levels)
            auc = auc * 100  # Scale to 0-100 range
        else:
            # For other errors, return a default value
            print("Returning default AUC value of 50")
            auc = 50.0

    return auc


def get_perturbated_data(
        vis: Tensor,
        image: Tensor,
        perturbation_step: Union[float, int],
        base_size: int,
        img_size: int,
        perturbation_type: str = "black"
) -> Tensor:
    """Apply perturbation mask to image by setting top-k pixels to specified value.

    Args:
        vis: Visibility/importance map for pixels
        image: Original image tensor
        perturbation_step: Percentage of pixels to perturb (0.0-1.0)
        base_size: Total number of pixels (img_size²)
        img_size: Image size (height/width)
        perturbation_type: Type of perturbation to apply:
            - "black": Set pixels to 0 (black)
            - "mean": Replace with mean image color
            - "blur": Apply Gaussian blur to selected pixels

    Returns:
        Perturbed image tensor
    """
    _data = image.clone()
    org_shape = (1, 3, img_size, img_size)

    # Select top-k indices based on importance
    k = int(base_size * perturbation_step)
    _, idx = torch.topk(vis, k, dim=-1)

    # Reshape data for processing
    _data = _data.reshape(org_shape[0], org_shape[1], -1)

    # Apply perturbation based on type
    if perturbation_type == "black":
        # Set pixels to black (0)
        idx = idx.unsqueeze(1).repeat(1, org_shape[1], 1)
        _data = _data.scatter_(-1, idx.reshape(1, org_shape[1], -1), 0)

    elif perturbation_type == "mean":
        # Calculate mean values for each channel
        mean_values = _data.mean(dim=2, keepdim=True)  # Shape: [1, 3, 1]
        idx_expanded = idx.unsqueeze(1).repeat(1, org_shape[1], 1)

        for i in range(org_shape[1]):
            channel_idx = idx_expanded[:, i, :].reshape(1, -1)
            # Create tensor of repeated mean values of the right shape
            channel_mean = mean_values[:, i, :].expand(1, channel_idx.size(1))
            _data[:, i].scatter_(-1, channel_idx, channel_mean)

    elif perturbation_type == "blur":
        # First reshape back to image format
        _data = _data.reshape(*org_shape)

        # Create a binary mask (1 for pixels to blur, 0 for unchanged)
        mask = torch.zeros((1, 1, img_size, img_size), device=_data.device)
        mask_flat = mask.reshape(1, -1)
        mask_flat.scatter_(-1, idx, 1.0)

        # Apply Gaussian blur using torchvision instead of F.gaussian_blur
        from torchvision import transforms

        # Define a blur transform with appropriate sigma
        blur_transform = transforms.GaussianBlur(kernel_size=11, sigma=5.0)

        # Apply blur transform to the image
        try:
            blurred = blur_transform(_data)
        except Exception as e:
            print(f"Error applying blur: {e}")
            # Fallback to smaller kernel size
            blurred = transforms.GaussianBlur(kernel_size=7, sigma=3.0)(_data)

        # Combine original and blurred image using the mask
        mask = mask.expand(-1, 3, -1, -1)  # Expand to match channels
        _data = _data * (1 - mask) + blurred * mask

        # No need to reshape at the end
        return _data

    else:
        raise ValueError(f"Unsupported perturbation type: {perturbation_type}")

    # Reshape back to image format
    _data = _data.reshape(*org_shape)

    return _data


def inpaint(
        image: Tensor,
        binary_mask: Tensor,
        top2_classes: Optional[Tensor] = None
) -> Image.Image:
    """Run inpainting on image using binary mask and class prompt.

    Args:
        image: Original image tensor
        binary_mask: Binary mask tensor (1=inpaint, 0=keep)
        top2_classes: Top 2 class predictions to guide inpainting

    Returns:
        PIL Image with inpainted regions
    """
    # Resize image and mask for the inpainting model (512x512)
    # Make sure image is unbatched before conversion to PIL
    if image.dim() == 4:
        image = image.squeeze(0)

    image_pil = to_pil_image(image.cpu()).resize((512, 512), Image.BICUBIC)

    # Handle mask shape
    if binary_mask.dim() == 4:
        binary_mask = binary_mask.squeeze(0)
    if binary_mask.dim() == 3 and binary_mask.size(0) > 1:
        binary_mask = binary_mask[0]  # Take first channel if multi-channel

    mask_pil = to_pil_image(binary_mask.cpu()).resize((512, 512), Image.NEAREST)

    # Get class information for prompting
    if top2_classes is not None:
        original_idx = top2_classes.indices[0][0].item()
        second_idx = top2_classes.indices[0][1].item()
        original_class = class_idx_to_label[original_idx]
        target_class = class_idx_to_label[second_idx]

        print(f"Original predicted class: '{original_class}' (index {original_idx})")
        print(f"Target (second) predicted class: '{target_class}' (index {second_idx})")

        # Use second highest class as inpainting prompt
        prompt = str(target_class)
    else:
        prompt = ""  # Fallback to empty prompt

    # Generate inpainting
    with torch.no_grad():
        result = inpaint_model(
            prompt=prompt,
            image=image_pil,
            mask_image=mask_pil,
            guidance_scale=7.5
        ).images[0]

    # Resize result back to original dimensions
    result_resized = result.resize((224, 224), Image.BICUBIC)
    return result_resized


def get_perturbated_inpaint_data(
        image: Tensor,
        mask: Tensor,
        perturbation_step: Union[float, int],
        base_size: int,
        img_size: int,
        top2_classes: Optional[Tensor] = None
) -> Tensor:
    """Apply perturbation via inpainting instead of masking.

    Args:
        image: Original image tensor
        mask: Importance mask tensor
        perturbation_step: Percentage of pixels to inpaint (0.0-1.0)
        base_size: Total number of pixels (img_size²)
        img_size: Image size (height/width)
        top2_classes: Top 2 class predictions to guide inpainting

    Returns:
        Tensor with inpainted regions
    """
    org_shape = (1, 3, img_size, img_size)

    # Select top-k indices based on importance
    k = int(base_size * perturbation_step)
    mask_flat = mask.reshape(1, -1)
    _, idx = torch.topk(mask_flat, k, dim=-1)

    # Create binary inpainting mask (1 where we want to inpaint)
    binary_mask = torch.zeros((1, 1, img_size, img_size), device=mask.device)
    flat_binary_mask = binary_mask.view(1, -1)
    flat_binary_mask.scatter_(-1, idx, 1.0)

    # Ensure binary mask values are either 0 or 1
    binary_mask = (binary_mask > 0.5).float()

    # Run inpainting
    inpainted_img = inpaint(image=image, binary_mask=binary_mask, top2_classes=top2_classes)

    # Convert PIL image back to tensor
    inpainted_tensor = to_tensor(inpainted_img).unsqueeze(0).to(image.device)
    return inpainted_tensor


def move_to_device_data_vis_and_target(data, target=None, vis=None) -> Dict:
    """Move data, vis, (and optionally target) to device.

    Args:
        data: Image tensor
        target: Target class tensor (optional)
        vis: Visibility/explanation mask tensor

    Returns:
        Dictionary with moved tensors
    """
    data = data.to(device)
    vis = vis.to(device)
    result = dict(data=data, vis=vis)
    if target is not None:
        result["target"] = target.to(device)
    return result


def plot_image(image, title="Input Image") -> None:
    """Plot image tensor using matplotlib.

    Args:
        image: Image tensor to plot [C,H,W] or [B,C,H,W]
        title: Title for the plot
    """
    image = image if len(image.shape) == 3 else image.squeeze(0)
    plt.figure(figsize=(8, 8))
    plt.imshow(image.cpu().detach().permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_mask(mask, title="Explanation Mask", colormap='jet', with_colorbar=True) -> None:
    """Plot mask tensor using matplotlib.

    Args:
        mask: Mask tensor to plot [1,H,W] or [B,1,H,W]
        title: Title for the plot
        colormap: Colormap to use for visualization
        with_colorbar: Whether to include a colorbar
    """
    mask = mask if len(mask.shape) == 2 else mask.squeeze()
    plt.figure(figsize=(8, 8))
    im = plt.imshow(mask.cpu().detach().numpy(), cmap=colormap)
    plt.title(title)
    plt.axis('off')
    if with_colorbar:
        plt.colorbar(im)
    plt.show()


def plot_comparison(original_img, mask, masked_img, inpainted_img=None, titles=None) -> None:
    """Plot comparison of original, mask, masked, and optionally inpainted images.

    Args:
        original_img: Original image tensor
        mask: Explanation mask tensor
        masked_img: Masked image tensor
        inpainted_img: Optional inpainted image tensor
        titles: Optional custom titles for the plots
    """
    n_plots = 4 if inpainted_img is not None else 3
    fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * 4, 4))

    # Set default titles if not provided
    if titles is None:
        titles = ["Original Image", "Explanation Mask", "Masked Image"]
        if inpainted_img is not None:
            titles.append("Inpainted Image")

    # Plot original image
    original_img = original_img if len(original_img.shape) == 3 else original_img.squeeze(0)
    axs[0].imshow(original_img.cpu().detach().permute(1, 2, 0))
    axs[0].set_title(titles[0])
    axs[0].axis('off')

    # Plot mask
    mask = mask if len(mask.shape) == 2 else mask.squeeze()
    im = axs[1].imshow(mask.cpu().detach().numpy(), cmap='jet')
    axs[1].set_title(titles[1])
    axs[1].axis('off')
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

    # Plot masked image
    masked_img = masked_img if len(masked_img.shape) == 3 else masked_img.squeeze(0)
    axs[2].imshow(masked_img.cpu().detach().permute(1, 2, 0))
    axs[2].set_title(titles[2])
    axs[2].axis('off')

    # Plot inpainted image if provided
    if inpainted_img is not None:
        inpainted_img = inpainted_img if len(inpainted_img.shape) == 3 else inpainted_img.squeeze(0)
        axs[3].imshow(inpainted_img.cpu().detach().permute(1, 2, 0))
        axs[3].set_title(titles[3])
        axs[3].axis('off')

    plt.tight_layout()
    plt.show()


def update_results_df(results_df: pd.DataFrame, vis_type: str, auc: float) -> pd.DataFrame:
    """Append new result row to results DataFrame.

    Args:
        results_df: Existing results DataFrame
        vis_type: Visualization type identifier
        auc: AUC score to record

    Returns:
        Updated DataFrame with new result
    """
    # Extract perturbation method from vis_type if present
    parts = vis_type.split('_')
    perturbation_method = parts[-1] if len(parts) > 0 and parts[-1] in ["black", "blue", "mean", "inpaint"] else "black"

    return results_df.append({
        'vis_type': vis_type,
        'perturbation_method': perturbation_method,
        'auc': auc
    }, ignore_index=True)


def save_obj_to_disk(path, obj) -> None:
    """Save Python object to disk as pickle.

    Args:
        path: File path to save to
        obj: Object to save
    """
    import pickle
    if isinstance(path, str) and not path.endswith('.pkl'):
        path += '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def save_best_auc_objects_to_disk(path, auc: float, vis, original_image, epoch_idx: int) -> None:
    """Save best AUC objects to disk for later analysis.

    Args:
        path: File path to save to
        auc: AUC score
        vis: Visibility/explanation mask
        original_image: Original input image
        epoch_idx: Training epoch index
    """
    obj_dict = {
        'auc': auc,
        'vis': vis,
        'original_image': original_image,
        'epoch_idx': epoch_idx
    }
    save_obj_to_disk(path=path, obj=obj_dict)


def run_perturbation_test(
        model,
        outputs,
        perturbation_method: str = "black",
        num_images_to_plot: int = 5,
        output_dir: Optional[Path] = None,
) -> Tuple[float, float]:
    """Run perturbation test on a model with the given outputs.

    Args:
        model: The model to evaluate
        outputs: List of dictionaries with image_resized, image_mask, target_class
        perturbation_method: Method to use for perturbation ("black", "mean", "blur", "inpaint")
        num_images_to_plot: Number of images to plot results for
        output_dir: Directory to save plots to (None for no saving)

    Returns:
        Tuple of (class_change_auc, target_prob_auc)

    Note:
        Lower AUC values indicate better explanation masks. Good masks should lead to:
        1. More class changes as important regions are perturbed (recorded as 0 in data)
        2. Decreasing target probability as important regions are perturbed
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Run the evaluation
    class_change_auc, target_prob_auc, sample_results = eval_perturbation_test(
        model=model,
        outputs=outputs,
        perturbation_method=perturbation_method,
        max_images_to_plot=num_images_to_plot,
        save_dir=output_dir
    )

    # Print results
    print(f"Perturbation results for {perturbation_method}:")
    print(f"  Class change AUC: {class_change_auc:.2f} (lower is better - more class changes)")
    print(f"  Target probability AUC: {target_prob_auc:.2f} (lower is better - more prob drops)")
    print(f"  These AUC values are averaged across {len(sample_results)} images")

    return class_change_auc, target_prob_auc


def run_perturbation_test_opt(
        model,
        outputs,
        stage: str,
        epoch_idx: int,
        is_convnet: bool,
        verbose: bool,
        img_size: int,
        experiment_path=None,
        perturbation_type: str = "POS",
        perturbation_method: str = "black",
) -> float:
    """Run optimized perturbation test without saving to CSV.

    Lightweight version of run_perturbation_test that doesn't save results.

    Args:
        model: Classification model to evaluate
        outputs: List of dicts with image_resized, image_mask, target_class
        stage: Stage identifier (train/val/test)
        epoch_idx: Training epoch index
        is_convnet: Whether the model is a ConvNet
        verbose: Whether to print verbose output
        img_size: Size of input images
        experiment_path: Path to save experiment results
        perturbation_type: "POS" or "NEG" for direction of perturbation
        perturbation_method: Method for perturbation ("black", "mean", "blur", "inpaint")

    Returns:
        AUC score from perturbation test
    """
    if experiment_path is None:
        experiment_path = Path(EXPERIMENTS_FOLDER_PATH, vit_config['evaluation']['experiment_folder_name'])

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path, exist_ok=True)

    # Set model to eval mode
    model.eval()

    # Run evaluation
    vit_type_experiment_path = Path(experiment_path, f'{stage}_vis_seg_cls_epoch_{epoch_idx}_{perturbation_method}')
    auc = eval_perturbation_test(
        model=model,
        outputs=outputs,
        img_size=img_size,
        is_convnet=is_convnet,
        verbose=verbose,
        perturbation_type=perturbation_type,
        perturbation_method=perturbation_method,
        save_dir='/home/amitesh/Projects/explainablity-transformer-cv/dummy_plots/' + str(vit_type_experiment_path)
    )

    return auc


def main() -> None:
    """Example usage of perturbation test.

    Demonstrates the following workflow:
    1. Get a mask with importance scores
    2. Take top K values based on perturbation step
    3. Calculate class changes and target class probability
    4. Calculate AUC over these metrics
    5. Plot results for multiple images
    """
    import torch
    from pathlib import Path
    import pandas as pd
    import os
    from transformers import ViTForImageClassification
    from PIL import Image
    from torchvision import transforms
    import glob
    import traceback

    print("\n=== Explainability Perturbation Test ===\n")
    print("Evaluating mask quality - lower AUC indicates better masks (more class changes when perturbed)\n")

    # Load ground truth classes from CSV
    gt_file = "/home/amitesh/Projects/explainablity-transformer-cv/gt_data_imagenet/val_ground_truth_2012.csv"
    if os.path.exists(gt_file):
        print(f"Loading ground truth classes from {gt_file}")
        gt_df = pd.read_csv(gt_file)
        gt_data = {row['img_name']: int(row['label']) for _, row in gt_df.iterrows()}
    else:
        print(f"Ground truth file {gt_file} not found. Will use dummy targets.")
        gt_data = {}

    # Initialize model
    print("Loading pre-trained ViT model...")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model.to(device)

    # Load multiple example images (using synthetic ones if real ones not available)
    print("Loading example images...")
    try:
        # Try to find some ImageNet validation images
        image_paths = glob.glob("/home/amitesh/Projects/explainablity-transformer-cv/vit_data/*.JPEG")[:10]
        if not image_paths:
            raise FileNotFoundError("No images found")
    except:
        # If not found, create synthetic images
        print("No real images found, creating synthetic ones...")
        image_paths = ["synthetic"] * 5

    # Prepare images and masks
    all_images = []
    all_masks = []
    all_targets = []

    for i, path in enumerate(image_paths):
        if path == "synthetic":
            # Create synthetic image (random noise)
            img_tensor = torch.rand(3, 224, 224)
            # Create synthetic mask (center blob)
            Y, X = torch.meshgrid(torch.arange(224), torch.arange(224), indexing="ij")
            center_x, center_y = 112, 112
            radius = 60
            mask_tensor = (((X - center_x) ** 2 + (Y - center_y) ** 2) < radius ** 2).float()
            # Use index as target class
            target = torch.tensor([i % 10])
        else:
            # Load and preprocess real image
            image = Image.open(path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            img_tensor = transform(image)

            # Create explanation mask (in real scenario, this comes from your explainer)
            # Here we just create a circular mask as example
            Y, X = torch.meshgrid(torch.arange(224), torch.arange(224), indexing="ij")
            center_x, center_y = 112, 112
            radius = 60
            mask_tensor = (((X - center_x) ** 2 + (Y - center_y) ** 2) < radius ** 2).float()

            # Get ground truth class from CSV if available, otherwise use 0
            img_name = os.path.basename(path)
            if img_name in gt_data:
                target = torch.tensor([gt_data[img_name]])
                print(f"Image {img_name}: Using ground truth class {gt_data[img_name]}")
            else:
                target = torch.tensor([0])
                print(f"Image {img_name}: No ground truth found, using default class 0")

        all_images.append(img_tensor)
        all_masks.append(mask_tensor.unsqueeze(0))  # Add channel dimension
        all_targets.append(target)

    # Create batch
    batch = {
        "image_resized": torch.stack(all_images),
        "image_mask": torch.stack(all_masks),
        "target_class": torch.cat(all_targets)
    }

    # Create outputs structure
    outputs = [batch]

    # Create output directory
    output_dir = Path(EXPERIMENTS_FOLDER_PATH) / "perturbation_tests"
    print(f"Output directory for results: {output_dir}")
    # Run perturbation tests with different methods
    methods = ["black", "mean", "blur"]
    results = {}

    for method in methods:
        try:
            print(f"\nTesting perturbation method: {method}")
            class_auc, prob_auc = run_perturbation_test(
                model=model,
                outputs=outputs,
                perturbation_method=method,
                num_images_to_plot=5,
                output_dir=output_dir / method
            )
            results[method] = (class_auc, prob_auc)
        except Exception as e:
            print(f"Error with method {method}: {e}")
            print("Detailed error:")
            traceback.print_exc()

    # Run inpainting if available
    if "inpaint_model" in globals():
        try:
            print("\nTesting inpainting perturbation")
            class_auc, prob_auc = run_perturbation_test(
                model=model,
                outputs=outputs,
                perturbation_method="inpaint",
                num_images_to_plot=5,
                output_dir=output_dir / "inpaint"
            )
            results["inpaint"] = (class_auc, prob_auc)
        except Exception as e:
            print(f"Error with inpainting: {e}")
            print("Detailed error:")
            traceback.print_exc()
    else:
        print("\nInpainting model not available. Skipping inpainting test.")

    # Compare results
    if results:
        print("\n=== Perturbation Test Results Summary ===")
        print("Method      | Class Change AUC   | Target Prob AUC")
        print("            | (lower = better)   | (lower = better)")
        print("-" * 50)
        for method, (class_auc, prob_auc) in results.items():
            print(f"{method:11s} | {class_auc:15.2f} | {prob_auc:14.2f}")
    else:
        print("\n=== No successful perturbation tests were completed ===")

    print("\n=== Perturbation test complete ===\n")


def plot_aggregate_perturbation_results(results, method, save_dir=None):
    """Plot aggregate perturbation results across all samples.

    Args:
        results: List of result dictionaries for all samples
        method: Perturbation method used
        save_dir: Directory to save plot to (None for no saving)
    """
    if not results:
        print("No results to plot")
        return

    # Extract steps - should be the same for all results
    steps = results[0]["steps"]

    # Extract and average class changes and target probabilities
    class_changes_all = [sample["class_changes"] for sample in results]
    target_probs_all = [sample["target_probs"] for sample in results]

    # Calculate average values at each step
    avg_class_changes = [
        sum(sample[i] for sample in class_changes_all) / len(class_changes_all)
        for i in range(len(steps))
    ]

    avg_target_probs = [
        sum(sample[i] for sample in target_probs_all) / len(target_probs_all)
        for i in range(len(steps))
    ]

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot average metrics
    plt.plot(steps, avg_class_changes, 'ro-',
             label='Class Unchanged - Avg (0=changed=good, 1=unchanged=bad)')
    plt.plot(steps, avg_target_probs, 'bo-',
             label='Target Probability - Avg')

    # Add individual image curves with transparency
    for i, (class_changes, target_probs) in enumerate(zip(class_changes_all, target_probs_all)):
        plt.plot(steps, class_changes, 'r--', alpha=0.2)
        plt.plot(steps, target_probs, 'b--', alpha=0.2)

    # Calculate AUC for the average curves
    class_auc = np.trapz(y=avg_class_changes, x=steps) * 100
    target_auc = np.trapz(y=[1 - p for p in avg_target_probs], x=steps) * 100

    plt.xlabel('Perturbation Amount')
    plt.ylabel('Value')
    plt.title(f'Aggregate Perturbation Results - {method.title()}\n' +
              f'Class Change AUC: {class_auc:.2f}, Target Prob AUC: {target_auc:.2f}\n' +
              f'(Lower is better - {len(results)} images)')
    plt.grid(True)
    plt.legend()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir / f"aggregate_{method}.png")

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
