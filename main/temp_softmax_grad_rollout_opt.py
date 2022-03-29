from tqdm import tqdm
from torchvision import transforms
from main.rollout_grad import get_rollout_grad
from utils import *
from loss_utils import *
from utils.consts import *
from pytorch_lightning import seed_everything
from objectives import objective_temp_softmax

vit_config = config['vit']
loss_config = vit_config['loss']

seed_everything(config['general']['seed'])


def temp_softmax_grad_rollout_optimization(vit_ours_model, vit_model, feature_extractor: ViTFeatureExtractor, image,
                                           num_steps: int, target_class=None) -> Dict[str, Tensor]:
    """
    Return the last layer's attention_scores of the CLS token for each stop point
    Stop points: - 1 minimum prediction loss (maximum probs)
                 - 1 maximum logits
                 - Each 10 iterations from iter 90 to the end ([90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
    """
    vit_ours_model.vit.encoder.x_attention.data = nn.Parameter(torch.ones_like(vit_ours_model.vit.encoder.x_attention))
    optimizer = optim.Adam([vit_ours_model.vit.encoder.x_attention], lr=vit_config['lr'])
    vit_ours_model.to(device)
    vit_model.to(device)
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = {'pixel_values': inputs['pixel_values'].to(device)}
    target = vit_model(**inputs)
    target_class_idx = torch.argmax(target.logits[0])
    total_losses, prediction_losses, correct_class_logits, correct_class_probs, tokens_mask_rollout_max_grad, tokens_mask_rollout_mean_relu_grad, temps = [], [], [], [], [], [], []
    d_masks = get_rollout_grad(vit_ours_model=vit_ours_model,
                               feature_extractor=feature_extractor,
                               image=transforms.ToPILImage()(
                                   image.reshape(3, vit_config['img_size'], vit_config['img_size'])),
                               discard_ratio=0.9)
    for iteration_idx in range(num_steps):
        optimizer.zero_grad()
        output = vit_ours_model(**inputs)
        correct_class_logit, correct_class_prob, prediction_loss = get_iteration_target_class_stats(
            output=output, target_class_idx=target_class_idx)
        loss = objective_temp_softmax(output=output.logits, target=target.logits,
                                      temp=vit_ours_model.vit.encoder.x_attention.clone())
        loss.backward()
        cls_attentions_probs = get_attention_probs_by_layer_of_the_CLS(model=vit_ours_model, layer=-1)
        token_mask_rollout_mean_relu_grad = cls_attentions_probs.clone() * d_masks['rollout_mean_relu_grad']
        token_mask_rollout_max_grad = cls_attentions_probs.clone() * d_masks['rollout_max_grad']

        correct_class_logits.append(correct_class_logit)
        correct_class_probs.append(correct_class_prob)
        prediction_losses.append(prediction_loss)
        total_losses.append(loss.item())
        tokens_mask_rollout_max_grad.append(token_mask_rollout_max_grad)
        tokens_mask_rollout_mean_relu_grad.append(token_mask_rollout_mean_relu_grad)
        optimizer.step()

    min_pred_loss_iter, min_total_loss_iter, max_prob_iter, max_logits_iter = get_best_k_values_iterations(
        prediction_losses=prediction_losses, total_losses=total_losses,
        correct_class_probs=correct_class_probs, logits=correct_class_logits, k=1)
    cls_attn_probs_by_stop_points = {'min_pred_loss_rollout_grad_max': tokens_mask_rollout_max_grad[min_pred_loss_iter],
                                     'max_logits_rollout_grad_max': tokens_mask_rollout_max_grad[max_logits_iter],
                                     'min_pred_loss_rollout_mean_relu_grad': tokens_mask_rollout_mean_relu_grad[
                                         min_pred_loss_iter],
                                     'max_logits_rollout_mean_relu_grad': tokens_mask_rollout_mean_relu_grad[
                                         max_logits_iter]}
    for iter_idx in [90, 100, 110, 120, 130, 140, 150, 160, 165, 170, 175, 180, 185, 190]:
        cls_attn_probs_by_stop_points[f'iter_{iter_idx}_rollout_grad_max'] = tokens_mask_rollout_max_grad[iter_idx]
        cls_attn_probs_by_stop_points[f'iter_{iter_idx}_rollout_mean_relu_grad'] = tokens_mask_rollout_mean_relu_grad[iter_idx]
    return cls_attn_probs_by_stop_points
