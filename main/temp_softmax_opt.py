from tqdm import tqdm
from utils import *
from loss_utils import *
from utils.consts import *
from pytorch_lightning import seed_everything
from objectives import objective_temp_softmax

vit_config = config['vit']
loss_config = vit_config['loss']

seed_everything(config['general']['seed'])
feature_extractor, vit_model = load_feature_extractor_and_vit_model(vit_config=vit_config)


def temp_softmax_optimization(vit_model: ViTForImageClassification, feature_extractor: ViTFeatureExtractor, image,
                              num_steps: int, target_class=None) -> Dict[str, Tensor]:
    """
    Return the last layer's attention_scores of the CLS token for each stop point
    Stop points: - 1 minimum prediction loss (maximum probs)
                 - 1 maximum logits
                 - Each 10 iterations from iter 90 to the end ([90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
    """
    vit_ours_model, optimizer = setup_model_and_optimizer(model_name='softmax_temp')
    inputs, original_transformed_image = get_image_and_inputs_and_transformed_image(image=image,
                                                                                    feature_extractor=feature_extractor)
    target = vit_model(**inputs)
    target_class_idx = torch.argmax(target.logits[0])
    total_losses, prediction_losses, correct_class_logits, correct_class_probs, tokens_mask, temps = [], [], [], [], [], []

    for iteration_idx in range(num_steps):
        optimizer.zero_grad()
        output = vit_ours_model(**inputs)
        correct_class_logit, correct_class_prob, prediction_loss = get_iteration_target_class_stats(
            output=output, target_class_idx=target_class_idx)
        loss = objective_temp_softmax(output=output.logits, target=target.logits,
                                      temp=vit_ours_model.vit.encoder.x_attention.clone())
        loss.backward()

        if vit_config['verbose']:
            compare_results_each_n_steps(iteration_idx=iteration_idx, target=target.logits, output=output.logits,
                                         prev_x_attention=vit_ours_model.vit.encoder.x_attention)
        cls_attentions_probs = get_attention_probs_by_layer_of_the_CLS(model=vit_ours_model, layer=-1)
        correct_class_logits.append(correct_class_logit)
        correct_class_probs.append(correct_class_prob)
        prediction_losses.append(prediction_loss)
        total_losses.append(loss.item())
        tokens_mask.append(cls_attentions_probs.clone())
        optimizer.step()

    min_pred_loss_iter, min_total_loss_iter, max_prob_iter, max_logits_iter = get_best_k_values_iterations(
        prediction_losses=prediction_losses, total_losses=total_losses,
        correct_class_probs=correct_class_probs, logits=correct_class_logits, k=1)
    cls_attn_probs_by_stop_points = {'min_pred_loss': tokens_mask[min_pred_loss_iter],
                                     'max_logits': tokens_mask[max_logits_iter]}
    for iter_idx in range(90, 200, 10):
        cls_attn_probs_by_stop_points[f'iter_{iter_idx}'] = tokens_mask[iter_idx]
    return cls_attn_probs_by_stop_points
