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


def temp_softmax_optimization(vit_model: ViTForImageClassification, feature_extractor: ViTFeatureExtractor, image, num_steps: int, target_class=None) -> Tensor:
    """
    Return the last layer's attention_scores of the CLS token for each stop point
    Stop points: [minimum prediction loss, ]
    :param vit_model:
    :param our_model:
    :param inputs:
    :param target_class:
    :return:
    """
    vit_ours_model, optimizer = setup_model_and_optimizer(model_name='softmax_temp')
    inputs, original_transformed_image = get_image_and_inputs_and_transformed_image(image=image,
                                                                                    feature_extractor=feature_extractor)
    target = vit_model(**inputs)
    target_class_idx = torch.argmax(target.logits[0])
    total_losses, prediction_losses, correct_class_logits, correct_class_probs, tokens_mask, temps = [], [], [], [], [], []

    for iteration_idx in tqdm(range(num_steps)):
        optimizer.zero_grad()
        output = vit_ours_model(**inputs)
        correct_class_logit, correct_class_prob, prediction_loss = get_iteration_target_class_stats(
            output=output, target_class_idx=target_class_idx)
        loss = objective_temp_softmax(output=output.logits, target=target.logits,
                         temp=vit_ours_model.vit.encoder.x_attention)
        loss.backward()

        if vit_config['verbose']:
            compare_results_each_n_steps(iteration_idx=iteration_idx, target=target.logits, output=output.logits,
                                         prev_x_attention=vit_ours_model.vit.encoder.x_attention,
                                         sampled_binary_patches=None)
        cls_attentions_probs = get_attention_probs_by_layer_of_the_CLS(model=vit_ours_model, layer=-1)
        correct_class_logits.append(correct_class_logit)
        correct_class_probs.append(correct_class_prob)
        prediction_losses.append(prediction_loss)
        total_losses.append(loss.item())
        tokens_mask.append(cls_attentions_probs.clone())
        optimizer.step()
    return cls_attentions_probs