# from utils.utils import *
from loss_utils import *
from utils.consts import *
from pytorch_lightning import seed_everything
from objectives import objective_grad_rollout

vit_config = config['vit']
loss_config = vit_config['loss']

seed_everything(config['general']['seed'])


def get_rollout_grad(vit_ours_model: ViTForImageClassification, feature_extractor: ViTFeatureExtractor, image=None,
                     inputs=None, discard_ratio: float = 0, return_resized: bool = True) -> Dict[str, Tensor]:
    if inputs is None:
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {'pixel_values': inputs['pixel_values'].to(device)}
    vit_ours_model.to(device)
    output = vit_ours_model(**inputs)
    target_idx = output.logits.argmax().item()
    loss = objective_grad_rollout(output=output.logits, target_idx=target_idx)
    loss.backward()
    attention_probs = get_attention_probs(model=vit_ours_model)
    gradients = get_attention_grads_probs(model=vit_ours_model, apply_relu=False)
    relu_gradients = get_attention_grads_probs(model=vit_ours_model, apply_relu=True)

    rollout_mean_relu_grad = rollout(attentions=attention_probs, head_fusion='mean', gradients=relu_gradients,
                                     discard_ratio=discard_ratio, return_resized=return_resized)
    rollout_max_grad = rollout(attentions=attention_probs, head_fusion='max', gradients=gradients,
                               discard_ratio=discard_ratio, return_resized=return_resized)
    return {'rollout_mean_relu_grad': rollout_mean_relu_grad, 'rollout_max_grad': rollout_max_grad}
