from transformers import BertForSequenceClassification, AutoTokenizer
from objectives import objective_temp_softmax

from tqdm import tqdm
from utils import *
from loss_utils import print_objective_every, log, entropy
import wandb
from log_utils import get_wandb_config
from utils.consts import *
from pytorch_lightning import seed_everything
from typing import Callable
from nlp.bert_utils import *

bert_config = config['bert']
loss_config = bert_config['loss']

seed_everything(config['general']['seed'])
bert_model = handle_model_config_and_freezing_for_task(model=load_BertModel(bert_config, model_type='infer'))
bert_model.eval()

# model = BertForSequenceClassification.from_pretrained(bert_config['model_name'])
# model_name = "textattack/bert-base-uncased-SST-2"
tokenizer = AutoTokenizer.from_pretrained(bert_config['model_name'])
# classifications = ["NEGATIVE", "POSITIVE"]

def bert_objective_temp_softmax(output: Tensor, target: Tensor, temp: Tensor,
                           contrastive_class_idx: Tensor = None) -> Tensor:
    target_idx = contrastive_class_idx if contrastive_class_idx is not None else torch.argmax(target)
    prediction_loss = ce_loss(output, target_idx.unsqueeze(0)) * loss_config['pred_loss_multiplier']
    entropy_loss = entropy(F.softmax(temp, dim=-1)) * loss_config['entropy_loss_multiplier']
    l1_loss = torch.mean(torch.abs(temp)) * loss_config['l1_loss_multiplier']
    print(f'entropy: {entropy_loss}, prediction_loss: {prediction_loss}, l1_loss: {l1_loss}')
    loss = entropy_loss + prediction_loss + l1_loss
    print(f'loss: {loss}, max_temp: {torch.max(temp)}, min_temp: {torch.min(temp)}')
    # log(loss=loss, entropy_loss=entropy_loss, prediction_loss=prediction_loss, x_attention=temp, output=output,
    #     target=target, contrastive_class_idx=target_idx.item())
    return loss

def optimize_params(bert_model, criterion: Callable):
    for idx, image_dict in enumerate(bert_config['text_batch']):

        text, correct_class_idx, contrastive_class_idx = get_text_spec(image_dict)
        wandb_config = get_wandb_config(vit_config=bert_config, experiment_name=experiment_name, image_name=text)
        # with wandb.init(project=config['general']['wandb_project'], entity=config['general']['wandb_entity'],
        #                 config=wandb_config) as run:

        bert_ours_model, optimizer = setup_model_and_optimizer(model_name='softmax_temp')

        # image_plot_folder_path = get_and_create_image_plot_folder_path(images_folder_path=IMAGES_FOLDER_PATH,
        #                                                                experiment_name=experiment_name,
        #                                                                image_name=text)
        print_number_of_trainable_and_not_trainable_params(model=bert_ours_model)
        # save_text_to_file(path=image_plot_folder_path, file_name='metrics_url',
        #                   text=run.url) if run is not None else ''
        # print(run.url)

        encoding, input_ids, attention_mask = get_input_tokens(text=text, tokenizer=tokenizer)
        target = bert_model(input_ids, attention_mask)
        target_class_idx = torch.argmax(target.logits[0])

        total_losses, prediction_losses, correct_class_logits, correct_class_probs, tokens_mask, temps = [], [], [], [], [], []

        for iteration_idx in tqdm(range(bert_config['num_steps'])):
            optimizer.zero_grad()
            output = bert_ours_model(input_ids, attention_mask)

            correct_class_logit, correct_class_prob, prediction_loss = get_iteration_target_class_stats(
                output=output, target_class_idx=target_class_idx)
            temps.append(bert_ours_model.bert.encoder.x_attention.clone())
            loss = criterion(output=output.logits, target=target.logits,
                             temp=bert_ours_model.bert.encoder.x_attention)
            loss.backward()

            compare_results_each_n_steps(iteration_idx=iteration_idx, target=target.logits, output=output.logits,
                                         prev_x_attention=bert_ours_model.bert.encoder.x_attention,
                                         sampled_binary_patches=None)

            cls_attentions_probs = get_attention_probs_by_layer_of_the_CLS(model=bert_model)
            print(cls_attentions_probs)

            correct_class_logits.append(correct_class_logit)
            correct_class_probs.append(correct_class_logit)
            prediction_losses.append(prediction_loss)
            total_losses.append(loss.item())
            tokens_mask.append(cls_attentions_probs.clone())
            optimizer.step()


if __name__ == '__main__':
    experiment_name = 'bert'
    print(experiment_name)
    optimize_params(bert_model=bert_model, criterion=bert_objective_temp_softmax)
