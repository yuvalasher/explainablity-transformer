import os
import gc
import random
import logging
import argparse
from itertools import chain
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from nlp.BERT_rationale_benchmark.metrics import token_prf_f1_macro, Rationale
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from transformers import AutoTokenizer, BertTokenizer
from nlp.BERT_rationale_benchmark.metrics import cal_metrics
from nlp.BERT_rationale_benchmark.models.pipeline.bert_pipeline import extract_docid_from_dataset_element
from nlp.datagetters.datagetters import load_datasets, load_documents
from nlp.BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from loss_utils import entropy

from utils.consts import *
from pytorch_lightning import seed_everything
from typing import Callable
from nlp.bert_utils import *
from utils import *
from collections import OrderedDict
import optuna
from optuna.trial import TrialState
import wandb

os.chdir('/home/amiteshel1/Projects/explainablity-transformer')
bert_config = config['bert']
loss_config = bert_config['loss']
ROOT_DIR = os.environ['ROOT_DIR']
seed_everything(config['general']['seed'])
BATCH_FIRST = True

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
logger = logging.getLogger(__name__)

date_exp_name = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")

wandb.init(project="my-test-project",
           name=f'''{date_exp_name}__pred_loss_{loss_config['pred_loss_multiplier']}__entropy_loss_{loss_config['entropy_loss_multiplier']}''')
wandb.config.update(bert_config)


def update_num_tokens_x_attenion(bert_ours_model, input_ids, device):
    input_num_tokens = input_ids.shape[1]
    bert_ours_model.bert.encoder.num_tokens = input_num_tokens
    bert_ours_model.bert.encoder.x_attention = nn.Parameter(
        torch.ones(bert_ours_model.config.num_hidden_layers, bert_ours_model.config.num_attention_heads,
                   input_num_tokens, requires_grad=True, device=device))
    return


def generate_scoring_by_head_agg(cls_attentions_probs, input_ids, tokenizer, top_k: int = 10, agg: str = 'median',
                                 print_tokens=False):
    if agg == 'median':
        scores = cls_attentions_probs.median(dim=0)[0]
    else:
        scores = cls_attentions_probs.mean(dim=0)

    top_k_tokens = torch.topk(scores, k=cls_attentions_probs.shape[-1] - 1, largest=True)[1].tolist()
    if print_tokens:
        for token_id in top_k_tokens[:top_k]:
            score = np.round(scores[token_id].item(), 5)
            print(tokenizer.convert_ids_to_tokens(input_ids[1:])[token_id], score)
    return top_k_tokens


def setup_model_and_optimizer(model_type: str, device: str):
    vit_ours_model = handle_model_config_and_freezing_for_task(
        model=load_BertModel(bert_config, model_type=model_type, device=device),
        freezing_transformer=bert_config['freezing_transformer'])
    optimizer = optim.Adam([vit_ours_model.bert.encoder.x_attention], lr=bert_config['evidence_classifier']['lr'])
    return vit_ours_model, optimizer


def load_data():
    # train, val, test = load_datasets('C:/Users/amitx/Documents/Amit/Projects_Python/explainablity-transformer/bert_models/movies/movies')
    train = pd.read_csv(ROOT_DIR + 'datasets/imdb/train.csv').sample(frac=1)
    test = pd.read_csv(ROOT_DIR + 'datasets/imdb/test.csv').sample(frac=1)

    return train, test


def load_interning_documents(cache, documents, model_params, tokenizer):
    if os.path.exists(cache):
        logger.info(f'Loading interned documents from {cache}')
        (interned_documents) = torch.load(cache)
    else:
        logger.info(f'Interning documents')
        interned_documents = {}
        for d, doc in tqdm(documents.items()):
            encoding = tokenizer.encode_plus(
                doc,
                add_special_tokens=True,
                max_length=model_params['max_length'],
                return_token_type_ids=False,
                padding=False,
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True,
            )
            interned_documents[d] = encoding

        torch.save((interned_documents), cache)
    return interned_documents


def load_doc_ids(test, train, val):
    return set(e.docid for e in
               chain.from_iterable(chain.from_iterable(map(lambda ann: ann.evidences, chain(train, val, test)))))


def get_pred_class_per_top_k(attention_mask_temp, filter_out_indices, input_ids_temp, test_classifier):
    input_ids_temp[0, filter_out_indices] = 0
    attention_mask_temp[0, filter_out_indices] = 0

    output = test_classifier(input_ids_temp, attention_mask_temp)
    pred_class = torch.argmax(output[0]).item()
    return pred_class


def _get_args():
    parser = argparse.ArgumentParser(description="""Trains a pipeline model.

        Step 1 is evidence identification, that is identify if a given sentence is evidence or not
        Step 2 is evidence classification, that is given an evidence sentence, classify the final outcome for the final task
         (e.g. sentiment or significance).

        These models should be separated into two separate steps, but at the moment:
        * prep data (load, intern documents, load json)
        * convert data for evidence identification - in the case of training data we take all the positives and sample some
          negatives
            * side note: this sampling is *somewhat* configurable and is done on a per-batch/epoch basis in order to gain a
              broader sampling of negative values.
        * train evidence identification
        * convert data for evidence classification - take all rationales + decisions and use this as input
        * train evidence classification
        * decode first the evidence, then run classification for each split

        """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data_dir', dest='data_dir', required=True,
                        help='Which directory contains a {train,val,test}.jsonl file?')
    parser.add_argument('--output_dir', dest='output_dir', required=True,
                        help='Where shall we write intermediate models + final data to?')
    parser.add_argument('--model_params', dest='model_params', required=False,
                        help='JSoN file for loading arbitrary model parameters (e.g. optimizers, pre-saved files, etc.')
    args = parser.parse_args()
    return args


def _get_files_pathes(args):
    os.makedirs(args.output_dir, exist_ok=True)
    evidence_classifier_output_dir = os.path.join(args.output_dir, 'classifier')
    os.makedirs(evidence_classifier_output_dir, exist_ok=True)

    cache = os.path.join(args.output_dir, 'preprocessed.pkl')
    model_hila_save_file = os.path.join(evidence_classifier_output_dir, 'classifier.pt')

    # model_save_file = os.path.join(evidence_classifier_output_dir, 'classifier.pt')
    # epoch_save_file = os.path.join(evidence_classifier_output_dir, 'classifier_epoch_data.pt')
    return cache, model_hila_save_file


def close_files(result_files, result_files_b):
    for res, k in enumerate(range(5, 85, 5)):
        result_files[res].close()
        result_files_b[res].close()


def plots_all_metrics(results_save_dir):
    arr_a_instance_macro = []
    arr_a_instance_micro = []
    for k in range(5, 85, 5):
        path_json = f'{results_save_dir}/temp/metrics_results_{k}.json'
        f = open(path_json)
        data = json.load(f)
        arr_a_instance_macro.append(data['token_prf']['instance_macro']['f1'])
        arr_a_instance_micro.append(data['token_prf']['instance_micro']['f1'])
    arr_b_instance_macro = []
    arr_b_instance_micro = []
    for k in range(5, 85, 5):
        path_json = f'{results_save_dir}/our/metrics_results_{k}.json'
        f = open(path_json)
        data = json.load(f)
        arr_b_instance_macro.append(data['token_prf']['instance_macro']['f1'])
        arr_b_instance_micro.append(data['token_prf']['instance_micro']['f1'])
    plt.figure()
    plt.plot(range(5, 85, 5), arr_a_instance_macro, label='model_a')
    plt.plot(range(5, 85, 5), arr_b_instance_macro, label='model_b')
    plt.legend()
    plt.grid()
    plt.title('token_prf -instance_macro - f1')
    plt.savefig(
        f'{results_save_dir}/{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}_token_prf_instance_macro_f1.png')

    plt.figure()
    plt.plot(range(5, 85, 5), arr_a_instance_micro, label='model_a')
    plt.plot(range(5, 85, 5), arr_b_instance_micro, label='model_b')
    plt.legend()
    plt.grid()
    plt.title('token_prf -instance_micro - f1')
    plt.savefig(
        f'{results_save_dir}/{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}_token_prf_instance_micro_f1.png')

    arr_a_instance_macro = []
    arr_a_instance_micro = []
    for k in range(5, 85, 5):
        path_json = f'{results_save_dir}/temp/metrics_results_{k}.json'
        f = open(path_json)
        data = json.load(f)
        arr_a_instance_macro.append(data['iou_scores'][0]['macro']['f1'])
        arr_a_instance_micro.append(data['iou_scores'][0]['micro']['f1'])
    arr_b_instance_macro = []
    arr_b_instance_micro = []
    for k in range(5, 85, 5):
        path_json = f'{results_save_dir}/our/metrics_results_{k}.json'
        f = open(path_json)
        data = json.load(f)
        arr_b_instance_macro.append(data['iou_scores'][0]['macro']['f1'])
        arr_b_instance_micro.append(data['iou_scores'][0]['micro']['f1'])
    plt.figure()
    plt.plot(range(5, 85, 5), arr_a_instance_macro, label='model_a')
    plt.plot(range(5, 85, 5), arr_b_instance_macro, label='model_b')
    plt.legend()
    plt.grid()
    plt.title('iou_scores -macro - f1')
    plt.savefig(
        f'{results_save_dir}/{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}_iou_scores_macro_f1.png')

    plt.figure()
    plt.plot(range(5, 85, 5), arr_a_instance_micro, label='model_a')
    plt.plot(range(5, 85, 5), arr_b_instance_micro, label='model_b')
    plt.legend()
    plt.grid()
    plt.title('iou_scores -macro - f1')
    plt.savefig(
        f'{results_save_dir}/{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}_iou_scores_micro_f1.png')
    arr_a_instance_macro = []
    arr_a_instance_micro = []
    for k in range(5, 85, 5):
        path_json = f'{results_save_dir}/temp/metrics_results_{k}.json'
        f = open(path_json)
        data = json.load(f)
        arr_a_instance_macro.append(data['rationale_prf']['instance_macro']['f1'])
        arr_a_instance_micro.append(data['rationale_prf']['instance_micro']['f1'])
    arr_b_instance_macro = []
    arr_b_instance_micro = []
    for k in range(5, 85, 5):
        path_json = f'{results_save_dir}/our/metrics_results_{k}.json'
        f = open(path_json)
        data = json.load(f)
        arr_b_instance_macro.append(data['rationale_prf']['instance_macro']['f1'])
        arr_b_instance_micro.append(data['rationale_prf']['instance_micro']['f1'])

    plt.figure()
    plt.plot(range(5, 85, 5), arr_a_instance_macro, label='model_a')
    plt.plot(range(5, 85, 5), arr_b_instance_macro, label='model_b')
    plt.legend()
    plt.grid()
    plt.title('rationale_prf -instance_macro - f1')
    plt.savefig(
        f'{results_save_dir}/{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}_rationale_prf_instance_macro_f1.png')

    plt.figure()
    plt.plot(range(5, 85, 5), arr_a_instance_micro, label='model_a')
    plt.plot(range(5, 85, 5), arr_b_instance_micro, label='model_b')
    plt.legend()
    plt.grid()
    plt.title('rationale_prf -instance_micro - f1')
    plt.savefig(
        f'{results_save_dir}/{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}_rationale_prf_instance_micro_f1.png')

    return


def wandb_plots_all_metrics(results_save_dir):
    arr_a_instance_macro = []
    arr_a_instance_micro = []
    for k in range(5, 85, 5):
        path_json = f'{results_save_dir}/temp/metrics_results_{k}.json'
        f = open(path_json)
        data = json.load(f)
        arr_a_instance_macro.append(data['token_prf']['instance_macro']['f1'])
        arr_a_instance_micro.append(data['token_prf']['instance_micro']['f1'])

    arr_b_instance_macro = []
    arr_b_instance_micro = []
    for k in range(5, 85, 5):
        path_json = f'{results_save_dir}/our/metrics_results_{k}.json'
        f = open(path_json)
        data = json.load(f)
        arr_b_instance_macro.append(data['token_prf']['instance_macro']['f1'])
        arr_b_instance_micro.append(data['token_prf']['instance_micro']['f1'])

    plt.figure()
    plt.plot(range(5, 85, 5), arr_a_instance_macro, label='model_a')
    plt.plot(range(5, 85, 5), arr_b_instance_macro, label='model_b')
    plt.legend()
    plt.grid()
    plt.title('token_prf -instance_macro - f1')
    plt.savefig(
        f'{results_save_dir}/{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}_token_prf_instance_macro_f1.png')

    plt.figure()
    plt.plot(range(5, 85, 5), arr_a_instance_micro, label='model_a')
    plt.plot(range(5, 85, 5), arr_b_instance_micro, label='model_b')
    plt.legend()
    plt.grid()
    plt.title('token_prf -instance_micro - f1')
    plt.savefig(
        f'{results_save_dir}/{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}_token_prf_instance_micro_f1.png')

    arr_a_instance_macro = []
    arr_a_instance_micro = []
    for k in range(5, 85, 5):
        path_json = f'{results_save_dir}/temp/metrics_results_{k}.json'
        f = open(path_json)
        data = json.load(f)
        arr_a_instance_macro.append(data['iou_scores'][0]['macro']['f1'])
        arr_a_instance_micro.append(data['iou_scores'][0]['micro']['f1'])
    arr_b_instance_macro = []
    arr_b_instance_micro = []
    for k in range(5, 85, 5):
        path_json = f'{results_save_dir}/our/metrics_results_{k}.json'
        f = open(path_json)
        data = json.load(f)
        arr_b_instance_macro.append(data['iou_scores'][0]['macro']['f1'])
        arr_b_instance_micro.append(data['iou_scores'][0]['micro']['f1'])
    plt.figure()
    plt.plot(range(5, 85, 5), arr_a_instance_macro, label='model_a')
    plt.plot(range(5, 85, 5), arr_b_instance_macro, label='model_b')
    plt.legend()
    plt.grid()
    plt.title('iou_scores -macro - f1')
    plt.savefig(
        f'{results_save_dir}/{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}_iou_scores_macro_f1.png')

    plt.figure()
    plt.plot(range(5, 85, 5), arr_a_instance_micro, label='model_a')
    plt.plot(range(5, 85, 5), arr_b_instance_micro, label='model_b')
    plt.legend()
    plt.grid()
    plt.title('iou_scores -macro - f1')
    plt.savefig(
        f'{results_save_dir}/{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}_iou_scores_micro_f1.png')
    arr_a_instance_macro = []
    arr_a_instance_micro = []
    for k in range(5, 85, 5):
        path_json = f'{results_save_dir}/temp/metrics_results_{k}.json'
        f = open(path_json)
        data = json.load(f)
        arr_a_instance_macro.append(data['rationale_prf']['instance_macro']['f1'])
        arr_a_instance_micro.append(data['rationale_prf']['instance_micro']['f1'])
    arr_b_instance_macro = []
    arr_b_instance_micro = []
    for k in range(5, 85, 5):
        path_json = f'{results_save_dir}/our/metrics_results_{k}.json'
        f = open(path_json)
        data = json.load(f)
        arr_b_instance_macro.append(data['rationale_prf']['instance_macro']['f1'])
        arr_b_instance_micro.append(data['rationale_prf']['instance_micro']['f1'])

    plt.figure()
    plt.plot(range(5, 85, 5), arr_a_instance_macro, label='model_a')
    plt.plot(range(5, 85, 5), arr_b_instance_macro, label='model_b')
    plt.legend()
    plt.grid()
    plt.title('rationale_prf -instance_macro - f1')
    plt.savefig(
        f'{results_save_dir}/{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}_rationale_prf_instance_macro_f1.png')

    plt.figure()
    plt.plot(range(5, 85, 5), arr_a_instance_micro, label='model_a')
    plt.plot(range(5, 85, 5), arr_b_instance_micro, label='model_b')
    plt.legend()
    plt.grid()
    plt.title('rationale_prf -instance_micro - f1')
    plt.savefig(
        f'{results_save_dir}/{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}_rationale_prf_instance_micro_f1.png')

    x1 = range(5, 85, 5)
    x2 = range(5, 85, 5)
    xs = [x1, x2]
    ys = [arr_a_instance_micro, arr_b_instance_micro]
    wandb.log({"rationale_prf_micro_f1": wandb.plot.line_series(xs, ys, keys=['model_a', 'model_b'],
                                                        title='rationale_prf -instance_micro - f1')})
    return


def write_results_to_json(doc_name, res, result_files, top_k_tokens_a):
    hard_rationales = []
    for index in top_k_tokens_a:
        hard_rationales.append({
            "start_token": index,
            "end_token": index + 1
        })
    result_dict = {
        "annotation_id": doc_name,
        "rationales": [{
            "docid": doc_name,
            "hard_rationale_predictions": hard_rationales
        }],
    }
    result_files[res].write(json.dumps(result_dict) + "\n")
    return


def plots_f1_acc_precision_recall(df_total_a, df_total_b, results_save_dir, test):
    f1_arr = []
    accuracy_arr = []
    precision_arr = []
    recall_arr = []
    f1_arr_b = []
    accuracy_arr_b = []
    precision_arr_b = []
    recall_arr_b = []
    k_arr = np.arange(5, 85, 5)
    for k in tqdm(range(len(k_arr)), position=0, leave=True):
        f1_arr.append(f1_score(df_total_a['label'], df_total_a[k]))
        f1_arr_b.append(f1_score(df_total_b['label'], df_total_b[k]))

        accuracy_arr.append(accuracy_score(df_total_a['label'], df_total_a[k]))
        accuracy_arr_b.append(accuracy_score(df_total_b['label'], df_total_b[k]))

        precision_arr.append(precision_score(df_total_a['label'], df_total_a[k]))
        precision_arr_b.append(precision_score(df_total_b['label'], df_total_b[k]))

        recall_arr.append(recall_score(df_total_a['label'], df_total_a[k]))
        recall_arr_b.append(recall_score(df_total_b['label'], df_total_b[k]))

    num_of_examples = len(test)
    plt.figure()
    plt.plot(k_arr, f1_arr, label='f1_score_model_a')
    plt.plot(k_arr, f1_arr_b, label='f1_score_model_b')
    plt.legend()
    plt.grid()
    plt.title('f1_score')
    plt.xlabel('k_tokens')
    plt.ylabel('score')
    plt.savefig(
        f'{results_save_dir}/{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}_f1_score_{round(f1_arr[-1] * 100)}_num_examples_{num_of_examples}.png')
    plt.figure()
    plt.plot(k_arr, accuracy_arr, label='accuracy_score_model_a')
    plt.plot(k_arr, accuracy_arr_b, label='accuracy_score_model_b')
    plt.legend()
    plt.grid()
    plt.title('accuracy_score')
    plt.xlabel('k_tokens')
    plt.ylabel('score')
    plt.savefig(
        f'{results_save_dir}/{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}_accuracy_score_{round(accuracy_arr[-1] * 100)}_num_examples_{num_of_examples}.png')
    plt.figure()
    plt.plot(k_arr, precision_arr, label='precision__model_a')
    plt.plot(k_arr, precision_arr_b, label='precision_model_b')
    plt.legend()
    plt.grid()
    plt.title('precision_score')
    plt.xlabel('k_tokens')
    plt.ylabel('score')
    plt.savefig(
        f'{results_save_dir}/{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}_precision_score_{round(precision_arr[-1] * 100)}_num_examples_{num_of_examples}.png')
    plt.figure()
    plt.plot(k_arr, recall_arr, label='recall_model_a')
    plt.plot(k_arr, recall_arr_b, label='recall_model_b')
    plt.legend()
    plt.grid()
    plt.title('recall_score')
    plt.xlabel('k_tokens')
    plt.ylabel('score')
    plt.savefig(
        f'{results_save_dir}/{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}_recall_score_{round(recall_arr[-1] * 100)}_num_examples_{num_of_examples}.png')


def bert_objective_temp_softmax(output: Tensor, target: Tensor, temp: Tensor, step: int,
                                contrastive_class_idx: Tensor = None) -> Tensor:
    # target_idx = contrastive_class_idx if contrastive_class_idx is not None else torch.argmax(target)
    prediction_loss = ce_loss(output, target) * loss_config['pred_loss_multiplier']
    entropy_loss = entropy(F.softmax(temp, dim=-1)) * loss_config['entropy_loss_multiplier']
    l1_loss = torch.mean(torch.abs(temp)) * loss_config['l1_loss_multiplier']
    loss = entropy_loss + prediction_loss + l1_loss
    if step % 100 == 0:
        print(f'l1_loss: {l1_loss}   entropy_loss: {entropy_loss}    prediction_loss: {prediction_loss}')
    return loss  # loss  # prediction_loss


def optuna_bert_objective_temp_softmax(output: Tensor, target: Tensor, temp: Tensor, step: int,
                                       pred_loss_multiplier: float, entropy_loss_multiplier: float) -> Tensor:
    prediction_loss = ce_loss(output, target) * pred_loss_multiplier
    entropy_loss = entropy(F.softmax(temp, dim=-1)) * entropy_loss_multiplier
    l1_loss = torch.mean(torch.abs(temp)) * 0
    loss = entropy_loss + prediction_loss + l1_loss
    if step % 100 == 0:
        print(f'l1_loss: {l1_loss}   entropy_loss: {entropy_loss}    prediction_loss: {prediction_loss}')
    return loss  # prediction_loss


def optimize_params(criterion: Callable):
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('cuda' if torch.cuda.is_available() else 'cpu')

    args = _get_args()
    cache, model_hila_save_file = _get_files_pathes(args)

    assert BATCH_FIRST

    train, val, test = load_datasets(args.data_dir)

    docids = load_doc_ids(test, train, val)
    documents = load_documents(args.data_dir, docids)

    logger.info(f'Loaded {len(documents)} documents')

    evidence_classifier, optimizer, scheduler, word_interner, de_interner, evidence_classes, tokenizer = \
        initialize_models(params=bert_config, model_type='softmax_temp', batch_first=BATCH_FIRST, device=device,
                          load_evidence_classifier=False)

    logger.info(f'We have {len(word_interner)} wordpieces')

    interned_documents = load_interning_documents(cache, documents, bert_config, tokenizer)

    # #
    # # logging.info(f'Beginning training classifier')
    # #
    # batch_size = bert_config['evidence_classifier']['batch_size']
    # epochs = bert_config['evidence_classifier']['epochs']
    # patience = bert_config['evidence_classifier']['patience']
    # max_grad_norm = bert_config['evidence_classifier'].get('max_grad_norm', None)
    #
    # results = {
    #     'train_loss': [],
    #     'train_f1': [],
    #     'train_acc': [],
    #     'val_loss': [],
    #     'val_f1': [],
    #     'val_acc': [],
    # }
    #
    # best_epoch = -1
    # best_val_acc = 0
    # best_val_loss = float('inf')
    # best_model_state_dict = None
    # start_epoch = 0
    # epoch_data = {}
    #
    # if os.path.exists(epoch_save_file):
    #     logging.info(f'Restoring model from {model_save_file}')
    #     evidence_classifier.load_state_dict(torch.load(model_save_file))
    #     epoch_data = torch.load(epoch_save_file)
    #     start_epoch = epoch_data['epoch'] + 1
    #     # handle finishing because patience was exceeded or we didn't get the best final epoch
    #     if bool(epoch_data.get('done', 0)):
    #         start_epoch = epochs
    #     results = epoch_data['results']
    #     best_epoch = start_epoch
    #     best_model_state_dict = OrderedDict({k: v.cpu() for k, v in evidence_classifier.state_dict().items()})
    #     logging.info(f'Restoring training from epoch {start_epoch}')
    #
    # start_epoch = epochs
    # logging.info(f'Training evidence classifier from epoch {start_epoch} until epoch {epochs}')
    # optimizer.zero_grad()
    # # criterion = nn.CrossEntropyLoss(reduction='none')
    # for epoch in tqdm(range(start_epoch, epochs)):
    #     epoch_train_data = random.sample(train, k=len(train))
    #     epoch_train_loss = 0
    #     epoch_training_acc = 0
    #
    #     evidence_classifier.train()
    #     logging.info(
    #         f'Training with {len(epoch_train_data) // batch_size} batches with {len(epoch_train_data)} examples')
    #
    #     for batch_start in tqdm(range(0, len(epoch_train_data), batch_size), position=0, leave=True):
    #         optimizer.zero_grad()
    #         batch_elements = epoch_train_data[batch_start:min(batch_start + batch_size, len(epoch_train_data))]
    #         targets = [evidence_classes[s.classification] for s in batch_elements]
    #         targets = torch.tensor(targets, dtype=torch.long, device=device)
    #         samples_encoding = [interned_documents[extract_docid_from_dataset_element(s)] for s in batch_elements]
    #         input_ids = torch.stack([samples_encoding[i]['input_ids'] for i in range(len(samples_encoding))]).squeeze(
    #             1).to(device)
    #         attention_masks = torch.stack(
    #             [samples_encoding[i]['attention_mask'] for i in range(len(samples_encoding))]).squeeze(1).to(device)
    #
    #         preds = evidence_classifier(input_ids=input_ids, attention_mask=attention_masks)
    #         epoch_training_acc += accuracy_score(preds.logits.argmax(dim=1).cpu(), targets.cpu(), normalize=False)
    #         # loss = criterion(preds.logits, targets).sum()
    #         loss = criterion(output=preds.logits, target=targets, temp=evidence_classifier.bert.encoder.x_attention)
    #         epoch_train_loss += loss.item()
    #         loss.backward()
    #         assert loss == loss  # for nans
    #         if max_grad_norm:
    #             torch.nn.utils.clip_grad_norm_(evidence_classifier.parameters(), max_grad_norm)
    #         optimizer.step()
    #         if scheduler:
    #             scheduler.step()
    #
    #     epoch_train_loss /= len(epoch_train_data)
    #     epoch_training_acc /= len(epoch_train_data)
    #     assert epoch_train_loss == epoch_train_loss  # for nans
    #     results['train_loss'].append(epoch_train_loss)
    #     logging.info(f'Epoch {epoch} training loss {epoch_train_loss}')
    #     logging.info(f'Epoch {epoch} training accuracy {epoch_training_acc}')
    #
    #     with torch.no_grad():
    #         epoch_val_loss = 0
    #         epoch_val_acc = 0
    #         epoch_val_data = random.sample(val, k=len(val))
    #         evidence_classifier.eval()
    #         val_batch_size = batch_size
    #         logging.info(
    #             f'Validating with {len(epoch_val_data) // val_batch_size} batches with {len(epoch_val_data)} examples')
    #         for batch_start in tqdm(range(0, len(epoch_val_data), val_batch_size), position=0, leave=True):
    #             batch_elements = epoch_val_data[batch_start:min(batch_start + val_batch_size, len(epoch_val_data))]
    #             targets = [evidence_classes[s.classification] for s in batch_elements]
    #             targets = torch.tensor(targets, dtype=torch.long, device=device)
    #             samples_encoding = [interned_documents[extract_docid_from_dataset_element(s)] for s in batch_elements]
    #             input_ids = torch.stack(
    #                 [samples_encoding[i]['input_ids'] for i in range(len(samples_encoding))]).squeeze(1).to(device)
    #             attention_masks = torch.stack(
    #                 [samples_encoding[i]['attention_mask'] for i in range(len(samples_encoding))]).squeeze(1).to(
    #                 device)
    #             preds = evidence_classifier(input_ids=input_ids, attention_mask=attention_masks)
    #             epoch_val_acc += accuracy_score(preds.logits.argmax(dim=1).cpu(), targets.cpu(), normalize=False)
    #             # loss = criterion(preds.logits, targets).sum()
    #             loss = criterion(output=preds.logits, target=targets, temp=evidence_classifier.bert.encoder.x_attention)
    #             epoch_val_loss += loss.item()
    #
    #         epoch_val_loss /= len(val)
    #         epoch_val_acc /= len(val)
    #         results["val_acc"].append(epoch_val_acc)
    #         results["val_loss"] = epoch_val_loss
    #
    #         logging.info(f'Epoch {epoch} val loss {epoch_val_loss}')
    #         logging.info(f'Epoch {epoch} val acc {epoch_val_acc}')
    #
    #         if epoch_val_acc > best_val_acc or (epoch_val_acc == best_val_acc and epoch_val_loss < best_val_loss):
    #             best_model_state_dict = OrderedDict({k: v.cpu() for k, v in evidence_classifier.state_dict().items()})
    #             best_epoch = epoch
    #             best_val_acc = epoch_val_acc
    #             best_val_loss = epoch_val_loss
    #             epoch_data = {
    #                 'epoch': epoch,
    #                 'results': results,
    #                 'best_val_acc': best_val_acc,
    #                 'done': 0,
    #             }
    #             torch.save(evidence_classifier.state_dict(), model_save_file)
    #             torch.save(epoch_data, epoch_save_file)
    #             logging.debug(f'Epoch {epoch} new best model with val accuracy {epoch_val_acc}')
    #     if epoch - best_epoch > patience:
    #         logging.info(f'Exiting after epoch {epoch} due to no improvement')
    #         epoch_data['done'] = 1
    #         torch.save(epoch_data, epoch_save_file)
    #         break
    #
    # epoch_data['done'] = 1
    # epoch_data['results'] = results
    # torch.save(epoch_data, epoch_save_file)
    # evidence_classifier.load_state_dict(best_model_state_dict)
    # evidence_classifier = evidence_classifier.to(device=device)
    # evidence_classifier.eval()

    test_classifier = BertForSequenceClassificationTest.from_pretrained(bert_config['model_name'],
                                                                        num_labels=len(evidence_classes)).to(device)

    test_classifier.load_state_dict(torch.load(model_hila_save_file))

    temp_classifier = Temp_BertForSequenceClassification.from_pretrained(bert_config['model_name'],
                                                                         num_labels=len(evidence_classes)).to(device)

    temp_classifier.bert.embeddings.load_state_dict(test_classifier.bert.embeddings.state_dict())
    temp_classifier.bert.encoder.layer.load_state_dict(test_classifier.bert.encoder.layer.state_dict())
    temp_classifier.bert.pooler.load_state_dict(test_classifier.bert.pooler.state_dict())
    temp_classifier.dropout.load_state_dict(test_classifier.dropout.state_dict())
    temp_classifier.classifier.load_state_dict(test_classifier.classifier.state_dict())

    # hila_model
    test_classifier.eval()
    temp_classifier.eval()
    test_batch_size = 1
    logging.info(
        f'Testing with {len(test) // test_batch_size} batches with {len(test)} examples')

    explanations = Generator(test_classifier)
    method = "transformer_attribution"
    method_expl = {"transformer_attribution": explanations.generate_LRP}

    results_save_dir = f'nlp/results/{date_exp_name}'
    results_epochs_save_dir = results_save_dir + '/epochs_loss'
    os.makedirs(results_save_dir, exist_ok=True)
    os.makedirs(results_epochs_save_dir, exist_ok=True)
    os.makedirs(f'{results_save_dir}/temp', exist_ok=True)
    os.makedirs(f'{results_save_dir}/our', exist_ok=True)

    method_folder = f'temp_results'
    method_folder_b = f'our_results'
    folder_a = os.path.join(results_save_dir, method_folder)
    folder_b = os.path.join(results_save_dir, method_folder_b)
    os.makedirs(os.path.join(folder_a), exist_ok=True)
    os.makedirs(folder_b, exist_ok=True)
    result_files = []
    result_files_b = []
    for i in range(5, 85, 5):
        result_files.append(
            open(os.path.join(folder_a, f'identifier_results_{i}.json'), 'w'))
        result_files_b.append(
            open(os.path.join(folder_b, f'identifier_results_{i}.json'), 'w'))

    test_batch_size = 1
    n = len(test)
    n = 2
    for batch_start in tqdm(range(0, n, test_batch_size)):

        batch_elements = test[batch_start:min(batch_start + test_batch_size, len(test))]
        samples_encoding = [interned_documents[extract_docid_from_dataset_element(s)] for s in batch_elements]

        targets = [evidence_classes[s.classification] for s in batch_elements]
        targets = torch.tensor(targets, dtype=torch.long, device=device)
        target_class_idx = targets.item()

        input_ids = torch.stack(
            [samples_encoding[i]['input_ids'] for i in range(len(samples_encoding))]).squeeze(1).to(device)

        attention_masks = torch.stack(
            [samples_encoding[i]['attention_mask'] for i in range(len(samples_encoding))]).squeeze(1).to(
            device)

        doc_name = extract_docid_from_dataset_element(batch_elements[0])
        inp = documents[doc_name].split()
        update_num_tokens_x_attenion(temp_classifier, input_ids, device=device)
        optimizer = optim.Adam([temp_classifier.bert.encoder.x_attention],
                               lr=bert_config['evidence_classifier']['lr'])
        epoch_temp_loss = []
        for step in range(100):  # bert_config['num_steps']):
            optimizer.zero_grad()
            preds = temp_classifier(input_ids, attention_masks)
            loss = criterion(output=preds.logits, target=targets, temp=temp_classifier.bert.encoder.x_attention, step=1)
            epoch_temp_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        fig = plt.figure()
        plt.plot(range(len(epoch_temp_loss)), epoch_temp_loss)
        plt.grid()
        plt.title(f"epoch_temp_loss__example_{batch_start}")
        plt.savefig(
            f'{results_epochs_save_dir}/epoch_temp_loss__example_{batch_start}.png')
        plt.close()
        wandb.log({f'epoch_temp_loss__example_{batch_start}.png': fig})

        data = [[x, y] for (x, y) in zip(range(len(epoch_temp_loss)), epoch_temp_loss)]
        table = wandb.Table(data=data, columns=["epochs", "loss"])
        wandb.log({f"epoch_temp_loss__example_{batch_start}": wandb.plot.line(table, "epochs", "loss",
                                                                              title=f"epoch_temp_loss__example_{batch_start}")})

        cls_attentions_probs = get_attention_probs_by_layer_of_the_CLS(model=temp_classifier)
        sorted_k_tokens = generate_scoring_by_head_agg(cls_attentions_probs, input_ids[0], top_k=2, agg='mean',
                                                       tokenizer=tokenizer,
                                                       print_tokens=False)
        cam_target = \
            method_expl[method](input_ids=input_ids, attention_mask=attention_masks, device=device,
                                index=target_class_idx)[0]
        cam_target = cam_target.clamp(min=0)
        cam = cam_target
        cam = scores_per_word_from_scores_per_token(inp, tokenizer, input_ids[0], cam)

        for res, k in enumerate(range(5, 85, 5)):
            # print("calculating top ", k)
            top_k_tokens_a = sorted_k_tokens[:k]
            _, top_k_tokens_b = cam.topk(k=k)
            write_results_to_json(doc_name, res, result_files, top_k_tokens_a)
            write_results_to_json(doc_name, res, result_files_b, top_k_tokens_b.tolist())

    close_files(result_files, result_files_b)

    cal_metrics(args.data_dir, split='test', results_path=f'{folder_a}/identifier_results_k.json',
                score_file=f'{results_save_dir}/temp/metrics_results_k.json')
    cal_metrics(args.data_dir, split='test', results_path=f'{folder_b}/identifier_results_k.json',
                score_file=f'{results_save_dir}/our/metrics_results_k.json')

    # plots_all_metrics(results_save_dir)
    wandb_plots_all_metrics(results_save_dir)
    print('FINISH')

    return


def objective(trial, documents, evidence_classes, interned_documents, model_hila_save_file, scheduler, test, tokenizer):
    criterion = optuna_bert_objective_temp_softmax

    entropy_loss_multiplier = trial.suggest_categorical('entropy_loss_multiplier',
                                                        [1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10, 100])
    pred_loss_multiplier = trial.suggest_categorical('pred_loss_multiplier',
                                                     [1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10, 100])

    gc.collect()

    torch.cuda.empty_cache()

    test_classifier = BertForSequenceClassificationTest.from_pretrained(bert_config['model_name'],
                                                                        num_labels=len(evidence_classes)).to(device)

    test_classifier.load_state_dict(torch.load(model_hila_save_file))

    temp_classifier = Temp_BertForSequenceClassification.from_pretrained(bert_config['model_name'],
                                                                         num_labels=len(evidence_classes)).to(device)

    temp_classifier.bert.embeddings.load_state_dict(test_classifier.bert.embeddings.state_dict())
    temp_classifier.bert.encoder.layer.load_state_dict(test_classifier.bert.encoder.layer.state_dict())
    temp_classifier.bert.pooler.load_state_dict(test_classifier.bert.pooler.state_dict())
    temp_classifier.dropout.load_state_dict(test_classifier.dropout.state_dict())
    temp_classifier.classifier.load_state_dict(test_classifier.classifier.state_dict())

    test_classifier.eval()
    temp_classifier.eval()

    test_batch_size = 1
    n = len(test)
    logging.info(f'Testing with {n // test_batch_size} batches with {n} examples')
    result_dict_list = []
    for batch_start in tqdm(range(0, n, test_batch_size)):

        batch_elements = test[batch_start:min(batch_start + test_batch_size, len(test))]
        samples_encoding = [interned_documents[extract_docid_from_dataset_element(s)] for s in batch_elements]

        targets = [evidence_classes[s.classification] for s in batch_elements]
        targets = torch.tensor(targets, dtype=torch.long, device=device)
        target_class_idx = targets.item()

        input_ids = torch.stack(
            [samples_encoding[i]['input_ids'] for i in range(len(samples_encoding))]).squeeze(1).to(device)

        attention_masks = torch.stack(
            [samples_encoding[i]['attention_mask'] for i in range(len(samples_encoding))]).squeeze(1).to(
            device)

        doc_name = extract_docid_from_dataset_element(batch_elements[0])
        inp = documents[doc_name].split()
        update_num_tokens_x_attenion(temp_classifier, input_ids, device=device)
        optimizer = optim.Adam([temp_classifier.bert.encoder.x_attention],
                               lr=bert_config['evidence_classifier']['lr'])
        epoch_temp_loss = []
        for step in range(bert_config['num_steps']):
            optimizer.zero_grad()
            preds = temp_classifier(input_ids, attention_masks)
            loss = criterion(output=preds.logits, target=targets, temp=temp_classifier.bert.encoder.x_attention, step=1,
                             pred_loss_multiplier=pred_loss_multiplier,
                             entropy_loss_multiplier=entropy_loss_multiplier)
            epoch_temp_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        cls_attentions_probs = get_attention_probs_by_layer_of_the_CLS(model=temp_classifier)
        sorted_k_tokens = generate_scoring_by_head_agg(cls_attentions_probs, input_ids[0], top_k=2, agg='mean',
                                                       tokenizer=tokenizer,
                                                       print_tokens=False)

        k = 80
        top_k_tokens_a = sorted_k_tokens[:k]
        hard_rationales = []

        for index in top_k_tokens_a:
            hard_rationales.append({
                "start_token": index,
                "end_token": index + 1
            })
        result_dict = {
            "annotation_id": doc_name,
            "rationales": [{
                "docid": doc_name,
                "hard_rationale_predictions": hard_rationales
            }],
        }
        result_dict_list.append(result_dict)

    truth = list(chain.from_iterable(Rationale.from_annotation(ann) for ann in test))
    pred = list(chain.from_iterable(Rationale.from_instance(inst) for inst in result_dict_list))
    token_level_truth = list(chain.from_iterable(rat.to_token_level() for rat in truth))
    token_level_pred = list(chain.from_iterable(rat.to_token_level() for rat in pred))
    truth, pred = set(token_level_truth), set(token_level_pred)
    f1_macro = token_prf_f1_macro(truth=truth, pred=pred)

    return f1_macro


def preprocess_params(device):
    args = _get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    save_dir = args.output_dir
    evidence_classifier_output_dir = os.path.join(save_dir, 'classifier')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(evidence_classifier_output_dir, exist_ok=True)
    cache = os.path.join(args.output_dir, 'preprocessed.pkl')
    model_save_file = os.path.join(evidence_classifier_output_dir, 'classifier.pt')
    model_hila_save_file = os.path.join(evidence_classifier_output_dir, 'classifier.pt')
    epoch_save_file = os.path.join(evidence_classifier_output_dir, 'classifier_epoch_data.pt')
    assert BATCH_FIRST
    train, val, test = load_datasets(args.data_dir)
    docids = load_doc_ids(test, train, val)
    documents = load_documents(args.data_dir, docids)
    logger.info(f'Loaded {len(documents)} documents')
    evidence_classifier, optimizer, scheduler, word_interner, de_interner, evidence_classes, tokenizer = \
        initialize_models(params=bert_config, model_type='softmax_temp', batch_first=BATCH_FIRST, device=device,
                          load_evidence_classifier=False)
    logger.info(f'We have {len(word_interner)} wordpieces')
    interned_documents = load_interning_documents(cache, documents, bert_config, tokenizer)
    return documents, evidence_classes, interned_documents, model_hila_save_file, scheduler, test, tokenizer


def run_optuna(n_trials=30):
    global device, documents, evidence_classes, interned_documents, model_hila_save_file, scheduler, test, tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('cuda' if torch.cuda.is_available() else 'cpu')
    documents, evidence_classes, interned_documents, model_hila_save_file, scheduler, test, tokenizer = preprocess_params(
        device)
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, documents, evidence_classes, interned_documents, model_hila_save_file, scheduler,
                                test, tokenizer), n_trials=n_trials)
    # study.optimize(objective, n_trials=40)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
#
# def foo():
#     x1 = range(5, 85, 5)
#     x2 = range(5, 85, 5)
#     xs = [x1]
#     ys = [x1]
#     data = [[x, y] for (x, y) in zip(x1, x1)]
#     table = wandb.Table(data=data,columns=['steps','loss'])
#     wandb.log({"rationale_prf_micro_f1": wandb.plot.line(table, x='steps',y='loss',
#                                                                 title='rationale_prf -instance_micro - f1')})
#     table = wandb.Table(data=data, columns=['steps', 'loss2'])
#     wandb.log({"rationale_prf_micro_f1_temp2": wandb.plot.line(table, x='steps', y='loss2',
#                                                          title='rationale_prf -instance_micro - f1')})

if __name__ == '__main__':
    # run_optuna(n_trials = 30)
    # foo()
    # experiment_name = 'bert'
    # print(experiment_name)
    # optimize_params(criterion=bert_objective_temp_softmax)

    print('tes')
