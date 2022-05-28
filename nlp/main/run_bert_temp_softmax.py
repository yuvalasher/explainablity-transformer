import argparse
import logging
from itertools import chain

from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, BertTokenizer

from nlp.datagetters.datagetters import load_datasets, load_documents
from nlp.utils_class.ExplanationGenerator import Generator
from loss_utils import entropy
from tqdm import tqdm
from datetime import datetime

from utils.consts import *
from pytorch_lightning import seed_everything
from typing import Callable
from nlp.bert_utils import *
from utils import *
import matplotlib.pyplot as plt
import pandas as pd

# os.chdir('C:/Users/amitx/Documents/Amit/Projects_Python/explainablity-transformer')
os.chdir('/home/amiteshel1/Projects/explainablity-transformer')
bert_config = config['bert']
loss_config = bert_config['loss']
ROOT_DIR = os.environ['ROOT_DIR']
seed_everything(config['general']['seed'])
bert_model = handle_model_config_and_freezing_for_task(model=load_BertModel(bert_config, model_type='infer'))
bert_model.eval()
BATCH_FIRST = True

# model = BertForSequenceClassification.from_pretrained(bert_config['model_name'])
# model_name = "textattack/bert-base-uncased-SST-2"
tokenizer = AutoTokenizer.from_pretrained(bert_config['model_name'])

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
logger = logging.getLogger(__name__)


# classifications = ["NEGATIVE", "POSITIVE"]

def bert_objective_temp_softmax(output: Tensor, target: Tensor, temp: Tensor,
                                contrastive_class_idx: Tensor = None) -> Tensor:
    target_idx = contrastive_class_idx if contrastive_class_idx is not None else torch.argmax(target)
    prediction_loss = ce_loss(output, target_idx.unsqueeze(0)) * loss_config['pred_loss_multiplier']
    entropy_loss = entropy(F.softmax(temp, dim=-1)) * loss_config['entropy_loss_multiplier']
    l1_loss = torch.mean(torch.abs(temp)) * loss_config['l1_loss_multiplier']
    # print(f'entropy: {entropy_loss}, prediction_loss: {prediction_loss}, l1_loss: {l1_loss}')
    loss = entropy_loss + prediction_loss + l1_loss
    # print(f'loss: {loss}, max_temp: {torch.max(temp)}, min_temp: {torch.min(temp)}')
    # log(loss=loss, entropy_loss=entropy_loss, prediction_loss=prediction_loss, x_attention=temp, output=output,
    #     target=target, contrastive_class_idx=target_idx.item())
    return loss

def update_num_tokens_x_attenion(bert_ours_model, input_ids):
    input_num_tokens = input_ids.shape[1]
    bert_ours_model.bert.encoder.num_tokens = input_num_tokens
    bert_ours_model.bert.encoder.x_attention = nn.Parameter(
        torch.ones(bert_ours_model.config.num_hidden_layers, bert_ours_model.config.num_attention_heads,
                   input_num_tokens, requires_grad=True))
    return

def generate_scoring_by_head_agg(cls_attentions_probs, input_ids, top_k: int = 10, agg: str = 'median',
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

def initialize_models(params: dict, batch_first: bool, use_half_precision=False):
    assert batch_first
    max_length = params['max_length']
    tokenizer = BertTokenizer.from_pretrained(params['model_name'])
    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    bert_dir = params['model_name']
    evidence_classes = dict((y, x) for (x, y) in enumerate(params['evidence_classifier']['classes']))
    evidence_classifier = BertForSequenceClassification.from_pretrained(bert_dir, num_labels=len(evidence_classes))
    word_interner = tokenizer.vocab
    de_interner = tokenizer.ids_to_tokens
    return evidence_classifier, word_interner, de_interner, evidence_classes, tokenizer

def load_data():
    # train, val, test = load_datasets('C:/Users/amitx/Documents/Amit/Projects_Python/explainablity-transformer/bert_models/movies/movies')
    train = pd.read_csv(ROOT_DIR + 'datasets/imdb/train.csv').sample(frac=1)
    test = pd.read_csv(ROOT_DIR + 'datasets/imdb/test.csv').sample(frac=1)

    return train, test



def optimize_params(bert_model, criterion: Callable):
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
    os.makedirs(args.output_dir, exist_ok=True)
    assert BATCH_FIRST
    train, val, test = load_datasets(args.data_dir)

    docids = set(e.docid for e in
                 chain.from_iterable(chain.from_iterable(map(lambda ann: ann.evidences, chain(train, val, test)))))
    documents = load_documents(args.data_dir, docids)
    logger.info(f'Loaded {len(documents)} documents')
    evidence_classifier, word_interner, de_interner, evidence_classes, tokenizer = \
        initialize_models(params=bert_config, batch_first=BATCH_FIRST)
    logger.info(f'We have {len(word_interner)} wordpieces')
    cache = os.path.join(args.output_dir, 'preprocessed.pkl')
    if os.path.exists(cache):
        logger.info(f'Loading interned documents from {cache}')
        (interned_documents) = torch.load(cache)
    else:
        logger.info(f'Interning documents')
        interned_documents = {}
        for d, doc in documents.items():
            encoding = tokenizer.encode_plus(
                doc,
                add_special_tokens=True,
                max_length=bert_config['max_length'],
                return_token_type_ids=False,
                pad_to_max_length=False,
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True,
            )
            interned_documents[d] = encoding
        torch.save((interned_documents), cache)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('cuda' if torch.cuda.is_available() else 'cpu')
    evidence_classifier = evidence_classifier.to(device)
    optimizer = None
    scheduler = None

    save_dir = args.output_dir

    logging.info(f'Beginning training classifier')
    evidence_classifier_output_dir = os.path.join(save_dir, 'classifier')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(evidence_classifier_output_dir, exist_ok=True)
    model_save_file = os.path.join(evidence_classifier_output_dir, 'classifier.pt')
    epoch_save_file = os.path.join(evidence_classifier_output_dir, 'classifier_epoch_data.pt')

    device = next(evidence_classifier.parameters()).device
    if optimizer is None:
        optimizer = torch.optim.Adam(evidence_classifier.parameters(), lr=bert_config['evidence_classifier']['lr'])
    # criterion = nn.CrossEntropyLoss(reduction='none')
    batch_size = bert_config['evidence_classifier']['batch_size']
    epochs = bert_config['evidence_classifier']['epochs']
    patience = bert_config['evidence_classifier']['patience']
    max_grad_norm = bert_config['evidence_classifier'].get('max_grad_norm', None)

    class_labels = [k for k, v in sorted(evidence_classes.items())]

    results = {
        'train_loss': [],
        'train_f1': [],
        'train_acc': [],
        'val_loss': [],
        'val_f1': [],
        'val_acc': [],
    }
    best_epoch = -1
    best_val_acc = 0
    best_val_loss = float('inf')
    best_model_state_dict = None
    start_epoch = 0
    epoch_data = {}

    # train, test = load_data()
    df_total = pd.DataFrame([])
    num_of_examples = bert_config['num_of_examples']
    data = test

    arr = [interned_documents[next(iter(s.evidences))[0].docid] for s in data]

    for idx in tqdm(range(min(num_of_examples, len(data))), position=0, leave=True):
        print(f'Start the {idx} example - text')

        # text, correct_class_idx, contrastive_class_idx = get_text_spec(data, idx)
        # encoding, input_ids, attention_mask = get_input_tokens(text=text, tokenizer=tokenizer)

        targets = [evidence_classes[data[idx].classification]]
        targets = torch.tensor(targets, dtype=torch.long, device=device)
        correct_class_idx = "neg" if targets.item() == 0 else "pos"
        input_ids, attention_mask = arr[idx]['input_ids'], arr[idx]['attention_mask']

        bert_ours_model, optimizer = setup_model_and_optimizer(model_type='infer')

        print_number_of_trainable_and_not_trainable_params(model=bert_ours_model)

        update_num_tokens_x_attenion(bert_ours_model, input_ids)

        target = bert_model(input_ids, attention_mask)
        target_class_idx = torch.argmax(target.logits[0])

        total_losses, prediction_losses, correct_class_logits, correct_class_probs, tokens_mask, temps = [], [], [], [], [], []

        for iteration_idx in tqdm(range(bert_config['num_steps']), position=0, leave=True):
            optimizer.zero_grad()

            output = bert_ours_model(input_ids, attention_mask)

            correct_class_logit, correct_class_prob, prediction_loss = get_iteration_target_class_stats(
                output=output, target_class_idx=target_class_idx)

            temps.append(bert_ours_model.bert.encoder.x_attention.clone())

            loss = criterion(output=output.logits, target=target.logits,
                             temp=bert_ours_model.bert.encoder.x_attention)

            loss.backward()

            # compare_results_each_n_steps(iteration_idx=iteration_idx, target=target.logits, output=output.logits,
            #                              prev_x_attention=bert_ours_model.bert.encoder.x_attention,
            #                              sampled_binary_patches=None)

            cls_attentions_probs = get_attention_probs_by_layer_of_the_CLS(model=bert_model)
            correct_class_logits.append(correct_class_logit)
            correct_class_probs.append(correct_class_prob)
            prediction_losses.append(prediction_loss)
            total_losses.append(loss.item())
            tokens_mask.append(cls_attentions_probs.clone())
            optimizer.step()

        # checkpoint = {
        #     'epoch': epoch,
        #     'model': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'lr_sched': lr_sched}
        # torch.save(checkpoint, 'checkpoint.pth')

        # print(text)
        # print("Start to cal")

        sorted_k_tokens = generate_scoring_by_head_agg(cls_attentions_probs, input_ids[0], top_k=100, agg='mean',
                                                       print_tokens=True)
        pred_class_arr = []
        k_arr = []
        step = bert_config['k_steps']
        total_k_tokens = bert_config['k_tokens']

        # # Hila Explainabilty
        inp = documents[data[idx].annotation_id].split()
        explanations = Generator(bert_ours_model)
        method = "transformer_attribution"
        method_expl = {"transformer_attribution": explanations.generate_LRP}
        # "partial_lrp": explanations_orig_lrp.generate_LRP_last_layer,
        # "last_attn": explanations_orig_lrp.generate_attn_last_layer,
        # "attn_gradcam": explanations_orig_lrp.generate_attn_gradcam,
        # "lrp": explanations_orig_lrp.generate_full_lrp,
        # "rollout": explanations_orig_lrp.generate_rollout}
        cam_target = method_expl[method](input_ids=input_ids, attention_mask=attention_mask, index=target_class_idx)[0]
        cam_target = cam_target.clamp(min=0)
        cam = cam_target
        cam = scores_per_word_from_scores_per_token(inp, tokenizer, input_ids[0], cam)

        for k in tqdm(np.arange(total_k_tokens + step, step=step), position=0, leave=True):
            top_k_tokens = sorted_k_tokens[:k]
            filter_out_indices = list(set(range(len(input_ids[0]))) - set(top_k_tokens))
            input_ids_temp = input_ids.clone()
            attention_mask_temp = attention_mask.clone()
            input_ids_temp[0, filter_out_indices] = 0
            attention_mask_temp[0, filter_out_indices] = 0
            output = bert_ours_model(input_ids_temp, attention_mask_temp)
            pred_class = torch.argmax(output[0]).item()
            # correct_class_logit, correct_class_k, prediction_loss = get_iteration_target_class_stats(
            #     output=output, target_class_idx=target_class_idx)
            pred_class_arr.append(pred_class)
            k_arr.append(k)
        # plt.plot(k_arr, pred_class_arr)
        df = pd.DataFrame([pred_class_arr])
        df['label'] = targets.item()
        df_total = pd.concat((df_total, df))

    f1_arr = []
    precision_arr = []
    recall_arr = []
    for k in tqdm(range(len(k_arr)), position=0, leave=True):
        f1_arr.append(f1_score(df_total['label'], df_total[k]))
        precision_arr.append(precision_score(df_total['label'], df_total[k]))
        recall_arr.append(recall_score(df_total['label'], df_total[k]))

    plt.plot(k_arr, f1_arr, label='f1_score')
    plt.plot(k_arr, precision_arr, label='precision')
    plt.plot(k_arr, recall_arr, label='recall')
    plt.legend()
    plt.grid()
    plt.savefig(
        f'nlp/results/{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}_num_examples_{num_of_examples}_f1_score_{round(f1_arr[-1] * 100)}.png')
    # plt.show()
    print('FINISH')


if __name__ == '__main__':
    experiment_name = 'bert'
    print(experiment_name)
    optimize_params(bert_model=bert_model, criterion=bert_objective_temp_softmax)
