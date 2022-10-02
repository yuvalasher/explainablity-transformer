from icecream import ic
from pathlib import Path
import pickle

BEST_AUC_VALUE = 6
BASE_MODEL_PKL_PATH = "/home/yuvalas/explainability/research/experiments/seg_cls/ft_3000/base_model/opt_objects"
OPTIMIZATION_PKL_PATH = "/home/yuvalas/explainability/research/experiments/seg_cls/ft_3000/opt_objects"


def load_obj(path: str):
    with open(Path(path), 'rb') as f:
        return pickle.load(f)


s_to_5 = 0
s_smaller_opt_than_baseline = 0
s_wrong = 0
n_samples = 3000
for image_idx in range(n_samples):
    image_base_model_path = Path(BASE_MODEL_PKL_PATH, f"{str(image_idx)}.pkl")
    image_optimization_path = Path(OPTIMIZATION_PKL_PATH, f"{str(image_idx)}.pkl")

    base_model_loaded_obj = load_obj(image_base_model_path)
    optimization_loaded_obj = load_obj(image_optimization_path)

    # ic(base_model_loaded_obj['auc'], optimization_loaded_obj['auc'])
    if optimization_loaded_obj['auc'] > base_model_loaded_obj['auc']:
        s_wrong += 1
    if optimization_loaded_obj['auc'] < BEST_AUC_VALUE and base_model_loaded_obj['auc'] > BEST_AUC_VALUE:
        s_to_5 += 1
    if optimization_loaded_obj['auc'] < base_model_loaded_obj['auc']:
        s_smaller_opt_than_baseline += 1

ic(s_to_5)
ic(100 * (s_to_5 / n_samples))

ic(s_smaller_opt_than_baseline)
ic(100 * (s_smaller_opt_than_baseline / n_samples))

ic(s_wrong)
