from pathlib import Path
from config import config
import torch

cuda = torch.cuda.is_available()

ROOT_DIR: Path = Path(__file__).parent.parent
DATA_PATH: Path = Path(ROOT_DIR, 'data')

IMAGENET_VAL_IMAGES_FOLDER_PATH = "/home/amiteshel1/Projects/explainablity-transformer/vit_data/"
IMAGENET_SEG_PATH = '/home/amiteshel1/Projects/explainablity-transformer-cv/datasets/gtsegs_ijcv.mat'

PLOTS_PATH: Path = Path(ROOT_DIR, 'research', 'plots')
IMAGES_FOLDER_PATH = IMAGENET_VAL_IMAGES_FOLDER_PATH
PICKLES_FOLDER_PATH: Path = Path(ROOT_DIR, 'pickles').resolve()
EXPERIMENTS_FOLDER_PATH: Path = Path(ROOT_DIR, 'research', 'experiments').resolve()
RESULTS_PICKLES_FOLDER_PATH = EXPERIMENTS_FOLDER_PATH
EVALUATION_FOLDER_PATH: Path = Path(ROOT_DIR, 'evaluation').resolve()
GT_VALIDATION_PATH_LABELS = Path(ROOT_DIR, "gt_data_imagenet", "val ground truth.txt")
IMAGENET_VAL_GT_CSV_FILE_PATH = Path(ROOT_DIR, "gt_data_imagenet", "val_ground_truth_2012.csv")
# IMAGES_LABELS_GT_PATH = Path(DATA_PATH, "ILSVRC2012_validation_ground_truth.txt")
PLTE_CHECKPOINTS_PATH = Path(ROOT_DIR, 'checkpoints').resolve()