from pathlib import Path
from config import config
import torch

cuda = torch.cuda.is_available()

ROOT_DIR: Path = Path(__file__).parent.parent
DATA_PATH: Path = Path(ROOT_DIR, 'data')

IMAGENET_VAL_IMAGES_FOLDER_PATH = ""  # TODO - need to download ImageNet validation 50K images data. Can be downloaded from https://image-net.org/challenges/LSVRC/2012/index.php
PLOTS_PATH: Path = Path(ROOT_DIR, 'research', 'plots')
IMAGES_FOLDER_PATH = IMAGENET_VAL_IMAGES_FOLDER_PATH
PICKLES_FOLDER_PATH: Path = Path(ROOT_DIR, 'pickles').resolve()
EXPERIMENTS_FOLDER_PATH: Path = Path(ROOT_DIR, 'research', 'experiments').resolve()

RESULTS_PICKLES_FOLDER_PATH = EXPERIMENTS_FOLDER_PATH
EVALUATION_FOLDER_PATH: Path = Path(ROOT_DIR, 'evaluation').resolve()

GT_VALIDATION_PATH_LABELS = Path(ROOT_DIR, "gt_data_imagenet", "val ground truth 2012.txt")
IMAGENET_VAL_GT_CSV_FILE_PATH = Path(ROOT_DIR, "gt_data_imagenet", "val_ground_truth_2012.csv")
IMAGENET_SEG_PATH = ""  # TODO - need to download gtsegs_ijcv.mat from the internet and insert the path. Can be downloaded from "http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat"
IMAGES_LABELS_GT_PATH = Path(DATA_PATH, config['general']['images_gt_filename'])
PLTE_CHECKPOINTS_PATH = Path(ROOT_DIR, 'checkpoints').resolve()
