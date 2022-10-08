from pathlib import Path
from config import config
import torch

# DGX_IMAGENET_ALL_VALIDATION_PATH = "/home/yuvalas/explainability_transformers/data/ILSVRC2012_val_Updated_DGX"
DGX_IMAGENET_ALL_VALIDATION_PATH = "/home/amiteshel1/Projects/explainablity-transformer/vit_data/"
cuda = torch.cuda.is_available()
# device = torch.device("cuda" if cuda else "cpu")

ROOT_DIR: Path = Path(__file__).parent.parent
DATA_PATH: Path = Path(ROOT_DIR, 'data')

# IMAGENET_TEST_IMAGES_FOLDER_PATH = Path(DATA_PATH, config["general"]["images_folder_test_DGX"])
# IMAGENET_TEST_IMAGES_ES_FOLDER_PATH = Path(DATA_PATH, config['general']['images_folder_test_ES_DGX'])
IMAGENET_TEST_IMAGES_ES_FOLDER_PATH = "/home/yuvalas/explainability/data/ILSVRC2012_test_earlystopping"
IMAGENET_TEST_IMAGES_FOLDER_PATH = "/home/yuvalas/explainability/data/ILSVRC2012_test_sampled"
IMAGENET_VAL_IMAGES_FOLDER_PATH = DGX_IMAGENET_ALL_VALIDATION_PATH

PLOTS_PATH: Path = Path(ROOT_DIR, 'research', 'plots')
ORIGINAL_IMAGES_FOLDER_PATH = Path(DATA_PATH, config['general']['images_folder_name'])
DGX_IMAGES_FOLDER_PATH = DGX_IMAGENET_ALL_VALIDATION_PATH
# TRAIN_IMAGES_FOLDER_PATH = Path(DATA_PATH, config['vit']['train_images_path'])
IMAGES_FOLDER_PATH = ORIGINAL_IMAGES_FOLDER_PATH if not cuda else DGX_IMAGES_FOLDER_PATH
IMAGES_LABELS_GT_PATH = Path(DATA_PATH, config['general']['images_gt_filename'])
PICKLES_FOLDER_PATH: Path = Path(ROOT_DIR, 'pickles').resolve()
EXPERIMENTS_FOLDER_PATH: Path = Path(ROOT_DIR, 'research', 'experiments').resolve()
EVALUATION_FOLDER_PATH: Path = Path(ROOT_DIR, 'evaluation').resolve()
GT_VALIDATION_PATH_LABELS = "/home/yuvalas/explainability/data/val ground truth 2012.txt"
