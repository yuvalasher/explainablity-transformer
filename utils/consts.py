from pathlib import Path
from config import config
import torch

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

ROOT_DIR: Path = Path(__file__).parent.parent
DATA_PATH: Path = Path(ROOT_DIR, 'data')
PLOTS_PATH: Path = Path(ROOT_DIR, 'research', 'plots')
IMAGES_FOLDER_PATH = Path(DATA_PATH, config['general']['images_folder_name']) if not cuda else Path(DATA_PATH,
                                                                                                    config['general'][
                                                                                                        'images_folder_name_DGX'])
IMAGES_LABELS_GT_PATH = Path(DATA_PATH, config['general']['images_gt_filename'])
PICKLES_FOLDER_PATH: Path = Path(ROOT_DIR, 'pickles').resolve()
EXPERIMENTS_FOLDER_PATH: Path = Path(ROOT_DIR, 'research', 'experiments').resolve()
