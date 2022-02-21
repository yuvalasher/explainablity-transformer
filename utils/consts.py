from pathlib import Path
from config import config

DATA_PATH: Path = Path(Path.cwd(), '..', 'data')
# PLOTS_PATH: Path = Path(Path.cwd(), '../research', 'plots')
PLOTS_PATH: Path = Path(Path.cwd(), '../research', 'plots')
IMAGES_FOLDER_PATH = Path(DATA_PATH, config['general']['images_folder_name'])
IMAGES_LABELS_GT_PATH = Path(DATA_PATH, config['general']['images_gt_filename'])
PICKLES_FOLDER_PATH: Path = Path(Path.cwd(), '../pickles').resolve()
