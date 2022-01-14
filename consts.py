from pathlib import Path
from config import config

DATA_PATH: Path = Path(Path.cwd(), 'data')
images_folder_path = Path(DATA_PATH, config['general']['images_folder_name'])
images_labels_gt_path = Path(DATA_PATH, config['general']['images_gt_filename'])
