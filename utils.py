import pickle
import json
import pandas as pd

from PIL import Image
import os
from pathlib import Path, WindowsPath
from typing import Any, Dict, List, Union
from consts import PICKLES_FOLDER_PATH

def create_df_of_img_name_with_label(path: Path) -> pd.DataFrame:
    dirlist = os.listdir(path)
    dataframes = []
    for dir in dirlist:
        print('dir', dir)
        if os.path.isdir(Path(path, dir)):
            images_path = Path(path, dir, 'images')
            files_names = os.listdir(images_path)
            df = pd.DataFrame(files_names, columns=['file_name'])
            df['label'] = dir
            dataframes.append(df)
    return pd.concat(dataframes)


def _get_class2_idx(path: Path) -> Dict:
    with open(path, 'r') as f:
        class2idx = json.load(f)
    return class2idx


def read_csv(base_path: Path, file_name: str) -> pd.DataFrame:
    class2idx = _get_class2_idx(path=Path(base_path, 'class2idx.json'))
    df = pd.read_csv(Path(base_path, f'{file_name}.csv'), index_col=0)
    df['label'] = df.label.map(lambda x: class2idx[x])
    return df


def read_gt_labels(path: Path) -> List[str]:
    with open(path, 'r') as f:
        return f.readlines()


def parse_gt_labels(labels: List[str]) -> List[int]:
    return [int(label.replace('\n', '')) for label in labels]


def get_image_from_path(path: str) -> Image:
    return Image.open(path)


def save_obj_to_disk(path: Union[WindowsPath, str], obj) -> None:
    print(path)
    if type(path) == str and path[-4:] != '.pkl':
        path += '.pkl'
    elif type(path) == WindowsPath and path.suffix != '.pkl':
        path = path.with_suffix('.pkl')

    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(obj_name: str) -> Any:
    with open(Path(f"{PICKLES_FOLDER_PATH}", f"{obj_name}.pkl"), 'rb') as f:
        return pickle.load(f)
