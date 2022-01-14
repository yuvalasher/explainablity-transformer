import json
import pandas as pd
from typing import Dict

from PIL import Image
import os
from pathlib import Path
from typing import List, Union, NewType

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