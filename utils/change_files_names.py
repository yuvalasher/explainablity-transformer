import os
import shutil
from pathlib import Path


def _remove_file_if_exists(path: Path) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


src_images_folder = '/home/yuvalas/wolf/Transformer-Explainability/baselines/data'
dst_images_folder = '/home/yuvalas/wolf/Transformer-Explainability/baselines/data_new'

if __name__ == '__main__':
    pictures_to_copy = os.listdir(src_images_folder)
    for image_idx, image_name in enumerate(pictures_to_copy):
        new_image_name = f'{str(image_idx).zfill(8)}.JPEG'
        src_path = Path(src_images_folder, image_name)
        dst_path = Path(dst_images_folder, new_image_name)
        shutil.copy(src_path, dst_path)
        # _remove_file_if_exists(path=src_path)
