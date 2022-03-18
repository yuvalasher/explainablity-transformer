import shutil
import random
from pathlib import Path
from utils.consts import ORIGINAL_IMAGES_FOLDER_PATH, DGX_IMAGES_FOLDER_PATH

NUM_PICTURES: int = 1000

if __name__ == '__main__':
    pictures_to_copy = random.sample(range(0, 50000), NUM_PICTURES)
    for picture in pictures_to_copy:
        picture_name = f'{str(picture).zfill(8)}.JPEG'
        shutil.copyfile(Path(ORIGINAL_IMAGES_FOLDER_PATH, picture_name), Path(DGX_IMAGES_FOLDER_PATH, picture_name))
