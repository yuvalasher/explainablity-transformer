from pathlib import Path

import torch
import h5py
from torch.utils.data.dataset import Dataset


class ImagenetResults(Dataset):
    def __init__(self, path):
        """
        :param vis_type: Indicate which iteration's mask should be considered
        """
        super(ImagenetResults, self).__init__()

        self.path: Path = Path(path)
        self.data = None

        print('Reading dataset length...')
        with h5py.File(self.path, 'r') as f:
            # tmp = h5py.File(self.path , 'r')
            self.data_length = len(f['/image'])

    def __len__(self):
        return self.data_length

    def __getitem__(self, item):
        if self.data is None:
            self.data = h5py.File(self.path, 'r')
        vis = torch.tensor(self.data['vis'][item])
        return vis
