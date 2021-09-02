from typing import Tuple, List
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from sklearn.model_selection import train_test_split
from skimage import io
from skimage.transform import resize


class ZeroDayDataset(Dataset):
    def __init__(self, images_path: List[Path], labels: List[int], output_size: Tuple[int, int] = (63, 135)):
        self._file_names = images_path
        self._labels = labels
        self._output_size = output_size

    def __len__(self):
        return len(self._file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _im_p = str(self._file_names[idx])
        _im = io.imread(_im_p)
        _im = resize(_im, self._output_size)
        _im = np.expand_dims(_im, axis=-1)
        _im = ToTensor()(_im)

        return _im, self._labels[idx]


def zero_data_dataset(bs_dir: Path, label_csv: Path, train_size: float, test_size: float, shuffle: bool,
                      random_state: int, output_size: Tuple[int, int] = (63, 135)) -> Tuple[
    ZeroDayDataset, ZeroDayDataset]:
    """

    :param output_size: image output size (w,h)
    :param bs_dir: images directory
    :param label_csv: label csv file
    :param train_size: float number in [0,1] range
    :param test_size: float number in [0,1] range
    :param shuffle: if True you can apply shuffling on the dataset
    :param random_state: set random seed
    :return: train dataset, validation dataset
    """
    file_names = list(bs_dir.glob("*.png"))
    label_df = pd.read_csv(label_csv)
    labels = []

    # extract labels
    for f in file_names:
        labels.append(label_df.loc[label_df["Id"] == f.stem]["Class"].values[0])

    x_train, x_valid, y_train, y_valid = train_test_split(file_names, labels, train_size=train_size,
                                                          test_size=test_size, random_state=random_state,
                                                          shuffle=shuffle)

    return ZeroDayDataset(x_train, y_train, output_size), ZeroDayDataset(x_valid, y_valid, output_size)

# ps = Path("/home/lezarus/Documents/Project/mixUpGAN/data/zero_day/images")
# lb = Path("/home/lezarus/Documents/Project/mixUpGAN/data/zero_day/trainLabels.csv")
# train_ds, test_ds = zero_data_dataset(bs_dir=ps, label_csv=lb, train_size=0.8, test_size=0.2, shuffle=True,
#                                       random_state=18)
