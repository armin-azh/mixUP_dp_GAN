from typing import Tuple, List
from pathlib import Path
import pandas as pd

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from skimage import io


class ZeroDayDataset(Dataset):
    def __init__(self, images_path: List[Path], labels: List[int]):
        self._file_names = images_path
        self._labels = labels

    def __len__(self):
        return len(self._file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _im_p = str(self._file_names[idx])
        _im = io.imread(_im_p)

        return _im,self._labels[idx]


def zero_data_dataset(bs_dir: Path, label_csv: Path, train_size: float, test_size: float, shuffle: bool,
                      random_state: int) -> Tuple[ZeroDayDataset, ZeroDayDataset]:
    """

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

    return ZeroDayDataset(x_train, y_train), ZeroDayDataset(x_valid, y_valid)


# ps = Path("/home/lezarus/Documents/Project/mixUpGAN/data/zero_day/images")
# lb = Path("/home/lezarus/Documents/Project/mixUpGAN/data/zero_day/trainLabels.csv")
# train_ds, test_ds = zero_data_dataset(bs_dir=ps, label_csv=lb, train_size=0.8, test_size=0.2, shuffle=True,
#                                       random_state=18)
# print(type(train_ds[0]))
