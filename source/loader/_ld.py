from typing import Tuple

from pathlib import Path

from torch.utils.data import DataLoader

from ._zero_day import zero_data_dataset


def zero_data_load(bs_dir: Path, label_csv: Path, train_size: float, test_size: float, shuffle: bool,
                   random_state: int, batch_size: int, num_worker: int = 0,
                   output_size: Tuple[int, int] = (63, 135)) -> Tuple[DataLoader, DataLoader]:
    """

    :param output_size: output image (w,h)
    :param bs_dir: images directory
    :param label_csv: label csv file
    :param train_size: float number in [0,1] range
    :param test_size: float number in [0,1] range
    :param shuffle: if True you can apply shuffling on the dataset
    :param random_state: set random seed
    :param batch_size: number of batches
    :param num_worker: number of thread worker for multiprocessing
    :return: train loader, validation loader
    """
    trains_ds, valid_ds = zero_data_dataset(bs_dir, label_csv, train_size, test_size, shuffle, random_state,
                                            output_size)
    return DataLoader(trains_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker), DataLoader(valid_ds,
                                                                                                             batch_size=batch_size,
                                                                                                             shuffle=shuffle,
                                                                                                             num_workers=num_worker)
