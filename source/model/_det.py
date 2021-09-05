from typing import Union
from pathlib import Path
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from skimage import io

from ._arch import Detector


class ZeroDayDetector:
    def __init__(self, lr: float, image_channel: int, feature_maps: int, tensorboard: bool, log_dir: Path,
                 state_dict: Union[None, OrderedDict], classes: int, weight_decay: float,
                 device: torch.device):
        self._lr = lr
        self._weight_decay = weight_decay
        self._image_channel = image_channel
        self._feature_maps = feature_maps
        self._has_tensorboard = tensorboard
        self._writer = SummaryWriter(log_dir=str(log_dir))
        self._device = device

        # defile detector network
        self._detector = Detector(image_channel=self._image_channel, features=self._feature_maps, classes=classes,
                                  state_dic=state_dict).to(self._device)

        # introduce the optimizer
        self._opt = torch.optim.RMSprop(params=self._detector.parameters(), lr=self._lr,
                                        weight_decay=self._weight_decay)

        # loss
        self._criterion = nn.CrossEntropyLoss()

    def train(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, epochs: int):
        pass

    def save_model(self, file_name: Path) -> None:
        """
        save path directory
        :param file_name:
        :return: None
        """

        torch.save(self._detector.state_dict(), file_name.joinpath("detector.pth"))

    def plot(self, res: dict, save_path: Path) -> None:
        """
        render and create plot
        :param res: training result
        :param save_path: path to save
        :return: none
        """
        weight_paths = save_path.joinpath("weights")
        weight_paths.mkdir(parents=True, exist_ok=True)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        x = np.arange(res.get("epochs")) + 1

        train_loss = res["train_loss"]
        valid_loss = res["valid_loss"]
        has_valid = res["has_valid"]

        t_gen = np.array(train_loss[1]).mean(axis=1)
        t_disc = np.array(train_loss[0]).mean(axis=1)

        np.save(str(weight_paths.joinpath("train_generator_loss.npy")), t_gen)
        np.save(str(weight_paths.joinpath("train_discriminator_loss.npy")), t_disc)

        if has_valid:
            v_gen = np.array(valid_loss[1]).mean(axis=1)
            v_disc = np.array(valid_loss[0]).mean(axis=1)

            np.save(str(weight_paths.joinpath("valid_generator_loss.npy")), v_gen)
            np.save(str(weight_paths.joinpath("valid_discriminator_loss.npy")), v_disc)

        plt.title(self._title)

        plt.plot(x, t_disc, color=colors[0], label="Train Disc Loss")
        plt.plot(x, t_gen, color=colors[1], label="Train Gen Loss")

        if has_valid:
            plt.plot(x, v_disc, color=colors[0], label="Valid Disc Loss", linestyle="-.")
            plt.plot(x, v_gen, color=colors[1], label="Valid Gen Loss", linestyle="-.")

        plt.hlines(y=0, xmin=0, xmax=res.get("epochs"), colors=colors[2], linestyles="dashed")

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()

        plt.savefig(str(save_path.joinpath("loss.png")))

        if self._has_tensorboard:
            _im = io.imread(str(save_path.joinpath("loss.png")))
            _im = ToTensor()(_im)
            self._writer.add_image("Final Result", _im, 1)
