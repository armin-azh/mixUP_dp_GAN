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

    @staticmethod
    def accuracy(pred: torch.Tensor, label: torch.Tensor):
        return torch.count_nonzero(torch.argmax(pred, axis=1) == label, dim=0) / pred.shape[0]

    def train(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, epochs: int, frequency: int = 5):

        print("[READY] training is now starting ...")
        glob_train_loss = []
        glob_valid_loss = []
        glob_train_acc = []
        glob_valid_acc = []
        for epoch in range(epochs):

            train_loss = []
            valid_loss = []
            train_acc = []
            valid_acc = []

            for batch_idx, data in enumerate(train_dataloader):
                real, label = data
                real = real.float().to(self._device)
                label = label.long().to(self._device)

                self._opt.zero_grad()
                pred = self._detector(real)
                loss = self._criterion(pred, label)
                loss.backward()
                self._opt.step()

                _acc = self.accuracy(pred, label)

                train_loss.append(loss.item())
                train_acc.append(_acc.item())

                # display
                if batch_idx % frequency == 0:
                    print('[TRAIN] => [%d/%d][%d/%d]\tLoss: %.4f\tAccuracy: %.4f'
                          % (epoch + 1, epochs, batch_idx + 1, len(train_dataloader), loss.item(), _acc.item()))

            glob_train_loss.append(train_loss)
            glob_train_acc.append(train_acc)

            # check validation
            for val_data in valid_dataloader:
                val_real, val_label = val_data
                val_real = val_real.float().to(self._device)
                val_label = val_label.long().to(self._device)

                val_pred = self._detector(val_real)
                val_loss = self._criterion(val_pred, val_label)

                _val_acc = self.accuracy(val_pred, val_label)

                valid_loss.append(val_loss.item())
                valid_acc.append(_val_acc.item())

            glob_valid_loss.append(valid_loss)
            glob_valid_acc.append(valid_acc)

        return {"loss": [glob_train_loss, glob_valid_loss], "accuracy": [glob_train_acc, glob_valid_acc],
                "epochs": epochs}

    def save_model(self, file_name: Path) -> None:
        """
        save path directory
        :param file_name:
        :return: None
        """

        torch.save(self._detector.state_dict(), file_name.joinpath("detector.pth"))

    def plot(self, res: dict, save_path: Path, posix: str, t_p: str = "acc") -> None:
        """
        render and create plot
        :param t_p:
        :param posix:
        :param res: training result
        :param save_path: path to save
        :return: none
        """

        weight_paths = save_path.joinpath("weights")
        weight_paths.mkdir(parents=True, exist_ok=True)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        x = np.arange(res.get("epochs")) + 1

        train_loss, valid_loss = res["loss"]
        
        plt.figure(figsize=(20,20))

        plt.subplot(2, 1, 1)

        t_loss = np.array(train_loss).mean(axis=1)
        v_loss = np.array(valid_loss).mean(axis=1)

        np.save(str(weight_paths.joinpath("train_loss.npy")), t_loss)
        np.save(str(weight_paths.joinpath("valid_loss.npy")), v_loss)

        plt.title(f"Loss ({posix})")

        plt.semilogy(x, t_loss, color=colors[0], label="Train Loss")
        plt.semilogy(x, v_loss, color=colors[1], label="valid Loss", linestyle="--")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        train_acc, valid_acc = res["accuracy"]

        t_acc = np.array(train_acc).mean(axis=1)
        v_acc = np.array(valid_acc).mean(axis=1)

        np.save(str(weight_paths.joinpath("train_acc.npy")), t_acc)
        np.save(str(weight_paths.joinpath("valid_acc.npy")), v_acc)

        plt.subplot(2, 1, 2)

        plt.title(f"Accuracy ({posix})")

        plt.plot(x, t_acc, color=colors[0], label="Train Accuracy")
        plt.plot(x, v_acc, color=colors[1], label="valid Accuracy", linestyle="--")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(str(save_path.joinpath("results.png")))

        if self._has_tensorboard:
            _im = io.imread(str(save_path.joinpath("results.png")))
            _im = ToTensor()(_im)
            self._writer.add_image("Final results", _im, 1)
