from typing import Tuple
from pathlib import Path
from datetime import datetime

import torch
from src.model import WGAN
import numpy as np
import matplotlib.pyplot as plt


class WGanTrainer(object):
    def __init__(self, save_path: Path, device: torch.device, image_shape: Tuple[int, int], latent_dim: int,
                 image_channel: int, lr: float, alpha: float):
        self._save_path = save_path

        self._plot_save_path = self._save_path.joinpath("plot")
        self._plot_save_path.mkdir(parents=True, exist_ok=True)
        self._images_save_path = self._save_path.joinpath("images")
        self._images_save_path.mkdir(parents=True, exist_ok=True)
        self._model_save_path = self._save_path.joinpath("model")
        self._model_save_path.mkdir(parents=True, exist_ok=True)

        self._latent_dim = latent_dim
        self._image_shape = image_shape
        self._image_channel = image_channel

        self._device = device

        self._lr = lr

        self._alpha = alpha

        self._model = WGAN(image_shape=self._image_shape, image_channels=self._image_channel,
                           latent_dim=self._latent_dim, lr=self._lr, alpha=alpha, device=1)

    def train(self, train_loader, epochs: int, validation_loader):
        res = self._model.fit(train_dataloader=train_loader, epochs=epochs, valid_dataloader=validation_loader,
                              image_save_path=self._images_save_path)

        self._model.save_model(self._model_save_path)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        x = np.arange(epochs) + 1

        train_loss = res["train_loss"]
        valid_loss = res["valid_loss"]

        t_gen = np.array(train_loss[1]).mean(axis=1)
        t_disc = np.array(train_loss[0]).mean(axis=1)
        v_gen = np.array(valid_loss[1]).mean(axis=1)
        v_disc = np.array(valid_loss[0]).mean(axis=1)

        # save loss plot
        plt.title("Without DP and Mixup")

        plt.semilogy(x, t_disc, color=colors[0], label="Train Disc Loss")
        plt.semilogy(x, t_gen, color=colors[1], label="Train Gen Loss")

        plt.semilogy(x, v_disc, color=colors[0], label="Valid Disc Loss", linestyle="-.")
        plt.semilogy(x, v_gen, color=colors[1], label="Valid Gen Loss", linestyle="-.")

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()

        plt.savefig(str(self._plot_save_path.joinpath("loss.png")))
