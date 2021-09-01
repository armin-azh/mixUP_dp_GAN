from typing import Union
import numpy as np
from datetime import datetime
from pathlib import Path

from torch import nn
import torch
import torchvision
from torch.utils.data import DataLoader

from src.model import AutoEncodeCustomV1


class Container:
    def __init__(self, save_path: Path, *args, **kwargs):
        self._save_path = save_path
        if self._save_path is not None:
            _cu = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
            self._save_path = self._save_path.joinpath(_cu)
            self._save_path.mkdir(parents=True, exist_ok=True)

        super(Container, self).__init__(*args, **kwargs)

    def fit(self, *args, **kwargs):
        raise NotImplementedError


class AutoEncoderCV1Container(Container):
    """
    AutoEncodeCustomV1 trainer
    """

    def __init__(self,
                 train_dataloader: DataLoader,
                 validation_dataloader: DataLoader,
                 im_channel: int = 1,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 epochs: int = 10,
                 save_path: Union[Path, None] = None,
                 *args,
                 **kwargs):
        self._train = train_dataloader
        self._valid = validation_dataloader
        self._ae = AutoEncodeCustomV1(im_channel=im_channel)
        self._criterion = nn.MSELoss()
        self._optimizer = torch.optim.Adam(params=self._ae.parameters(),
                                           lr=lr,
                                           weight_decay=weight_decay)
        self._epochs = epochs
        if save_path is not None:
            save_path = save_path.joinpath("auto_encoder")
        super(AutoEncoderCV1Container, self).__init__(save_path=save_path, *args, **kwargs)

    def _summary(self):
        print(f"Train set batch size: {len(self._train)}")
        print(f"Validation set batch size: {len(self._valid)}")
        print(f"Epoch: {self._epochs}")

    def fit(self, *args, **kwargs) -> dict:
        """
        fit the dataset on the auto encoder model
        :return: result
        """

        total_loss = []
        self._summary()
        print("starting ...")
        for epoch in range(self._epochs):
            print("[Running] #", end='')
            for data in self._train:
                print("\b", end="")
                print(">", end="")
                img, _ = data
                img = torch.autograd.Variable(img.float())

                output = self._ae.forward(img)
                loss = self._criterion(output, img)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                print("#", end="")
            print("\n")
            total_loss.append(loss.data)
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, self._epochs, np.mean(total_loss)))
            if epoch % 10 == 0 and self._save_path is not None:
                pic = output.cpu().data
                torchvision.utils.save_image(pic, self._save_path.joinpath(f"image_{epoch}.jpg"))

        return {"loss": total_loss, "epoch": self._epochs}
