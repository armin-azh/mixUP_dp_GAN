import numpy as np

from torch import nn
import torch
from torch.utils.data import DataLoader

from src.model import AutoEncodeCustomV1


class Container:
    def __init__(self, *args, **kwargs):
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
        super(AutoEncoderCV1Container, self).__init__(*args, **kwargs)

    def fit(self, *args, **kwargs) -> dict:
        """
        fit the dataset on the auto encoder model
        :return: result
        """

        total_loss = []
        for epoch in range(self._epochs):
            for data in self._train:
                img, _ = data
                img = torch.autograd.Variable(img)

                output = self._ae.forward(img)
                loss = self._criterion(output, img)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
            total_loss.append(loss.data)
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, self._epochs, np.mean(total_loss)))
        return {"loss": total_loss, "epoch": self._epochs}
