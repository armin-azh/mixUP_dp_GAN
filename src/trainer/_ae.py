import numpy as np
from torch import nn
import torch

from src.model import AutoEncodeCustomV1


class AutoEncoderCV1Trainer(nn.Module):
    """
    AutoEncodeCustomV1 trainer
    """

    def __init__(self, im_channel: int = 1, lr: float = 1e-3, weight_decay: float = 1e-5, epochs: int = 10):
        super(AutoEncoderCV1Trainer, self).__init__()
        self._ae = AutoEncodeCustomV1(im_channel=im_channel)
        self._criterion = nn.MSELoss()
        self._optimizer = torch.optim.Adam(params=self._ae.parameters(),
                                           lr=lr,
                                           weight_decay=weight_decay)
        self._epochs = epochs

    def fit(self, dataloader) -> dict:
        """
        fit the dataset on the auto encoder model
        :param dataloader:
        :return: result
        """

        total_loss = []
        for epoch in range(self._epochs):
            for data in dataloader:
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
