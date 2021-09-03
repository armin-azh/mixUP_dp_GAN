from typing import Tuple
import torch
from torch import nn
import numpy as np


class CustomDiscriminatorV1(nn.Module):
    def __init__(self, gpu_num: int, feature_num: int, input_channel: int):
        super(CustomDiscriminatorV1, self).__init__()
        self._input_channel = input_channel
        self._num_feature = feature_num
        self._gpu_num = gpu_num

        self._sequence_model = nn.Sequential(
            nn.Conv2d(in_channels=self._input_channel,
                      out_channels=self._num_feature,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self._input_channel,
                      out_channels=self._input_channel * 2,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(self._input_channel * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self._input_channel * 2,
                      out_channels=self._input_channel * 4,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(self._input_channel * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self._input_channel * 8,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.Sigmoid()

        )

    def forward(self, input_tensor):
        return self._sequence_model(input_tensor)


class DCGANDiscriminator(nn.Module):
    def __init__(self, feature_maps: int, image_channels: int) -> None:
        super(DCGANDiscriminator, self).__init__()

        self._disc = nn.Sequential(
            self._dic_block(image_channels, feature_maps, batch_norm=False),
            self._dic_block(feature_maps, feature_maps * 2),
            self._dic_block(feature_maps * 2, feature_maps * 4),
            self._dic_block(feature_maps * 4, feature_maps * 8),
            self._dic_block(feature_maps * 8, 1, kernel_size=4),
            self._dic_block(1, 1, kernel_size=(2, 4), padding=0, last_block=True),
        )

    @staticmethod
    def _dic_block(in_channel: int, out_channel: int, kernel_size: int = 4, stride: int = 2, padding: int = 1,
                   bias: bool = False, batch_norm: bool = True, last_block: bool = False) -> nn.Sequential:
        """

        :param in_channel: input channels
        :param out_channel: output channels
        :param kernel_size: kernel size
        :param stride: stride
        :param padding: padding
        :param bias: has bias
        :param batch_norm: batch norm
        :param last_block: is last block
        :return:
        """

        if not last_block:
            block = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channel) if batch_norm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
            )

        else:
            block = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias),
                nn.Sigmoid(),
            )

        return block

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        _res = self._disc(input_tensor)
        return _res.view(-1, 1).squeeze(1)


class WDiscriminator(nn.Module):
    def __init__(self, image_shape: Tuple[int, int]):
        super(WDiscriminator, self).__init__()
        self._image_shape = image_shape

        self._seq_model = nn.Sequential(
            nn.Linear(int(np.prod(self._image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)

        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        flatten = img.view(img.size(0), -1)
        return self._seq_model(flatten)
