from typing import Tuple

import torch
from torch import nn
import numpy as np


class CustomGeneratorV1(nn.Module):
    def __init__(self, gpu_num: int, feature_num: int, z_num: int, output_channel: int) -> None:
        super(CustomGeneratorV1, self).__init__()
        self._gpu_num = gpu_num
        self._num_feature = feature_num
        self._num_z_vec = z_num
        self._output_channel = output_channel
        self._sequence_model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self._num_z_vec,
                               out_channels=self._num_feature * 8,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(self._num_feature * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=self._num_feature * 8,
                               out_channels=self._num_feature * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(self._num_feature * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=self._num_feature * 4,
                               out_channels=self._num_feature * 2,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(self._num_feature * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=self._num_feature * 2,
                               out_channels=self._num_feature,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(self._num_feature),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=self._num_feature,
                               out_channels=self._output_channel,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.Tanh()
        )

    def forward(self, input_tensor):
        return self._sequence_model(input_tensor)


class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim: int, feature_maps: int, image_channels: int):
        """

        :param latent_dim: dimension of latent space
        :param feature_maps: number of feature map
        :param image_channels: number of image channel in database
        """
        super(DCGANGenerator, self).__init__()
        self._gen = nn.Sequential(
            self._gen_block(latent_dim, feature_maps * 16, kernel_size=4, stride=1, padding=0),
            self._gen_block(feature_maps * 16, feature_maps * 8),
            self._gen_block(feature_maps * 8, feature_maps * 4),
            self._gen_block(feature_maps * 4, feature_maps * 2),
            self._gen_block(feature_maps * 2, feature_maps * 1),
            self._gen_block(feature_maps, image_channels, last_block=True, stride=(1, 2), kernel_size=(3, 4)),
        )

    @staticmethod
    def _gen_block(in_channel: int, out_channel: int, kernel_size: int = 4, stride: int = 2, padding: int = 1,
                   bias: bool = False, last_block: bool = False) -> nn.Sequential:
        """

        :param in_channel: input channel
        :param out_channel: output channel
        :param kernel_size: filter size
        :param stride: stride
        :param padding: padding
        :param bias: has bias
        :param last_block: is last block
        :return: Sequential module
        """

        if not last_block:
            block = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
            )
        else:
            block = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias),
                nn.Tanh(),
            )
        return block

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        forward propagation
        :param noise: noise tensor
        :return:
        """
        return self._gen(noise)


class WGenerator(nn.Module):
    def __init__(self, latent_dim: int, image_shape: Tuple[int, int]):
        super(WGenerator, self).__init__()
        self._latent_dim = latent_dim
        self._image_shape = image_shape
        self._seq_model = nn.Sequential(
            *self.make_block(self._latent_dim, 128, normalized=False),
            *self.make_block(128, 256),
            *self.make_block(256, 512),
            *self.make_block(512, 1024),
            nn.Linear(1024, int(np.prod(self._image_shape))),
            nn.Tanh()
        )

    @staticmethod
    def make_block(in_feature: int, out_feature: int, normalized: bool = True):
        layers = [nn.Linear(in_features=in_feature, out_features=out_feature)]
        if normalized:
            layers.append(nn.BatchNorm1d(out_feature, 0.8))

        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        img = self._seq_model(noise)
        return img.view(img.size(0), *self._image_shape)
