from typing import Tuple
import torch.nn as nn
import torch
import numpy as np


class SimpleGenerator(nn.Module):
    def __init__(self, latent_size: int, image_shape: Tuple[int, int]):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(latent_size, 2 * latent_size),
            nn.ReLU(),
            nn.Linear(2 * latent_size, int(np.prod(image_shape))))

    def forward(self, x):
        return self.main(x)


class SimpleDiscriminator(nn.Module):
    def __init__(self, image_size: Tuple[int, int]):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(int(np.prod(image_size)), int(np.prod(image_size) / 2)),
            nn.ReLU(),
            nn.Linear(int(np.prod(image_size) / 2), 1),
        )

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    def __init__(self, latent_dim: int, image_channel: int, features: int):
        """
        generator network
        :param latent_dim: latent dimension for create noise
        :param image_channel: image channel (default ~ 1)
        :param features: hidden feature map
        """
        super(Generator, self).__init__()
        self._latent_size = latent_dim
        self._image_channel = image_channel
        self._features = features
        self._model = nn.Sequential(
            self.make_block(self._latent_size, self._features * 8, kernel_size=4, stride=1, padding=0),
            self.make_block(self._features * 8, self._features * 4, kernel_size=3, padding=0),
            self.make_block(self._features * 4, self._features * 2),
            self.make_block(self._features * 2, self._features * 1),
            self.make_block(self._features * 1, self._features * 1),
            self.make_block(self._features, self._image_channel, kernel_size=(3, 4), stride=(1, 2), final_layer=True),
        )

    @staticmethod
    def make_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.Tanh(),
            )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        forward propagation throughout network
        :param noise: tensor in shape (b,latent_dim)
        :return: tensor in shape (b,image_channel,w,h)
        """
        x = noise.view(len(noise), self._latent_size, 1, 1)
        return self._model(x)


class Critic(nn.Module):
    def __init__(self, image_channel: int, features: int):
        """
        discriminator network
        :param image_channel: number of image channels
        :param features: number of hidden layers
        """
        super(Critic, self).__init__()
        self._features = features
        self._image_channel = image_channel
        self._model = nn.Sequential(
            self.make_block(self._image_channel, self._features),
            self.make_block(self._features, self._features * 2),
            self.make_block(self._features * 2, 1, final_layer=True),
        )

    @staticmethod
    def make_block(in_channels, out_channels, kernel_size=4, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        forward propagation
        :param img: tensor in shape (b,image_channel,w,h)
        :return: tensor in shape (b,m)
        """
        _pred = self._model(img)
        return _pred.view(len(_pred), -1)


class Detector(nn.Module):
    def __init__(self, image_channel: int, features: int, classes: int, state_dic):
        """
        detector
        :param image_channel: number of image channels
        :param features: number of features maps
        :param classes: number of classes
        :param state_dic: if use transfer learning
        """
        super(Detector, self).__init__()

        self._bone = Critic(image_channel=image_channel, features=features)

        if state_dic is not None:
            self._bone.load_state_dict(state_dict=state_dic)

        self._seq_model = nn.Sequential(
            self._bone,
            nn.Flatten(),
            nn.Linear(in_features=84, out_features=classes),
            nn.Softmax(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self._seq_model(img)
