from typing import Tuple
import torch
import numpy as np

# arch
from ._arch import SimpleGenerator, SimpleDiscriminator


class WGan:
    def __init__(self, image_size: Tuple[int, int], latent_dim: int, beta1: float, lr: float):
        self._latent_dim = latent_dim
        self._image_size = image_size
        self._betas = (beta1, 0.999)
        self._learning_rate = lr

        self._generator = SimpleGenerator(latent_size=self._latent_dim, image_shape=self._image_size)
        self._discriminator = SimpleDiscriminator(image_size=image_size)

        self_gen_opt = torch.optim.Adam(self._generator.parameters(), lr=self._learning_rate, betas=self._betas)
        self._disc_opt = torch.optim.Adam(self._discriminator.parameters(), lr=self._learning_rate, betas=self._betas)



