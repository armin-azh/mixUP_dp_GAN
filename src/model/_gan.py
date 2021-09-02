from typing import Union, Tuple
from pathlib import Path
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
import torchvision

from ._gen import DCGANGenerator
from ._disc import DCGANDiscriminator

from pytorch_lightning import LightningModule


class DCGAN(nn.Module):
    def __init__(
            self,
            beta1: float = 0.5,
            feature_maps_gen: int = 64,
            feature_maps_disc: int = 64,
            image_channels: int = 1,
            latent_dim: int = 100,
            lr: float = 0.0002,
            device: int = 0):
        super(DCGAN, self).__init__()
        self._betas = (beta1, 0.999)
        self._image_channels = image_channels
        self._latent_dim = latent_dim
        self._gen_feature_map = feature_maps_gen
        self._disc_feature_map = feature_maps_disc
        self._lr = lr
        self._device = torch.device("cuda:0" if (torch.cuda.is_available() and device > 0) else "cpu")

        # loss function
        self._criterion = nn.BCELoss()

        # module
        self._generator = self._make_generator()
        self._discriminator = self._make_discriminator()

        # optimizers
        self._gen_opt = torch.optim.Adam(params=self._generator.parameters(), lr=self._lr, betas=self._betas)
        self._disc_opt = torch.optim.Adam(params=self._discriminator.parameters(), lr=self._lr, betas=self._betas)

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    def _make_generator(self) -> nn.Module:
        """
        get generator module
        :return: module
        """
        gen = DCGANGenerator(latent_dim=self._latent_dim, feature_maps=self._gen_feature_map,
                             image_channels=self._image_channels).to(self._device)
        gen.apply(self._weights_init)
        return gen

    def _make_discriminator(self) -> nn.Module:
        """
        get discriminator module
        :return: module
        """
        disc = DCGANDiscriminator(feature_maps=self._disc_feature_map, image_channels=self._image_channels).to(
            self._device)
        disc.apply(self._weights_init)
        return disc

    def _forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        generator forward propagation
        :param noise:
        :return:
        """
        noise = noise.view(*noise.shape, 1, 1)
        return self._generator.forward(noise)

    def _get_simple_noise(self, n_samples: int, latent_dim: int) -> torch.Tensor:
        """
        get new noise
        :param n_samples: batch size
        :param latent_dim: latent dimension
        :return: noise
        """
        return torch.randn(n_samples, latent_dim, device=self._device)

    def _gen_fake_pred(self, real: torch.Tensor) -> torch.Tensor:
        """
        reproduce new fake samples
        :param real: new batch samples
        :return:
        """
        batch_size = len(real)
        noise = self._get_simple_noise(n_samples=batch_size, latent_dim=self._latent_dim)
        fake = self._forward(noise=noise)
        fake_pred = self._discriminator.forward(fake)
        return fake_pred

    def _get_gen_loss(self, real: torch.Tensor) -> torch.Tensor:
        """
        get generator loss
        :param real: fake tensor
        :return: generator loss
        """
        fake_pred = self._gen_fake_pred(real)
        fake_gt = torch.ones_like(fake_pred)
        gen_loss = self._criterion(fake_pred, fake_gt)

        return gen_loss

    def _get_disc_loss(self, real: torch.Tensor) -> torch.Tensor:
        """
        get generator loss
        :param real:
        :return:
        """
        real_pred = self._discriminator.forward(real)
        real_gt = torch.ones_like(real_pred)
        real_loss = self._criterion(real_pred, real_gt)

        # Train with fake
        fake_pred = self._gen_fake_pred(real)
        fake_gt = torch.zeros_like(fake_pred)
        fake_loss = self._criterion(fake_pred, fake_gt)

        disc_loss = real_loss + fake_loss

        return disc_loss

    def _gen_step(self, real: torch.Tensor) -> torch.Tensor:
        """
        preform generator loss
        :param real:
        :return:
        """
        gen_loss = self._get_gen_loss(real)
        return gen_loss

    def _disc_step(self, real: torch.Tensor) -> torch.Tensor:
        """
        perform discriminator loss
        :param real:
        :return:
        """
        disc_loss = self._get_disc_loss(real)
        return disc_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _ = batch

        # Train discriminator
        result = None
        if optimizer_idx == 0:
            result = self._disc_step(real)

        # Train generator
        if optimizer_idx == 1:
            result = self._gen_step(real)

        return result

    def _summary(self):
        print(f"[Device] {self._device}")

    def fit(self, train_dataloader, epochs: int, frequency: int = 5, valid_dataloader: Union[None, DataLoader] = None,
            image_save_path: Union[Path, None] = None):
        self._summary()

        glob_gen_loss = []
        glob_disc_loss = []

        val_glob_gen_loss = []
        val_glob_disc_loss = []

        print("[READY] training is now starting ...")
        for epoch in range(epochs):
            gen_loss = []
            disc_loss = []

            # train phase
            for batch_idx, data in enumerate(train_dataloader):
                real_data = data[0].float().to(self._device)

                self._discriminator.zero_grad()
                error_d_real = self._disc_step(real_data)
                error_d_real.backward()
                self._disc_opt.step()
                disc_loss.append(error_d_real.item())

                self._generator.zero_grad()
                error_g_fake = self._gen_step(real_data)
                error_g_fake.backward()
                self._gen_opt.step()
                gen_loss.append(error_g_fake.item())

                if batch_idx % frequency == 0:
                    print('[TRAIN] => [%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                          % (epoch + 1, epochs, batch_idx + 1, len(train_dataloader), error_d_real.item(),
                             error_g_fake.item()))

            glob_disc_loss.append(disc_loss)
            glob_gen_loss.append(gen_loss)

            if valid_dataloader is not None:
                val_gen_loss = []
                val_disc_loss = []
                for val_batch_idx, data in enumerate(valid_dataloader):
                    real_data = data[0].float().to(self._device)

                    # discriminator
                    error_d_real = self._disc_step(real_data)
                    val_disc_loss.append(error_d_real.item())

                    # generator
                    error_g_fake = self._gen_step(real_data)
                    val_gen_loss.append(error_g_fake.item())
                val_glob_disc_loss.append(val_disc_loss)
                val_glob_gen_loss.append(val_gen_loss)

            if image_save_path is not None:
                sample_noise = self._get_simple_noise(1, self._latent_dim)
                output = self._forward(sample_noise)
                torchvision.utils.save_image(output, image_save_path.joinpath(f"image_{epoch + 1}.jpg"))

        return {"train_loss": [glob_disc_loss, glob_gen_loss], "valid_loss": [val_glob_disc_loss, val_glob_gen_loss]}

    def save_model(self, file_name: Path) -> None:
        """
        save path directory
        :param file_name:
        :return: None
        """

        torch.save(self._discriminator.state_dict, file_name.joinpath("discriminator.pt"))
        torch.save(self._generator.state_dict, file_name.joinpath("generator.pt"))

    @staticmethod
    def mix_up_observation(data: torch.Tensor, y: torch.Tensor, alpha: float, device: torch.device):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        b_s = data.size()[0]
        index = torch.randperm(b_s).to(device)

        mixed_x = lam * data + (1 - lam) * data[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mix_up_criterion(self, pred: torch.Tensor, target_a, target_b, lam):
        """

        :param pred:
        :param target_a:
        :param target_b:
        :param lam:
        :return:
        """
        return lam * self._criterion(pred, target_a) + (1 - lam) * self._criterion(pred, target_b)
