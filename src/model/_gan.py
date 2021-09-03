from typing import Union, Tuple
from pathlib import Path
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
import torchvision

from ._gen import DCGANGenerator, WGenerator
from ._disc import DCGANDiscriminator, WDiscriminator


class DCGAN(nn.Module):
    def __init__(
            self,
            beta1: float = 0.5,
            feature_maps_gen: int = 64,
            feature_maps_disc: int = 64,
            image_channels: int = 1,
            latent_dim: int = 100,
            lr: float = 0.0002,
            alpha: float = 1,
            device: int = 0):
        super(DCGAN, self).__init__()
        self._betas = (beta1, 0.999)
        self._image_channels = image_channels
        self._latent_dim = latent_dim
        self._gen_feature_map = feature_maps_gen
        self._disc_feature_map = feature_maps_disc
        self._lr = lr
        self._alpha = alpha
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

    def _get_mix_up_disc_loss(self, real1: torch.Tensor, real2: torch.Tensor) -> torch.Tensor:
        """
        calculate mix up loss value
        :param real1:
        :param real2:
        :return:
        """
        lam = np.random.beta(self._alpha, self._alpha)

        perm1 = torch.randperm(real1.size(0), device=self._device)
        # fake1 = self._forward(noise=self._get_simple_noise(n_samples=len(real1), latent_dim=self._latent_dim))
        # data1 = torch.cat([real1, fake1])
        # ones1 = torch.autograd.Variable(torch.ones(real1.size(0), 1))
        # zeros1 = torch.autograd.Variable(torch.zeros(fake1.size(0), 1))
        # label1 = torch.cat([ones1, zeros1])

        data1 = real1[perm1]
        label1 = torch.ones(data1.size(0), device=self._device)

        perm2 = torch.randperm(real2.size(0), device=self._device)
        # fake2 = self._forward(noise=self._get_simple_noise(n_samples=len(real2), latent_dim=self._latent_dim))
        # data2 = torch.cat([real2, fake2])
        # ones2 = torch.autograd.Variable(torch.ones(real2.size(0), 1))
        # zeros2 = torch.autograd.Variable(torch.zeros(fake2.size(0), 1))
        # label2 = torch.cat([ones2, zeros2])

        data2 = real2[perm2]
        label2 = torch.ones(data2.size(0), device=self._device)

        mixed_data = lam * data1 + (1 - lam) * data2
        mixed_label = lam * label1 + (1 - lam) * label2

        real_pred = self._discriminator.forward(mixed_data)
        real_gt = mixed_label
        real_loss = self._criterion(real_pred, real_gt)

        # Train with fake
        fake_pred = self._gen_fake_pred(mixed_data)
        fake_gt = torch.zeros_like(fake_pred)
        fake_loss = self._criterion(fake_pred, fake_gt)

        disc_loss = real_loss + fake_loss

        return disc_loss

    def _disc_mix_up_step(self, real1: torch.Tensor, real2: torch.Tensor) -> torch.Tensor:
        """
        return loss value
        :param real1:
        :param real2:
        :return:
        """
        return self._get_mix_up_disc_loss(real1=real1, real2=real2)

    def mix_up_data(self, real1: torch.Tensor, label1: torch.Tensor, real2: torch.Tensor, label2: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        mix up the 2 batches
        :param label2:
        :param label1:
        :param real1: batch of first dataloader
        :param real2: batch of second dataloader
        :return: mixed up data
        """
        _d = self._alpha * real1 + (1 - self._alpha) * real2
        _l = self._alpha * label1 + (1 - self._alpha) * label2
        return _d, _l

    def fit_with_mix_up(self, train_dataloader, train_dataloader_2, epochs: int, frequency: int = 5,
                        valid_dataloader: Union[None, DataLoader] = None,
                        image_save_path: Union[Path, None] = None):
        """
        fit the model with mix up augmentation
        :param train_dataloader:
        :param train_dataloader_2:
        :param epochs:
        :param frequency:
        :param valid_dataloader:
        :param image_save_path:
        :return:
        """
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
            for batch_idx, data in enumerate(zip(train_dataloader, train_dataloader_2)):
                (data1, y1), (data2, y2) = data
                data1 = data1.float().to(self._device)
                data2 = data2.float().to(self._device)

                self._discriminator.zero_grad()
                error_d_real = self._disc_mix_up_step(data1, data2)
                error_d_real.backward()
                self._disc_opt.step()
                disc_loss.append(error_d_real.item())

                self._generator.zero_grad()
                error_g_fake = self._gen_step(data1)
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


class WGAN(nn.Module):
    def __init__(
            self,
            beta1: float = 0.5,
            image_channels: int = 1,
            image_shape: Tuple[int, int] = (64, 128),
            latent_dim: int = 100,
            lr: float = 0.0002,
            alpha: float = 1,
            device: int = 0):
        super(WGAN, self).__init__()
        self._betas = (beta1, 0.999)
        self._image_channels = image_channels
        self._latent_dim = latent_dim
        self._image_shape = image_shape
        self._lr = lr
        self._alpha = alpha
        self._device = torch.device("cuda:0" if (torch.cuda.is_available() and device > 0) else "cpu")

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
        gen = WGenerator(latent_dim=self._latent_dim, image_shape=self._image_shape).to(self._device)
        gen.apply(self._weights_init)
        return gen

    def _make_discriminator(self) -> nn.Module:
        """
        get discriminator module
        :return: module
        """
        disc = WDiscriminator(image_shape=self._image_shape).to(self._device)
        disc.apply(self._weights_init)
        return disc

    def _forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        generator forward propagation
        :param noise:
        :return:
        """
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
        gen_loss = -torch.mean(fake_pred)
        return gen_loss

    def _get_disc_loss(self, real: torch.Tensor) -> torch.Tensor:
        """
        get generator loss
        :param real:
        :return:
        """
        real_pred = self._discriminator.forward(real)

        # Train with fake
        fake_pred = self._gen_fake_pred(real)

        disc_loss = -torch.mean(real_pred) + torch.mean(fake_pred)

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

                for p in self._discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

                if batch_idx % frequency == 0:
                    self._generator.zero_grad()
                    error_g_fake = self._gen_step(real_data)
                    error_g_fake.backward()
                    self._gen_opt.step()
                    gen_loss.append(error_g_fake.item())
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

        torch.save(self._discriminator.state_dict, file_name.joinpath("w_discriminator.pt"))
        torch.save(self._generator.state_dict, file_name.joinpath("w_generator.pt"))
