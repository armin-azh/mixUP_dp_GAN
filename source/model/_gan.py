from pathlib import Path
from typing import Tuple, Union
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# arch
from ._arch import SimpleGenerator, SimpleDiscriminator
from ._arch import Generator, Critic


class WGan:
    def __init__(self, image_size: Tuple[int, int], latent_dim: int, beta1: float, lr: float, image_channel: int,
                 gen_features: int, disc_features: int, critic_repeat: int, c_lambda: int, device: torch.device):
        self._latent_dim = latent_dim
        self._image_size = image_size
        self._betas = (beta1, 0.999)
        self._learning_rate = lr
        self._image_channel = image_channel
        self._gen_features = gen_features
        self._disc_features = disc_features
        self._critic_repeat = critic_repeat
        self._c_lambda = c_lambda
        self._device = device

        self._generator = Generator(latent_dim=self._latent_dim, image_channel=self._image_channel,
                                    features=self._gen_features).to(self._device)
        self._generator.apply(self.weights_init)

        self._discriminator = Critic(image_channel=self._image_channel, features=self._disc_features).to(self._device)
        self._discriminator.apply(self.weights_init)

        self._gen_opt = torch.optim.Adam(self._generator.parameters(), lr=self._learning_rate, betas=self._betas)
        self._disc_opt = torch.optim.Adam(self._discriminator.parameters(), lr=self._learning_rate, betas=self._betas)

    def get_noise(self, n_samples: int):
        return torch.randn(n_samples, self._latent_dim, device=self._device)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def get_gradient(self, real: torch.Tensor, fake: torch.Tensor, epsilon):
        mixed_images = real * epsilon + fake * (1 - epsilon)
        mixed_scores = self._discriminator(mixed_images)
        grad = torch.autograd.grad(inputs=mixed_images, outputs=mixed_scores, grad_outputs=torch.ones_like(mixed_scores)
                                   , create_graph=True, retain_graph=True)[0]
        return grad

    @staticmethod
    def gradient_penalty(grad: torch.Tensor):
        grad = grad.view(len(grad), -1)
        grad_norm = grad.norm(2, dim=1)
        penalty = torch.mean((grad_norm - 1) ** 2)
        return penalty

    @staticmethod
    def get_gen_loss(critic_fake_pred: torch.Tensor):
        return -1 * torch.mean(critic_fake_pred)

    def get_critic_loss(self, critic_fake_pred: torch.Tensor, critic_real_pred: torch.Tensor,
                        grad_penalty: torch.Tensor):
        return torch.mean(critic_fake_pred) - torch.mean(critic_real_pred) + grad_penalty * self._c_lambda

    def generate_new_sample(self):
        noise = self.get_noise(n_samples=1)
        n_image = self._generator(noise)
        return n_image

    def train(self, train_dataloader, epochs: int, frequency: int = 5, valid_dataloader: Union[None, DataLoader] = None,
              image_save_path: Union[Path, None] = None):

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
                real = data[0].float().to(self._device)
                cur_batch_size = len(real)

                # critic update
                mean_iter_critic_loss = 0
                for _ in range(self._critic_repeat):
                    self._disc_opt.zero_grad()
                    fake_noise = self.get_noise(cur_batch_size)
                    fake = self._generator(fake_noise)
                    critic_fake_pred = self._discriminator(fake.detach())
                    critic_real_pred = self._discriminator(real)
                    epsilon = torch.rand(len(real), 1, 1, 1, device=self._device, requires_grad=True)
                    grad = self.get_gradient(fake=fake.detach(), real=real, epsilon=epsilon)
                    grad_penalty = self.gradient_penalty(grad=grad)
                    critic_loss = self.get_critic_loss(critic_fake_pred=critic_fake_pred,
                                                       critic_real_pred=critic_real_pred,
                                                       grad_penalty=grad_penalty)
                    mean_iter_critic_loss += critic_loss.item() / self._critic_repeat
                    critic_loss.backward(retain_graph=True)
                    self._disc_opt.step()
                disc_loss.append(mean_iter_critic_loss)

                # generator update
                self._gen_opt.zero_grad()
                fake_noise_2 = self.get_noise(cur_batch_size)
                fake_2 = self._generator(fake_noise_2)
                critic_fake_pred = self._discriminator(fake_2)

                g_loss = self.get_gen_loss(critic_fake_pred=critic_fake_pred)
                g_loss.backward()
                self._gen_opt.step()
                gen_loss.append(g_loss.item())

                # display
                if batch_idx % frequency == 0:
                    print('[TRAIN] => [%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                          % (epoch + 1, epochs, batch_idx + 1, len(train_dataloader), mean_iter_critic_loss,
                             g_loss.item()))

            glob_disc_loss.append(disc_loss)
            glob_gen_loss.append(gen_loss)

            # validation phase
            if valid_dataloader is not None:
                val_gen_loss = []
                val_disc_loss = []
                for val_batch_idx, data in enumerate(valid_dataloader):
                    val_real = data[0].float().to(self._device)

                    val_fake_noise = self.get_noise(len(val_real))
                    val_fake = self._generator(val_fake_noise.detach())

                    val_real_pred = self._discriminator(val_real)
                    val_fake_pred = self._discriminator(val_fake)

                    val_epsilon = torch.rand(len(val_real), 1, 1, 1, device=self._device, requires_grad=True)
                    val_grad = self.get_gradient(fake=val_fake.detach(), real=val_real, epsilon=val_epsilon)
                    val_grad_penalty = self.gradient_penalty(grad=val_grad)

                    val_critic_loss = self.get_critic_loss(critic_fake_pred=val_fake_pred,
                                                           critic_real_pred=val_real_pred,
                                                           grad_penalty=val_grad_penalty)
                    val_disc_loss.append(val_critic_loss.item())

                    val_g_loss = self.get_gen_loss(val_fake_pred)
                    val_gen_loss.append(val_g_loss)

                val_glob_disc_loss.append(val_disc_loss)
                val_glob_gen_loss.append(val_gen_loss)

            if image_save_path is not None:
                output = self.generate_new_sample()
                torchvision.utils.save_image(output, image_save_path.joinpath(f"image_{epoch + 1}.jpg"))

        return {"train_loss": [glob_disc_loss, glob_gen_loss],
                "valid_loss": [val_glob_disc_loss, val_glob_gen_loss],
                "has_valid": False if valid_dataloader is None else True,
                "epochs": epochs}

    def save_model(self, file_name: Path) -> None:
        """
        save path directory
        :param file_name:
        :return: None
        """

        torch.save(self._discriminator.state_dict, file_name.joinpath("w_discriminator.pt"))
        torch.save(self._generator.state_dict, file_name.joinpath("w_generator.pt"))

    @staticmethod
    def plot(res: dict, save_path: Path) -> None:
        weight_paths = save_path.joinpath("weights")
        weight_paths.mkdir(parents=True, exist_ok=True)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        x = np.arange(res.get("epochs")) + 1

        train_loss = res["train_loss"]
        valid_loss = res["valid_loss"]
        has_valid = res["has_valid"]

        t_gen = np.array(train_loss[1]).mean(axis=1)
        t_disc = np.array(train_loss[0]).mean(axis=1)

        np.save(str(weight_paths.joinpath("train_generator_loss.npy")), t_gen)
        np.save(str(weight_paths.joinpath("train_discriminator_loss.npy")), t_disc)

        if has_valid:
            v_gen = np.array(valid_loss[1]).mean(axis=1)
            v_disc = np.array(valid_loss[0]).mean(axis=1)

            np.save(str(weight_paths.joinpath("valid_generator_loss.npy")), v_gen)
            np.save(str(weight_paths.joinpath("valid_discriminator_loss.npy")), v_disc)

        plt.title("Without DP and Mixup")

        plt.plot(x, t_disc, color=colors[0], label="Train Disc Loss")
        plt.plot(x, t_gen, color=colors[1], label="Train Gen Loss")

        if has_valid:
            plt.plot(x, v_disc, color=colors[0], label="Valid Disc Loss", linestyle="-.")
            plt.plot(x, v_gen, color=colors[1], label="Valid Gen Loss", linestyle="-.")

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()

        plt.savefig(str(save_path.joinpath("loss.png")))