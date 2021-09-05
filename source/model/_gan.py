from pathlib import Path
from typing import Tuple, Union
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from skimage import io

# arch
# from ._arch import SimpleGenerator, SimpleDiscriminator
from ._arch import Generator, Critic


class WGan:
    def __init__(self, image_size: Tuple[int, int], latent_dim: int, beta1: float, lr: float, image_channel: int,
                 gen_features: int, disc_features: int, critic_repeat: int, c_lambda: int, alpha: float,
                 clip: float, sigma: float, batch_size: int, log_dir: Path, tensorboard: bool, device: torch.device):
        """
        use wasserstein distance to learn the discriminator and generator
        :param image_size: (w,h) of the image
        :param latent_dim: number of latent dimension
        :param beta1: adam optimizer parameter
        :param lr: learning rate
        :param image_channel: number of image channel
        :param gen_features: generator hidden feature maps
        :param disc_features: discriminator hidden feature maps
        :param critic_repeat: number of critic iteration ( number of updating discriminator)
        :param c_lambda: critic penalty scale
        :param alpha: mix - up parameter
        :param clip: gradient clip scale
        :param sigma: differential privacy noise scale
        :param batch_size: batch size
        :param log_dir: log directory for tensorboard result
        :param tensorboard: enable tensorboard
        :param device: use cuda or cpu
        """
        self._latent_dim = latent_dim
        self._image_size = image_size
        self._betas = (beta1, 0.999)
        self._learning_rate = lr
        self._image_channel = image_channel
        self._gen_features = gen_features
        self._disc_features = disc_features
        self._critic_repeat = critic_repeat
        self._c_lambda = c_lambda
        self._alpha = alpha
        self._clip_weight = clip
        self._sigma = sigma
        self._batch_size = batch_size
        self._device = device
        self._log_dir = log_dir
        self._has_tensorboard = tensorboard
        self._writer = SummaryWriter(log_dir=str(self._log_dir))

        # define generator
        self._generator = Generator(latent_dim=self._latent_dim, image_channel=self._image_channel,
                                    features=self._gen_features).to(self._device)
        self._generator.apply(self.weights_init)  # initiate weight parameter

        # define discriminator
        self._discriminator = Critic(image_channel=self._image_channel, features=self._disc_features).to(self._device)
        self._discriminator.apply(self.weights_init)  # initiate weight parameter

        # introduce optimizers
        self._gen_opt = torch.optim.Adam(self._generator.parameters(), lr=self._learning_rate, betas=self._betas)
        self._disc_opt = torch.optim.Adam(self._discriminator.parameters(), lr=self._learning_rate, betas=self._betas)

        self._title = None  # final plot title

    def get_noise(self, n_samples: int):
        """
        generate noise samples
        :param n_samples: number of batches
        :return: tensor in shape (b,latent_dim)
        """
        return torch.randn(n_samples, self._latent_dim, device=self._device)

    @staticmethod
    def weights_init(m):
        """
        a callback function for initialize the parameters
        :param m: layer
        :return:
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def get_gradient(self, real: torch.Tensor, fake: torch.Tensor, epsilon):
        """
        calculate gradient
        :param real: real image samples ~ p(x)
        :param fake: generated image samples ~ p(z)
        :param epsilon: mix up sample weight
        :return: tensor in shape (b,m)
        """
        mixed_images = real * epsilon + fake * (1 - epsilon)
        mixed_scores = self._discriminator(mixed_images)
        grad = torch.autograd.grad(inputs=mixed_images, outputs=mixed_scores, grad_outputs=torch.ones_like(mixed_scores)
                                   , create_graph=True, retain_graph=True)[0]
        return grad

    @staticmethod
    def gradient_penalty(grad: torch.Tensor):
        """
        calculate gradient penalty
        :param grad: tensor in shape (b,m)
        :return:
        """
        grad = grad.view(len(grad), -1)
        grad_norm = grad.norm(2, dim=1)
        penalty = torch.mean((grad_norm - 1) ** 2)
        return penalty

    @staticmethod
    def get_gen_loss(critic_fake_pred: torch.Tensor):
        """
        calculate generator loss
        :param critic_fake_pred: D(G(z))
        :return:
        """
        return -1 * torch.mean(critic_fake_pred)

    def get_critic_loss(self, critic_fake_pred: torch.Tensor, critic_real_pred: torch.Tensor,
                        grad_penalty: torch.Tensor):
        """
        calculate discriminator loss
        :param critic_fake_pred: D(G(z))
        :param critic_real_pred: D(x)
        :param grad_penalty:
        :return:
        """
        return torch.mean(critic_fake_pred) - torch.mean(critic_real_pred) + grad_penalty * self._c_lambda

    def generate_new_sample(self) -> torch.Tensor:
        """
        generate new sample from p(z)
        :return:
        """
        noise = self.get_noise(n_samples=1)
        n_image = self._generator(noise)
        return n_image

    def get_lambda(self) -> float:
        """
        generate new lambda value for mix up
        :return:
        """
        return np.random.beta(self._alpha, self._alpha)

    def mix_data(self, real1: torch.Tensor, real2: torch.Tensor) -> torch.Tensor:
        """
        mix the 2 real batches together
        :param real1: tensor in shape (b,1,w,h)
        :param real2: tensor in shape (b,1,w,h)
        :return: tensor in shape (b,1,w,h)
        """
        _perm1 = torch.randperm(real1.size()[0], device=self._device)
        _perm2 = torch.randperm(real2.size()[0], device=self._device)
        _lam = self.get_lambda()
        mix_real = _lam * real1[_perm1] + (1 - _lam) * real2[_perm2]
        return mix_real

    @staticmethod
    def create_title(has_mix_up: bool, has_dp: bool) -> str:
        """
        create title for plot
        :return:
        """
        pre_fix = "W-GAN "
        if not has_dp and not has_mix_up:
            return pre_fix + "Without DP and Mix-up"
        if has_dp:
            pre_fix += "With DP "
        if has_mix_up:
            pre_fix += "and Mix-up"
        return pre_fix

    def train(self, train_dataloader: DataLoader,
              epochs: int, frequency: int = 5, valid_dataloader: Union[None, DataLoader] = None,
              image_save_path: Union[Path, None] = None,
              train_dataloader_2: Union[None, DataLoader] = None,
              dp: bool = False) -> dict:
        """

        :param train_dataloader: training data
        :param epochs: number of epochs
        :param frequency: show the result
        :param valid_dataloader: validation data
        :param image_save_path: path to save generated samples
        :param train_dataloader_2: training data ( if mix-up enabled)
        :param dp: enable differential privacy
        :return: result
        """

        global_step = 1

        # some variable to show result
        glob_gen_loss = []
        glob_disc_loss = []

        val_glob_gen_loss = []
        val_glob_disc_loss = []

        # determine if mix-up enable or not
        has_train_loader = False if train_dataloader_2 is None else True

        if train_dataloader_2 is None:
            train_dataloader_2 = [None] * len(train_dataloader)

        # add noise to the gradients if dp is enabled
        if dp:
            for p in self._discriminator.parameters():
                p.register_hook(
                    lambda grad_: grad_ + (1 / self._batch_size) * torch.normal(mean=0, std=self._sigma, size=p.shape,
                                                                                device=self._device))

        # start training process
        print("[READY] training is now starting ...")
        for epoch in range(epochs):
            gen_loss = []
            disc_loss = []

            # train phase
            for batch_idx, data in enumerate(zip(train_dataloader, train_dataloader_2)):

                # prepare real sample whether mix-up is enabled or not
                if has_train_loader is False:
                    (real1, label1), *other = data
                    real = real1.float().to(self._device)
                    cur_batch_size = len(real)
                elif has_train_loader is True:
                    (real1, label1), (real2, label2) = data
                    real1 = real1.float().to(self._device)
                    real2 = real2.float().to(self._device)
                    real = self.mix_data(real1=real1, real2=real2)
                    cur_batch_size = len(real)
                else:
                    raise RuntimeError("This option had not provided ...")

                # critic update
                mean_iter_critic_loss = 0
                for _ in range(self._critic_repeat):
                    self._disc_opt.zero_grad()

                    # generate fake sample
                    fake_noise = self.get_noise(cur_batch_size)
                    fake = self._generator(fake_noise)

                    # passing through the network
                    critic_fake_pred = self._discriminator(fake.detach())
                    critic_real_pred = self._discriminator(real)

                    # compute the gradients
                    epsilon = torch.rand(len(real), 1, 1, 1, device=self._device, requires_grad=True)
                    grad = self.get_gradient(fake=fake.detach(), real=real, epsilon=epsilon)
                    grad_penalty = self.gradient_penalty(grad=grad)
                    critic_loss = self.get_critic_loss(critic_fake_pred=critic_fake_pred,
                                                       critic_real_pred=critic_real_pred,
                                                       grad_penalty=grad_penalty)
                    mean_iter_critic_loss += critic_loss.item() / self._critic_repeat

                    # assign gradient
                    critic_loss.backward(retain_graph=True)
                    self._disc_opt.step()

                    # clip the parameters
                    for p in self._discriminator.parameters():
                        p.data.clamp_(-self._clip_weight, self._clip_weight)

                disc_loss.append(mean_iter_critic_loss)

                # update generator
                self._gen_opt.zero_grad()

                # generate new sample
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

                if self._has_tensorboard:
                    output = self.generate_new_sample()
                    output = torch.squeeze(output, dim=0)
                    self._writer.add_image("Image on every step", output, global_step)

                if image_save_path is not None:
                    output = self.generate_new_sample()
                    torchvision.utils.save_image(output,
                                                 image_save_path.joinpath(f"image_epoch({epoch + 1})_{global_step}.jpg")
                                                 )

                    # next glob step
                global_step += 1

            glob_disc_loss.append(disc_loss)
            glob_gen_loss.append(gen_loss)

            if self._has_tensorboard:
                self._writer.add_scalars("Loss", {"Discriminator": np.mean(disc_loss),
                                                  "Generator": np.mean(gen_loss)}, epoch)

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

        self._title = self.create_title(has_mix_up=has_train_loader, has_dp=dp)

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

        torch.save(self._discriminator.state_dict(), file_name.joinpath("w_discriminator.pth"))
        torch.save(self._generator.state_dict(), file_name.joinpath("w_generator.pth"))

    def plot(self, res: dict, save_path: Path) -> None:
        """
        render and create plot
        :param res: training result
        :param save_path: path to save
        :return: none
        """
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

        plt.title(self._title)

        plt.plot(x, t_disc, color=colors[0], label="Train Disc Loss")
        plt.plot(x, t_gen, color=colors[1], label="Train Gen Loss")

        if has_valid:
            plt.plot(x, v_disc, color=colors[0], label="Valid Disc Loss", linestyle="-.")
            plt.plot(x, v_gen, color=colors[1], label="Valid Gen Loss", linestyle="-.")

        plt.hlines(y=0, xmin=0, xmax=res.get("epochs"), colors=colors[2], linestyles="dashed")

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()

        plt.savefig(str(save_path.joinpath("loss.png")))

        if self._has_tensorboard:
            _im = io.imread(str(save_path.joinpath("loss.png")))
            _im = ToTensor()(_im)
            self._writer.add_image("Final Result", _im, 1)
