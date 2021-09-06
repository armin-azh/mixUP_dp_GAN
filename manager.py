import argparse
from pathlib import Path
import random
from datetime import datetime
import torch
import numpy as np
import os

# tools
from source.loader import get_zero_dataloader

# model
from source.model import WGan
from source.model import Classifier

from source.tools import save_parameters

operations = {
    "w_gan": "train_w_gan",
    "classifier": "train_classifier"
}


def main(arguments: argparse.Namespace) -> None:
    """
    main function for run out command
    :param arguments: command line namespace
    :return: None
    """
    # determine seed
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    log_dir = base_dir.joinpath("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and arguments.device > 0) else "cpu")

    if arguments.op == operations.get("w_gan"):
        im_path = Path(arguments.input)
        im_path = im_path.absolute()
        label_path = Path(arguments.input_lb)

        save_path = Path(arguments.out)
        _cu = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        save_path = save_path.joinpath("w_gan").joinpath(_cu)
        save_path.mkdir(parents=True, exist_ok=True)

        plot_save_path = save_path.joinpath("plot")
        plot_save_path.mkdir(parents=True, exist_ok=True)
        images_save_path = save_path.joinpath("images")
        images_save_path.mkdir(parents=True, exist_ok=True)
        model_save_path = save_path.joinpath("model")
        model_save_path.mkdir(parents=True, exist_ok=True)

        train_loader_2 = None
        train_loader, _ = get_zero_dataloader(im_path, label_path, arguments.train_size, arguments.test_size,
                                              arguments.shuffle, arguments.seed, arguments.batch,
                                              arguments.num_worker, (arguments.width, arguments.height))

        if arguments.mix_up:
            train_loader_2, _ = get_zero_dataloader(im_path, label_path, arguments.train_size, arguments.test_size,
                                                    arguments.shuffle, arguments.seed, arguments.batch,
                                                    arguments.num_worker, (arguments.width, arguments.height))

        save_parameters(vars(arguments), save_path.joinpath("parameters.txt"))

        model = WGan(image_size=(arguments.width, arguments.height),
                     latent_dim=arguments.latent_dim,
                     beta1=arguments.beta1,
                     lr=arguments.lr,
                     image_channel=arguments.channel,
                     gen_features=arguments.gen_feature_map,
                     disc_features=arguments.disc_feature_map,
                     critic_repeat=arguments.c_repeat,
                     c_lambda=arguments.c_lambda,
                     alpha=arguments.alpha,
                     clip=arguments.clip,
                     sigma=arguments.sigma,
                     batch_size=arguments.batch,
                     device=device,
                     log_dir=log_dir.joinpath("GAN").joinpath(_cu),
                     tensorboard=arguments.tensorboard)

        res = model.train(train_dataloader=train_loader,
                          epochs=arguments.epochs,
                          frequency=arguments.show_rate,
                          valid_dataloader=None,
                          image_save_path=images_save_path,
                          train_dataloader_2=train_loader_2,
                          dp=arguments.dp)

        model.save_model(file_name=model_save_path)
        model.plot(res=res, save_path=plot_save_path)

    elif arguments.op == operations.get("classifier"):
        im_path = Path(arguments.input)
        im_path = im_path.absolute()
        label_path = Path(arguments.input_lb)

        save_path = Path(arguments.out)
        _cu = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        save_path = save_path.joinpath("classifier").joinpath(_cu)
        save_path.mkdir(parents=True, exist_ok=True)

        plot_save_path = save_path.joinpath("plot")
        plot_save_path.mkdir(parents=True, exist_ok=True)
        model_save_path = save_path.joinpath("model")
        model_save_path.mkdir(parents=True, exist_ok=True)

        train_loader, valid_loader = get_zero_dataloader(im_path, label_path, arguments.train_size, arguments.test_size,
                                                         arguments.shuffle, arguments.seed, arguments.batch,
                                                         arguments.num_worker, (arguments.width, arguments.height))

        state_dic = None
        if arguments.pretrained != "":
            state_dic = torch.load(arguments.pretrained)

        save_parameters(vars(arguments), save_path.joinpath("parameters.txt"))

        model = Classifier(lr=arguments.lr,
                           image_channel=arguments.channel,
                           feature_maps=arguments.disc_feature_map,
                           tensorboard=arguments.tensorboard,
                           log_dir=log_dir.joinpath("classifier").joinpath(_cu),
                           state_dict=state_dic,
                           device=device,
                           weight_decay=arguments.weight_decay,
                           classes=9)

        res = model.train(train_dataloader=train_loader, valid_dataloader=valid_loader, epochs=arguments.epochs)

        model.save_model(file_name=model_save_path)
        model.plot(res=res, save_path=plot_save_path, posix=arguments.posix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", help="input path", type=str, default="/home/lezarus/Documents/Project/mixUpGAN"
                                                                        "/data/zero_day/images")
    parser.add_argument("--input_lb", help="label csv file", type=str, default="/home/lezarus/Documents/Project"
                                                                               "/mixUpGAN/data/zero_day/trainLabels.csv")
    parser.add_argument("--out", help="output path", type=str)
    parser.add_argument("--pretrained", help="pretrain model path", type=str, default="")
    parser.add_argument("--op", help="choose the operation", type=str, choices=list(operations.values()))
    parser.add_argument("--seed", help="random seed", type=float, default=999)

    # loader parameter
    parser.add_argument("--num_worker", help="number of workers", type=int, default=0)
    parser.add_argument("--shuffle", help="enable shuffling", type=bool, default=True)
    parser.add_argument("--train_size", help="train size", type=float, default=.7)
    parser.add_argument("--test_size", help="test size", type=float, default=.3)

    # network parameters
    parser.add_argument("--weight_decay", help="determine the weight decay", type=float, default=1e-5)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--width", help="set image width", type=int, default=65)
    parser.add_argument("--height", help="set image height", type=int, default=130)
    parser.add_argument("--channel", help="image channel", type=int, default=1)
    parser.add_argument("--beta1", help="adam hyper parameter", type=float, default=0.5)
    parser.add_argument("--gen_feature_map", help="generator feature map", type=int, default=64)
    parser.add_argument("--disc_feature_map", help="discriminator feature map", type=int, default=64),

    # train phase parameter
    parser.add_argument("--epochs", help="number of epochs", type=int, default=20)
    parser.add_argument("--batch", help="batch size", type=int, default=64)
    parser.add_argument("--clip", help="clip weight for (differential privacy)", type=float, default=0.01)
    parser.add_argument("--c_repeat", help="critic repeat for (differential privacy)", type=int, default=5)
    parser.add_argument("--c_lambda", help="critic lambda", type=int, default=10)
    parser.add_argument("--sigma", help="noise scale coefficient", type=float, default=12.)
    parser.add_argument("--latent_dim", help="latent dimension", type=int, default=100)
    parser.add_argument("--show_rate", help="show status is specific rate", type=int, default=5)
    parser.add_argument("--device", help="use cuda device", type=int, default=1)
    parser.add_argument("--alpha", help="mixup coefficient", type=float, default=1)

    # mode
    parser.add_argument("--mix_up", help="enable training with mix_up", action="store_true")
    parser.add_argument("--dp", help="enable differential privacy on learning process", action="store_true")
    parser.add_argument("--tensorboard", help="enable tensorboard to show the results", action="store_true")

    parser.add_argument("--posix", help="title for plot header on classifier", type=str, default="")

    args = parser.parse_args()
    main(args)
