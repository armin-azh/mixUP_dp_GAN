import argparse
from pathlib import Path
import random
import torch

# tools
from src.tools import convert_binary_to_image

# model
from src.trainer import AutoEncoderCV1Container
from src.loader import get_zero_dataloader
from src.model import DCGAN

operations = {
    "conv_bin_to_im": "cvt_bin_im",
    "auto_encoder": "train_ae",
    "dc_gan": "train_dc_gan"
}


def main(arguments: argparse.Namespace) -> None:
    """
    main function for run out command
    :param arguments: command line namespace
    :return: None
    """
    # determine seed
    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)

    if arguments.op == operations.get("conv_bin_to_im"):
        in_dir = Path(arguments.input)
        out_dir = Path(arguments.out)
        convert_binary_to_image(input_dir=in_dir, output_dir=out_dir)

    elif arguments.op == operations.get("auto_encoder"):
        im_path = Path(arguments.input)
        im_path = im_path.absolute()
        label_path = Path(arguments.input_lb)
        save_path = Path(arguments.out)
        label_path = label_path.absolute()
        train_loader, valid_loader = get_zero_dataloader(im_path, label_path, arguments.train_size, arguments.test_size,
                                                         arguments.shuffle, arguments.seed, arguments.batch,
                                                         arguments.num_worker, (arguments.width, arguments.height))
        model = AutoEncoderCV1Container(train_dataloader=train_loader,
                                        validation_dataloader=valid_loader,
                                        im_channel=arguments.channel,
                                        lr=arguments.lr,
                                        weight_decay=arguments.weight_decay,
                                        epochs=arguments.epochs,
                                        save_path=save_path)

        model.fit()

    elif arguments.op == operations.get("dc_gan"):
        im_path = Path(arguments.input)
        im_path = im_path.absolute()
        label_path = Path(arguments.input_lb)

        # paths
        save_path = Path(arguments.out)
        save_path = save_path.joinpath("dc_gan")
        save_path.mkdir(parents=True, exist_ok=True)
        plot_save_path = save_path.joinpath("plot")
        plot_save_path.mkdir(parents=True, exist_ok=True)
        images_save_path = save_path.joinpath("images")
        images_save_path.mkdir(parents=True, exist_ok=True)

        train_loader, valid_loader = get_zero_dataloader(im_path, label_path, arguments.train_size, arguments.test_size,
                                                         arguments.shuffle, arguments.seed, arguments.batch,
                                                         arguments.num_worker, (arguments.width, arguments.height))

        device = torch.device("cuda:0" if (torch.cuda.is_available() and arguments.device > 0) else "cpu")

        model = DCGAN(beta1=arguments.beta1,
                      feature_maps_gen=arguments.gen_feature_map,
                      feature_maps_disc=arguments.disc_feature_map,
                      image_channels=arguments.channel,
                      latent_dim=arguments.latent_dim,
                      lr=arguments.lr,
                      device=arguments.device).to(device)

        res = model.fit(train_dataloader=train_loader, epochs=arguments.epochs, valid_dataloader=valid_loader)
        print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input path", type=str, default="/home/lezarus/Documents/Project/mixUpGAN"
                                                                        "/data/zero_day/images")
    parser.add_argument("--input_lb", help="label csv file", type=str, default="/home/lezarus/Documents/Project"
                                                                               "/mixUpGAN/data/zero_day/trainLabels.csv")
    parser.add_argument("--out", help="output path", type=str)
    parser.add_argument("--op", help="choose the operation", type=str, choices=list(operations.values()))
    parser.add_argument("--weight_decay", help="determine the weight decay", type=float, default=1e-5)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--seed", help="random seed", type=float, default=999)
    parser.add_argument("--epochs", help="number of epochs", type=int, default=5)
    parser.add_argument("--shuffle", help="enable shuffling", type=bool, default=True)
    parser.add_argument("--batch", help="batch size", type=int, default=64)
    parser.add_argument("--num_worker", help="number of workers", type=int, default=0)
    parser.add_argument("--train_size", help="train size", type=float, default=.7)
    parser.add_argument("--test_size", help="test size", type=float, default=.3)
    parser.add_argument("--width", help="set image width", type=int, default=100)
    parser.add_argument("--height", help="set image height", type=int, default=135)
    parser.add_argument("--channel", help="image channel", type=int, default=1)
    parser.add_argument("--beta1", help="adam hyper parameter", type=float, default=0.5)
    parser.add_argument("--gen_feature_map", help="generator feature map", type=int, default=64)
    parser.add_argument("--disc_feature_map", help="discriminator feature map", type=int, default=64),
    parser.add_argument("--latent_dim", help="latent dimension", type=int, default=100)
    parser.add_argument("--device", help="use cuda device", type=int, default=1)
    args = parser.parse_args()
    main(args)
