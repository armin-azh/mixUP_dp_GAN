import argparse
from pathlib import Path

# tools
from .tools import convert_binary_to_image

# model
from src.trainer import AutoEncoderCV1Container
from src.loader import get_zero_dataset

operations = {
    "conv_bin_to_im": "cvt_bin_im",
    "auto_encoder": "train_ae"
}


def main(arguments: argparse.Namespace) -> None:
    """
    main function for run out command
    :param arguments: command line namespace
    :return: None
    """
    if arguments.op == operations.get("conv_bin_to_im"):
        in_dir = Path(arguments.input)
        out_dir = Path(arguments.out)
        convert_binary_to_image(input_dir=in_dir, output_dir=out_dir)

    elif arguments.op == operations.get("auto_encoder"):
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input path", type=str, default="data/zero_day/images")
    parser.add_argument("--input_lb", help="label csv file", type=str, default="data/zero_day/trainLabels.csv")
    parser.add_argument("--out", help="output path", type=str)
    parser.add_argument("--op", help="choose the operation", type=str, choices=list(operations.values()))
    parser.add_argument("--weight_decay", help="determine the weight decay", type=float, default=1e-5)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--seed", help="random seed", type=float, default=999)
    parser.add_argument("--epochs", help="number of epochs", type=int, default=10)
    parser.add_argument("--shuffle", help="enable shuffling", type=bool, default=True)
    args = parser.parse_args()
    main(args)
