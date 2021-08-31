import argparse
from pathlib import Path

# tools
from .tools import convert_binary_to_image

operations = {
    "conv_bin_to_im": "cvt_bin_im"
}


def main(arguments: argparse.Namespace) -> None:
    """
    main function for run out command
    :param arguments: command line namespace
    :return: None
    """
    if arguments.op == operations.get("conv_bin_to_im"):
        in_dir = Path(arguments.input_dir)
        out_dir = Path(arguments.ouput_dir)
        convert_binary_to_image(input_dir=in_dir, output_dir=out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="input directory", type=str)
    parser.add_argument("--output_dir", help="output directory", type=str)
    parser.add_argument("--op", help="choose the operation", type=str, choices=list(operations.values()))
    args = parser.parse_args()
    main(args)
