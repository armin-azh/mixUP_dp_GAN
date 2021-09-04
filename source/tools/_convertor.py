from pathlib import Path
from PIL import Image


def convert_binary_to_image(input_dir: Path, output_dir: Path) -> None:
    """
    convert binary code to image
    :param input_dir: binary files root directory
    :param output_dir: save path directory
    :return: None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.exists() or input_dir.is_file():
        raise ValueError(f"{str(input_dir)} is file or not exists")


