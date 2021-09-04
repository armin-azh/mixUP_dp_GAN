from pathlib import Path


def save_parameters(args: dict, filename: Path) -> None:
    """
    save command parameter to txt file
    :param args: arguments
    :param filename: file path
    :return: None
    """
    with open(str(filename), "w") as file:
        for key, value in args.items():
            file.write(f"{key}\t{value}\n")
