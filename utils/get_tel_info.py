import yaml
import argparse
from pathlib import Path

def main(args) -> dict:
    tel = args
    yaml_data = read_yaml_file()

    tel_info = yaml_data[tel.lower()]
    return tel_info

def read_yaml_file() -> dict:
    filepath_tel_info = Path(__file__).parent.parent / "tel_info.yaml"
    with open(filepath_tel_info, "r") as f:
        info = yaml.safe_load(f)
    return info

def get_args() -> tuple:
    parser = argparse.ArgumentParser(
        description="Program to obtain information about telescopes (ORT/GMRT)."
    )

    parser.add_argument(
        "tel", type=str,
        help="Telescope name. Current options are ORT/GMRT only."
    )

    args = parser.parse_args()
    tel = args.tel

    return tel

if __name__ == "__main__":
    args = get_args()
    tel_info = main(args)

    for k,v in tel_info.items():
        print(f"{k} = {v}")