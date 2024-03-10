import argparse
from pathlib import Path

import numpy as np


def main(path_raw_dataset_dir: Path):
    # import original datasets
    red = np.genfromtxt(
        path_raw_dataset_dir / "winequality-red.csv", delimiter=";", skip_header=True
    )
    white = np.genfromtxt(
        path_raw_dataset_dir / "winequality-white.csv", delimiter=";", skip_header=True
    )
    x = np.concatenate((red, white), axis=0)

    # add covariate for red vs. white
    red_color = np.array([1] * red.shape[0] + [0] * white.shape[0], dtype=np.float64).reshape(
        (-1, 1)
    )
    x = np.concatenate((x, red_color), axis=1)

    # extract outcome and covariates
    y = x[:, -2].copy().reshape((-1, 1))
    x = np.delete(x, -2, 1)
    categorical = [x.shape[1]]

    np.save(
        path_raw_dataset_dir.parent / "data.npy", {"x": x, "y": y, "categorical": categorical}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    main(args.path)
