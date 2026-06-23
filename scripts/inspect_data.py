import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.uwgi.dataset import build_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/uw-madison-gi-tract-image-segmentation")
    args = parser.parse_args()

    root = Path(args.root)
    meta = build_metadata(root)
    labels = pd.read_csv(root / "train.csv")
    print("Images:", len(meta))
    print("Cases:", meta["case"].nunique())
    print("Positive slices:", int(meta["has_mask"].sum()))
    print("Class rows:")
    print(labels.groupby("class")["segmentation"].apply(lambda s: s.notna().sum()))
    print("Folds:")
    print(meta.groupby("fold").agg(images=("id", "count"), cases=("case", "nunique"), positives=("has_mask", "sum")))


if __name__ == "__main__":
    main()
