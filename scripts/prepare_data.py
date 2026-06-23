import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd


COMPETITION = "uw-madison-gi-tract-image-segmentation"


def safe_extract_member(zf: zipfile.ZipFile, member: str | zipfile.ZipInfo, destination: Path) -> Path:
    info = zf.getinfo(member) if isinstance(member, str) else member
    destination = destination.resolve()
    target = (destination / info.filename).resolve()
    if os.path.commonpath([destination, target]) != str(destination):
        raise ValueError(f"Unsafe zip member path: {info.filename}")
    if info.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(info) as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


def extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            safe_extract_member(zf, info, destination)


def extract_sample_zip(zip_path: Path, destination: Path, sample_cases: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        train_csv_name = next((name for name in names if name.endswith("train.csv")), None)
        if train_csv_name is None:
            raise FileNotFoundError("train.csv was not found inside the Kaggle zip.")

        with zf.open(train_csv_name) as f:
            labels = pd.read_csv(f)
        labels["case"] = labels["id"].str.extract(r"^(case\d+)_")
        selected_cases = labels["case"].drop_duplicates().head(sample_cases).tolist()
        selected = labels[labels["case"].isin(selected_cases)].drop(columns=["case"])
        selected.to_csv(destination / "train.csv", index=False)

        sample_submission = next((name for name in names if name.endswith("sample_submission.csv")), None)
        if sample_submission:
            extracted = safe_extract_member(zf, sample_submission, destination)
            if extracted != destination / "sample_submission.csv":
                extracted.replace(destination / "sample_submission.csv")

        prefixes = tuple(f"train/{case}/" for case in selected_cases)
        extracted_count = 0
        for name in names:
            normalized = name.replace("\\", "/")
            if normalized.startswith(prefixes) and not normalized.endswith("/"):
                safe_extract_member(zf, name, destination)
                extracted_count += 1

        if extracted_count == 0:
            raise FileNotFoundError("No train images were extracted. Zip layout may differ from expected Kaggle layout.")
        print(f"Extracted {len(selected_cases)} cases and {extracted_count} image files.")


def download_with_kaggle(destination: Path) -> Path:
    project_kaggle_dir = Path.cwd() / ".kaggle"
    project_kaggle_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(project_kaggle_dir.resolve()))
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(destination)],
        check=True,
    )
    zip_path = destination / f"{COMPETITION}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    return zip_path


def validate_data(root: Path) -> None:
    required = [root / "train.csv", root / "train"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected files after extraction: {missing}")
    print(f"Data ready: {root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, help="Path to Kaggle competition zip.")
    parser.add_argument("--download", action="store_true", help="Download through Kaggle CLI.")
    parser.add_argument("--sample-cases", type=int, help="Extract only the first N cases from the Kaggle zip.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw") / COMPETITION,
        help="Extraction destination.",
    )
    args = parser.parse_args()

    if args.download:
        zip_path = download_with_kaggle(args.out.parent)
    elif args.zip:
        zip_path = args.zip
    else:
        candidates = [
            Path.home() / "Downloads" / f"{COMPETITION}.zip",
            Path("data/raw") / f"{COMPETITION}.zip",
        ]
        found = [path for path in candidates if path.exists()]
        if not found:
            raise FileNotFoundError(
                "No zip found. Pass --zip or place the Kaggle zip in Downloads/data/raw."
            )
        zip_path = found[0]

    print(f"Extracting {zip_path} -> {args.out}")
    if args.sample_cases:
        extract_sample_zip(zip_path, args.out, args.sample_cases)
    else:
        extract_zip(zip_path, args.out)
    validate_data(args.out)


if __name__ == "__main__":
    main()
