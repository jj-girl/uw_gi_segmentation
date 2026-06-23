import argparse
from pathlib import Path

import yaml


def with_fold_suffix(output_dir: str, fold: int) -> str:
    path = Path(output_dir)
    name = path.name
    if name.endswith("_fold0"):
        name = name[: -len("_fold0")]
    elif "_fold" in name:
        name = name.rsplit("_fold", 1)[0]
    return str(path.with_name(f"{name}_fold{fold}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base YAML config with fold0 values.")
    parser.add_argument("--out-dir", default="configs/nnunet_route_folds")
    parser.add_argument("--folds", type=int, default=None, help="Override data.num_folds.")
    args = parser.parse_args()

    base_path = Path(args.base)
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    num_folds = int(args.folds or cfg["data"].get("num_folds", 5))
    cfg["data"]["num_folds"] = num_folds

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fold in range(num_folds):
        fold_cfg = yaml.safe_load(yaml.safe_dump(cfg, sort_keys=False))
        fold_cfg["data"]["valid_fold"] = fold
        fold_cfg["train"]["output_dir"] = with_fold_suffix(cfg["train"]["output_dir"], fold)
        out_path = out_dir / f"{base_path.stem}_fold{fold}.yaml"
        out_path.write_text(yaml.safe_dump(fold_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        written.append(out_path)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
