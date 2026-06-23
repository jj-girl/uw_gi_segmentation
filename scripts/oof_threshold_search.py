import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.threshold_search import collect_predictions
from src.uwgi.dataset import CLASSES
from src.uwgi.utils import ensure_dir, get_device, load_yaml


def parse_grid(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x]


def candidate_key(mask_thr: float, cls_thr: float | None) -> tuple[float, float | None]:
    return float(mask_thr), None if cls_thr is None else float(cls_thr)


def update_scores(
    scores: dict[str, dict[tuple[float, float | None], float]],
    counts: dict[str, dict[tuple[float, float | None], int]],
    probs: np.ndarray,
    targets: np.ndarray,
    cls_probs: np.ndarray | None,
    mask_grid: list[float],
    cls_grid: list[float],
) -> None:
    for channel, cls_name in enumerate(CLASSES):
        target = targets[:, channel] > 0.5
        target_sum = target.reshape(target.shape[0], -1).sum(axis=1)
        candidate_cls_grid = cls_grid if cls_probs is not None else [None]
        for mask_thr in mask_grid:
            pred = probs[:, channel] > mask_thr
            pred_sum = pred.reshape(pred.shape[0], -1).sum(axis=1)
            intersection = (pred & target).reshape(pred.shape[0], -1).sum(axis=1)
            for cls_thr in candidate_cls_grid:
                key = candidate_key(mask_thr, cls_thr)
                if cls_thr is None:
                    gated_pred_sum = pred_sum
                    gated_intersection = intersection
                else:
                    keep = cls_probs[:, channel] >= cls_thr
                    gated_pred_sum = pred_sum * keep
                    gated_intersection = intersection * keep
                denom = gated_pred_sum + target_sum
                dice = np.where(denom > 0, (2.0 * gated_intersection + 1e-7) / (denom + 1e-7), 1.0)
                scores[cls_name][key] = scores[cls_name].get(key, 0.0) + float(dice.sum())
                counts[cls_name][key] = counts[cls_name].get(key, 0) + int(dice.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-config-glob", required=True)
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mask-grid", default="0.25,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--cls-grid", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--disable-cls-gate", action="store_true")
    args = parser.parse_args()

    config_paths = sorted(Path(path) for path in glob.glob(args.fold_config_glob))
    if not config_paths:
        raise FileNotFoundError(f"No configs matched {args.fold_config_glob}")

    mask_grid = parse_grid(args.mask_grid)
    cls_grid = parse_grid(args.cls_grid)
    scores: dict[str, dict[tuple[float, float | None], float]] = {name: {} for name in CLASSES}
    counts: dict[str, dict[tuple[float, float | None], int]] = {name: {} for name in CLASSES}

    for config_path in config_paths:
        cfg = load_yaml(config_path)
        checkpoint = Path(cfg["train"]["output_dir"]) / args.checkpoint_name
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for {config_path}: {checkpoint}")
        device = get_device(cfg["train"]["device"])
        probs, targets, cls_probs = collect_predictions(cfg, checkpoint, device)
        if args.disable_cls_gate:
            cls_probs = None
        update_scores(scores, counts, probs, targets, cls_probs, mask_grid, cls_grid)

    results = {}
    for cls_name in CLASSES:
        best = {"dice": -1.0, "mask_threshold": 0.5, "cls_threshold": None}
        for key, score_sum in scores[cls_name].items():
            count = counts[cls_name][key]
            dice = score_sum / max(count, 1)
            if dice > best["dice"]:
                mask_thr, cls_thr = key
                best = {"dice": float(dice), "mask_threshold": float(mask_thr), "cls_threshold": cls_thr}
        results[cls_name] = best
    results["mean_dice"] = float(np.mean([results[name]["dice"] for name in CLASSES]))
    results["num_folds"] = len(config_paths)

    out = Path(args.out)
    ensure_dir(out.parent)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved OOF thresholds: {out}")


if __name__ == "__main__":
    main()
