import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.threshold_search import collect_predictions, dice_numpy
from src.uwgi.dataset import CLASSES
from src.uwgi.postprocess import postprocess_slice
from src.uwgi.utils import ensure_dir, get_device, load_yaml


def evaluate(probs: np.ndarray, targets: np.ndarray, cls_probs: np.ndarray | None, cfg: dict) -> dict:
    post_cfg = cfg.get("postprocess", {})
    mask_thresholds = post_cfg.get("mask_thresholds", 0.5)
    cls_thresholds = post_cfg.get("cls_thresholds", 0.5)
    min_area = post_cfg.get("min_area", 0)
    class_scores = {name: [] for name in CLASSES}
    class_positive_scores = {name: [] for name in CLASSES}
    class_empty_scores = {name: [] for name in CLASSES}
    class_presence = {
        name: {
            "target_positive": 0,
            "target_empty": 0,
            "pred_positive_on_target_positive": 0,
            "pred_positive_on_target_empty": 0,
        }
        for name in CLASSES
    }
    for idx in range(probs.shape[0]):
        pred = postprocess_slice(
            probs[idx],
            cls_probs=cls_probs[idx] if cls_probs is not None else None,
            mask_thresholds=mask_thresholds,
            cls_thresholds=cls_thresholds,
            min_area=min_area,
        )
        for channel, name in enumerate(CLASSES):
            pred_mask = pred[channel]
            target_mask = targets[idx, channel]
            score = dice_numpy(pred_mask, target_mask)
            class_scores[name].append(score)
            target_has_mask = bool(target_mask.sum() > 0)
            pred_has_mask = bool(pred_mask.sum() > 0)
            if target_has_mask:
                class_positive_scores[name].append(score)
                class_presence[name]["target_positive"] += 1
                class_presence[name]["pred_positive_on_target_positive"] += int(pred_has_mask)
            else:
                class_empty_scores[name].append(score)
                class_presence[name]["target_empty"] += 1
                class_presence[name]["pred_positive_on_target_empty"] += int(pred_has_mask)

    results = {"classes": {}, "summary": {}}
    for name in CLASSES:
        presence = class_presence[name]
        target_positive = presence["target_positive"]
        target_empty = presence["target_empty"]
        results["classes"][name] = {
            "dice_all_slices": float(np.mean(class_scores[name])),
            "dice_positive_slices": float(np.mean(class_positive_scores[name])) if class_positive_scores[name] else None,
            "dice_empty_slices": float(np.mean(class_empty_scores[name])) if class_empty_scores[name] else None,
            "target_positive_slices": int(target_positive),
            "target_empty_slices": int(target_empty),
            "positive_slice_detection_rate": (
                float(presence["pred_positive_on_target_positive"] / target_positive) if target_positive else None
            ),
            "empty_slice_false_positive_rate": (
                float(presence["pred_positive_on_target_empty"] / target_empty) if target_empty else None
            ),
        }

    results["summary"]["mean_dice_all_slices"] = float(
        np.mean([results["classes"][name]["dice_all_slices"] for name in CLASSES])
    )
    positive_values = [
        results["classes"][name]["dice_positive_slices"]
        for name in CLASSES
        if results["classes"][name]["dice_positive_slices"] is not None
    ]
    empty_fp_values = [
        results["classes"][name]["empty_slice_false_positive_rate"]
        for name in CLASSES
        if results["classes"][name]["empty_slice_false_positive_rate"] is not None
    ]
    results["summary"]["mean_dice_positive_slices"] = float(np.mean(positive_values)) if positive_values else None
    results["summary"]["mean_empty_slice_false_positive_rate"] = (
        float(np.mean(empty_fp_values)) if empty_fp_values else None
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = get_device(cfg["train"]["device"])
    probs, targets, cls_probs = collect_predictions(cfg, Path(args.checkpoint), device)
    results = evaluate(probs, targets, cls_probs, cfg)
    out = Path(args.out) if args.out else Path(cfg["train"]["output_dir"]) / "eval_metrics.json"
    ensure_dir(out.parent)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved evaluation: {out}")


if __name__ == "__main__":
    main()
