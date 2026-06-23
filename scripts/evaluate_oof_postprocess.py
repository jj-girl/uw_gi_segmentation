import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.threshold_search import collect_predictions, dice_numpy
from src.uwgi.dataset import CLASSES
from src.uwgi.postprocess import enforce_z_continuity, postprocess_slice, remove_small_components_3d
from src.uwgi.train import load_or_build_metadata
from src.uwgi.utils import ensure_dir, get_device, load_yaml


def parse_z_min_run(value) -> int | list[int]:
    if value is None:
        return 1
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    text = str(value)
    if "," in text:
        return [int(x) for x in text.split(",") if x]
    return int(text)


def parse_bool_list(value, length: int) -> list[bool]:
    if value is None:
        return [False] * length
    if isinstance(value, bool):
        return [value] * length
    if isinstance(value, (list, tuple)):
        return [bool(x) for x in value]
    text = str(value)
    if "," in text:
        return [item.strip().lower() in {"1", "true", "yes", "y"} for item in text.split(",") if item.strip()]
    return [text.strip().lower() in {"1", "true", "yes", "y"}] * length


def parse_int_list(value, length: int, default: int = 0) -> list[int]:
    if value is None:
        return [default] * length
    array = np.asarray(value, dtype=np.int32)
    if array.ndim == 0:
        return [int(array)] * length
    return [int(x) for x in array.tolist()]


def apply_z_continuity(masks: np.ndarray, meta, min_run: int) -> np.ndarray:
    min_runs = np.asarray(min_run, dtype=np.int32)
    if min_runs.ndim == 0:
        if int(min_runs) <= 1:
            return masks
        min_runs = np.repeat(int(min_runs), masks.shape[1])
    if np.all(min_runs <= 1):
        return masks
    cleaned = masks.copy()
    for _, group in meta.groupby(["case", "day"], sort=False):
        order = group.sort_values("slice").index.to_numpy()
        volume = np.transpose(cleaned[order], (1, 0, 2, 3))
        for channel, channel_min_run in enumerate(min_runs):
            volume[channel : channel + 1] = enforce_z_continuity(
                volume[channel : channel + 1],
                min_run=int(channel_min_run),
            )
        cleaned[order] = np.transpose(volume, (1, 0, 2, 3))
    return cleaned


def apply_3d_components(
    masks: np.ndarray,
    meta,
    min_volume,
    keep_largest,
    connectivity: int,
) -> np.ndarray:
    min_volumes = parse_int_list(min_volume, masks.shape[1], default=0)
    keep_largest_values = parse_bool_list(keep_largest, masks.shape[1])
    if all(value <= 0 for value in min_volumes) and not any(keep_largest_values):
        return masks
    cleaned = masks.copy()
    for _, group in meta.groupby(["case", "day"], sort=False):
        order = group.sort_values("slice").index.to_numpy()
        volume = np.transpose(cleaned[order], (1, 0, 2, 3))
        for channel, (channel_min_volume, channel_keep_largest) in enumerate(
            zip(min_volumes, keep_largest_values)
        ):
            volume[channel] = remove_small_components_3d(
                volume[channel],
                min_volume=int(channel_min_volume),
                keep_largest=bool(channel_keep_largest),
                connectivity=connectivity,
            )
        cleaned[order] = np.transpose(volume, (1, 0, 2, 3))
    return cleaned


def evaluate_masks(preds: np.ndarray, targets: np.ndarray) -> dict:
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

    for idx in range(preds.shape[0]):
        for channel, name in enumerate(CLASSES):
            pred_mask = preds[idx, channel]
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


def aggregate_fold_results(fold_results: list[dict]) -> dict:
    results = {"classes": {}, "summary": {}}
    for name in CLASSES:
        total_slices = 0
        total_positive = 0
        total_empty = 0
        dice_all_sum = 0.0
        dice_positive_sum = 0.0
        dice_empty_sum = 0.0
        positive_detected = 0.0
        empty_false_positive = 0.0
        for fold_result in fold_results:
            cls_result = fold_result["classes"][name]
            target_positive = int(cls_result["target_positive_slices"])
            target_empty = int(cls_result["target_empty_slices"])
            slices = target_positive + target_empty
            total_slices += slices
            total_positive += target_positive
            total_empty += target_empty
            dice_all_sum += cls_result["dice_all_slices"] * slices
            if cls_result["dice_positive_slices"] is not None:
                dice_positive_sum += cls_result["dice_positive_slices"] * target_positive
            if cls_result["dice_empty_slices"] is not None:
                dice_empty_sum += cls_result["dice_empty_slices"] * target_empty
            if cls_result["positive_slice_detection_rate"] is not None:
                positive_detected += cls_result["positive_slice_detection_rate"] * target_positive
            if cls_result["empty_slice_false_positive_rate"] is not None:
                empty_false_positive += cls_result["empty_slice_false_positive_rate"] * target_empty

        results["classes"][name] = {
            "dice_all_slices": float(dice_all_sum / max(total_slices, 1)),
            "dice_positive_slices": float(dice_positive_sum / total_positive) if total_positive else None,
            "dice_empty_slices": float(dice_empty_sum / total_empty) if total_empty else None,
            "target_positive_slices": int(total_positive),
            "target_empty_slices": int(total_empty),
            "positive_slice_detection_rate": float(positive_detected / total_positive) if total_positive else None,
            "empty_slice_false_positive_rate": float(empty_false_positive / total_empty) if total_empty else None,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-config-glob", required=True)
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--z-min-run", default=None)
    args = parser.parse_args()

    config_paths = sorted(Path(path) for path in glob.glob(args.fold_config_glob))
    if not config_paths:
        raise FileNotFoundError(f"No configs matched {args.fold_config_glob}")

    fold_results = []
    for config_path in config_paths:
        cfg = load_yaml(config_path)
        checkpoint = Path(cfg["train"]["output_dir"]) / args.checkpoint_name
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for {config_path}: {checkpoint}")
        device = get_device(cfg["train"]["device"])
        probs, targets, cls_probs = collect_predictions(cfg, checkpoint, device)

        metadata = load_or_build_metadata(cfg)
        valid_fold = cfg["data"]["valid_fold"]
        valid_meta = metadata[metadata["fold"] == valid_fold].reset_index(drop=True)
        if cfg["data"].get("limit_valid_samples"):
            valid_meta = valid_meta.head(int(cfg["data"]["limit_valid_samples"])).reset_index(drop=True)

        preds = []
        post_cfg = cfg.get("postprocess", {})
        for idx in range(probs.shape[0]):
            preds.append(
                postprocess_slice(
                    probs[idx],
                    cls_probs=cls_probs[idx] if cls_probs is not None else None,
                    mask_thresholds=post_cfg.get("mask_thresholds", 0.5),
                    cls_thresholds=post_cfg.get("cls_thresholds", 0.5),
                    min_area=post_cfg.get("min_area", 0),
                )
            )
        preds = np.stack(preds, axis=0)
        z_min_run = parse_z_min_run(args.z_min_run if args.z_min_run is not None else post_cfg.get("z_min_run", 1))
        preds = apply_z_continuity(preds, valid_meta, z_min_run)
        preds = apply_3d_components(
            preds,
            valid_meta,
            min_volume=post_cfg.get("min_volume", 0),
            keep_largest=post_cfg.get("keep_largest_component", False),
            connectivity=int(post_cfg.get("component_connectivity", 1)),
        )
        fold_result = evaluate_masks(preds, targets)
        fold_result["config"] = str(config_path)
        fold_result["fold"] = int(valid_fold)
        fold_results.append(fold_result)

    results = {
        "summary": aggregate_fold_results(fold_results),
        "folds": fold_results,
        "num_folds": len(config_paths),
        "z_min_run": parse_z_min_run(args.z_min_run) if args.z_min_run is not None else "config",
        "component_postprocess": "config",
    }

    out = Path(args.out)
    ensure_dir(out.parent)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))
    print(f"Saved OOF evaluation: {out}")


if __name__ == "__main__":
    main()
