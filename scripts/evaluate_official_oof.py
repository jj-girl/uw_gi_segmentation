from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_oof_postprocess import apply_3d_components, apply_z_continuity
from scripts.threshold_search import collect_predictions
from src.uwgi.dataset import CLASSES
from src.uwgi.postprocess import postprocess_slice
from src.uwgi.train import load_or_build_metadata
from src.uwgi.utils import ensure_dir, get_device, load_yaml


def dice_3d(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    denom = int(pred.sum()) + int(target.sum())
    if denom == 0:
        return 1.0
    inter = int(np.logical_and(pred, target).sum())
    return float((2.0 * inter + eps) / (denom + eps))


def surface(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    mask = mask.astype(bool)
    if not mask.any():
        return mask
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_xor(mask, eroded)


def hd95_mm(pred: np.ndarray, target: np.ndarray, spacing: tuple[float, float, float]) -> float | None:
    from scipy import ndimage

    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return None

    pred_surface = surface(pred)
    target_surface = surface(target)
    if not pred_surface.any() or not target_surface.any():
        return None

    target_distance = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    pred_distance = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([target_distance[pred_surface], pred_distance[target_surface]])
    if distances.size == 0:
        return None
    return float(np.percentile(distances, 95))


def hausdorff_score(hd95: float | None, shape: tuple[int, int, int], spacing: tuple[float, float, float]) -> float:
    if hd95 is None or not np.isfinite(hd95):
        return 0.0
    diagonal = float(np.sqrt(sum(((dim - 1) * sp) ** 2 for dim, sp in zip(shape, spacing))))
    if diagonal <= 0:
        return 1.0 if hd95 == 0 else 0.0
    return float(max(0.0, 1.0 - hd95 / diagonal))


def parse_float_list(value: str | None, default: list[float]) -> list[float]:
    if value is None:
        return default
    return [float(item) for item in value.split(",") if item.strip()]


def collect_fold_masks(config_path: Path, checkpoint_name: str, device) -> tuple[np.ndarray, np.ndarray, object]:
    cfg = load_yaml(config_path)
    checkpoint = Path(cfg["train"]["output_dir"]) / checkpoint_name
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for {config_path}: {checkpoint}")
    probs, targets, cls_probs = collect_predictions(cfg, checkpoint, device)

    metadata = load_or_build_metadata(cfg)
    valid_fold = cfg["data"]["valid_fold"]
    valid_meta = metadata[metadata["fold"] == valid_fold].reset_index(drop=True)
    if cfg["data"].get("limit_valid_samples"):
        valid_meta = valid_meta.head(int(cfg["data"]["limit_valid_samples"])).reset_index(drop=True)

    post_cfg = cfg.get("postprocess", {})
    preds = []
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
    preds = apply_z_continuity(preds, valid_meta, post_cfg.get("z_min_run", 1))
    preds = apply_3d_components(
        preds,
        valid_meta,
        min_volume=post_cfg.get("min_volume", 0),
        keep_largest=post_cfg.get("keep_largest_component", False),
        connectivity=int(post_cfg.get("component_connectivity", 1)),
    )
    return preds, targets.astype(np.uint8), valid_meta


def evaluate_volumes(
    preds: np.ndarray,
    targets: np.ndarray,
    meta,
    z_spacing: float,
    weights: tuple[float, float],
    class_names: list[str] | None = None,
) -> list[dict]:
    rows = []
    selected_classes = CLASSES if class_names is None else class_names
    selected_indices = [(CLASSES.index(class_name), class_name) for class_name in selected_classes]
    for (case, day), group in meta.groupby(["case", "day"], sort=False):
        order = group.sort_values("slice").index.to_numpy()
        volume_pred = np.transpose(preds[order], (1, 0, 2, 3))
        volume_target = np.transpose(targets[order], (1, 0, 2, 3))
        spacing_h = float(group["spacing_h"].median()) if "spacing_h" in group else 1.5
        spacing_w = float(group["spacing_w"].median()) if "spacing_w" in group else 1.5
        spacing = (float(z_spacing), spacing_h, spacing_w)
        shape = tuple(int(x) for x in volume_pred.shape[1:])
        for channel, class_name in selected_indices:
            dsc = dice_3d(volume_pred[channel], volume_target[channel])
            hd = hd95_mm(volume_pred[channel], volume_target[channel], spacing)
            hd_score = hausdorff_score(hd, shape, spacing)
            combined = weights[0] * dsc + weights[1] * hd_score
            rows.append(
                {
                    "case": str(case),
                    "day": str(day),
                    "class": class_name,
                    "num_slices": int(len(order)),
                    "target_voxels": int(volume_target[channel].sum()),
                    "pred_voxels": int(volume_pred[channel].sum()),
                    "dice_3d": dsc,
                    "hd95_mm": None if hd is None else hd,
                    "hausdorff_score_proxy": hd_score,
                    "combined_proxy": combined,
                }
            )
    return rows


def summarize(rows: list[dict], class_names: list[str] | None = None) -> dict:
    classes = {}
    selected_classes = CLASSES if class_names is None else class_names
    for class_name in selected_classes:
        class_rows = [row for row in rows if row["class"] == class_name]
        classes[class_name] = {
            "mean_dice_3d": float(np.mean([row["dice_3d"] for row in class_rows])),
            "mean_hd95_mm": float(np.mean([row["hd95_mm"] for row in class_rows if row["hd95_mm"] is not None])),
            "mean_hausdorff_score_proxy": float(np.mean([row["hausdorff_score_proxy"] for row in class_rows])),
            "mean_combined_proxy": float(np.mean([row["combined_proxy"] for row in class_rows])),
            "num_volumes": int(len(class_rows)),
        }
    return {
        "mean_dice_3d": float(np.mean([value["mean_dice_3d"] for value in classes.values()])),
        "mean_hd95_mm": float(np.mean([value["mean_hd95_mm"] for value in classes.values()])),
        "mean_hausdorff_score_proxy": float(
            np.mean([value["mean_hausdorff_score_proxy"] for value in classes.values()])
        ),
        "mean_combined_proxy": float(np.mean([value["mean_combined_proxy"] for value in classes.values()])),
        "classes": classes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate OOF predictions with a Kaggle-style 3D proxy. "
            "The exact private leaderboard implementation is not public here; this reports 3D Dice, HD95 in mm, "
            "and a normalized Hausdorff proxy combined with the official 0.4/0.6 weights."
        )
    )
    parser.add_argument("--fold-config-glob", required=True)
    parser.add_argument("--checkpoint-name", default="best_postprocess.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--z-spacing", type=float, default=3.0)
    parser.add_argument("--weights", default="0.4,0.6")
    parser.add_argument(
        "--classes",
        default=None,
        help="Optional comma-separated subset of classes to evaluate, e.g. small_bowel.",
    )
    args = parser.parse_args()

    weights = parse_float_list(args.weights, [0.4, 0.6])
    if len(weights) != 2:
        raise ValueError("--weights must contain two comma-separated floats")
    weights_tuple = (float(weights[0]), float(weights[1]))
    class_names = None
    if args.classes:
        class_names = [item.strip() for item in args.classes.split(",") if item.strip()]
        unknown = sorted(set(class_names) - set(CLASSES))
        if unknown:
            raise ValueError(f"Unknown classes in --classes: {unknown}; valid classes: {CLASSES}")

    config_paths = sorted(Path(path) for path in glob.glob(args.fold_config_glob))
    if not config_paths:
        raise FileNotFoundError(f"No configs matched {args.fold_config_glob}")

    all_rows = []
    for config_path in config_paths:
        cfg = load_yaml(config_path)
        device = get_device(args.device if args.device != "auto" else cfg["train"]["device"])
        preds, targets, meta = collect_fold_masks(config_path, args.checkpoint_name, device)
        fold_rows = evaluate_volumes(preds, targets, meta, args.z_spacing, weights_tuple, class_names=class_names)
        for row in fold_rows:
            row["fold"] = int(cfg["data"]["valid_fold"])
            row["config"] = str(config_path)
        all_rows.extend(fold_rows)

    result = {
        "summary": summarize(all_rows, class_names=class_names),
        "volumes": all_rows,
        "num_folds": len(config_paths),
        "checkpoint_name": args.checkpoint_name,
        "z_spacing": args.z_spacing,
        "weights": {"dice": weights_tuple[0], "hausdorff": weights_tuple[1]},
        "classes_evaluated": CLASSES if class_names is None else class_names,
        "note": (
            "Kaggle-style proxy: 3D Dice plus HD95 normalized by volume physical diagonal. "
            "Use for local model selection; hidden leaderboard may differ."
        ),
    }
    out = Path(args.out)
    ensure_dir(out.parent)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved official-style OOF evaluation: {out}")


if __name__ == "__main__":
    main()
