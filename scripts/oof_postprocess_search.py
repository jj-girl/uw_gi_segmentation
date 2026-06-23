import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_oof_postprocess import aggregate_fold_results, evaluate_masks
from scripts.threshold_search import collect_predictions, dice_numpy
from src.uwgi.dataset import CLASSES
from src.uwgi.postprocess import enforce_z_continuity, remove_small_components, remove_small_components_3d
from src.uwgi.train import load_or_build_metadata
from src.uwgi.utils import ensure_dir, get_device, load_yaml


def parse_int_grid(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x]


def normalize_float_list(value, length: int) -> list[float]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return [float(array)] * length
    return [float(x) for x in array.tolist()]


def normalize_int_list(value, length: int) -> list[int]:
    array = np.asarray(value, dtype=np.int32)
    if array.ndim == 0:
        return [int(array)] * length
    return [int(x) for x in array.tolist()]


def valid_metadata(cfg: dict):
    metadata = load_or_build_metadata(cfg)
    valid_fold = cfg["data"]["valid_fold"]
    valid_meta = metadata[metadata["fold"] == valid_fold].reset_index(drop=True)
    if cfg["data"].get("limit_valid_samples"):
        valid_meta = valid_meta.head(int(cfg["data"]["limit_valid_samples"])).reset_index(drop=True)
    return valid_meta


def apply_z_to_channel(masks: np.ndarray, meta, min_run: int) -> np.ndarray:
    if min_run <= 1:
        return masks.astype(np.uint8)
    cleaned = masks.copy().astype(np.uint8)
    for _, group in meta.groupby(["case", "day"], sort=False):
        order = group.sort_values("slice").index.to_numpy()
        volume = cleaned[order][None, ...]
        cleaned[order] = enforce_z_continuity(volume, min_run=min_run)[0]
    return cleaned


def apply_components_to_channel(
    masks: np.ndarray,
    meta,
    min_volume: int,
    keep_largest: bool,
    connectivity: int,
) -> np.ndarray:
    if min_volume <= 0 and not keep_largest:
        return masks.astype(np.uint8)
    cleaned = masks.copy().astype(np.uint8)
    for _, group in meta.groupby(["case", "day"], sort=False):
        order = group.sort_values("slice").index.to_numpy()
        cleaned[order] = remove_small_components_3d(
            cleaned[order],
            min_volume=min_volume,
            keep_largest=keep_largest,
            connectivity=connectivity,
        )
    return cleaned


def build_component_label_cache(masks: np.ndarray, meta, connectivity: int) -> list[dict]:
    structure = None
    try:
        from scipy import ndimage

        structure = ndimage.generate_binary_structure(rank=3, connectivity=connectivity)
    except Exception as exc:  # pragma: no cover - scipy is a project dependency here.
        raise RuntimeError("scipy is required for cached 3D component search") from exc

    cache = []
    for _, group in meta.groupby(["case", "day"], sort=False):
        order = group.sort_values("slice").index.to_numpy()
        volume = masks[order].astype(np.uint8)
        labels, num_labels = ndimage.label(volume, structure=structure)
        sizes = np.bincount(labels.ravel())
        if sizes.size:
            sizes[0] = 0
        cache.append(
            {
                "order": order,
                "labels": labels.astype(np.int32, copy=False),
                "sizes": sizes.astype(np.int64, copy=False),
                "num_labels": int(num_labels),
            }
        )
    return cache


def build_component_stats_cache(masks: np.ndarray, targets: np.ndarray, meta, connectivity: int) -> list[dict]:
    try:
        from scipy import ndimage

        structure = ndimage.generate_binary_structure(rank=3, connectivity=connectivity)
    except Exception as exc:  # pragma: no cover - scipy is a project dependency here.
        raise RuntimeError("scipy is required for cached 3D component search") from exc

    cache = []
    for _, group in meta.groupby(["case", "day"], sort=False):
        order = group.sort_values("slice").index.to_numpy()
        volume = masks[order].astype(np.uint8)
        target_volume = targets[order].astype(np.uint8)
        labels, num_labels = ndimage.label(volume, structure=structure)
        sizes = np.bincount(labels.ravel(), minlength=num_labels + 1)
        if sizes.size:
            sizes[0] = 0

        num_slices = labels.shape[0]
        slice_counts = np.zeros((num_labels + 1, num_slices), dtype=np.int32)
        slice_intersections = np.zeros((num_labels + 1, num_slices), dtype=np.int32)
        target_sum = target_volume.reshape(num_slices, -1).sum(axis=1).astype(np.int32)
        for slice_idx in range(num_slices):
            slice_labels = labels[slice_idx].ravel()
            slice_counts[:, slice_idx] = np.bincount(slice_labels, minlength=num_labels + 1)
            target_pixels = target_volume[slice_idx].astype(bool).ravel()
            if target_pixels.any():
                slice_intersections[:, slice_idx] = np.bincount(
                    slice_labels[target_pixels],
                    minlength=num_labels + 1,
                )
        slice_counts[0] = 0
        slice_intersections[0] = 0
        cache.append(
            {
                "sizes": sizes.astype(np.int64, copy=False),
                "num_labels": int(num_labels),
                "slice_counts": slice_counts,
                "slice_intersections": slice_intersections,
                "target_sum": target_sum,
            }
        )
    return cache


def apply_components_from_label_cache(
    masks: np.ndarray,
    component_cache: list[dict],
    min_volume: int,
    keep_largest: bool,
) -> np.ndarray:
    if min_volume <= 0 and not keep_largest:
        return masks.astype(np.uint8)
    cleaned = np.zeros_like(masks, dtype=np.uint8)
    for item in component_cache:
        labels = item["labels"]
        sizes = item["sizes"]
        if item["num_labels"] == 0:
            continue
        keep = np.zeros(item["num_labels"] + 1, dtype=bool)
        if keep_largest:
            largest = int(sizes.argmax())
            if largest > 0 and sizes[largest] >= max(min_volume, 1):
                keep[largest] = True
        else:
            keep = sizes >= max(min_volume, 1)
            keep[0] = False
        cleaned[item["order"]] = keep[labels].astype(np.uint8)
    return cleaned


def evaluate_component_stats_cache(
    component_cache: list[dict],
    min_volume: int,
    keep_largest: bool,
) -> dict:
    pred_sums = []
    target_sums = []
    intersections = []
    for item in component_cache:
        target_sums.append(item["target_sum"])
        if item["num_labels"] == 0:
            pred_sums.append(np.zeros_like(item["target_sum"], dtype=np.int64))
            intersections.append(np.zeros_like(item["target_sum"], dtype=np.int64))
            continue

        sizes = item["sizes"]
        keep = np.zeros(item["num_labels"] + 1, dtype=bool)
        if keep_largest:
            largest = int(sizes.argmax())
            if largest > 0 and sizes[largest] >= max(min_volume, 1):
                keep[largest] = True
        else:
            keep = sizes >= max(min_volume, 1)
            keep[0] = False
        if keep.any():
            pred_sums.append(item["slice_counts"][keep].sum(axis=0, dtype=np.int64))
            intersections.append(item["slice_intersections"][keep].sum(axis=0, dtype=np.int64))
        else:
            pred_sums.append(np.zeros_like(item["target_sum"], dtype=np.int64))
            intersections.append(np.zeros_like(item["target_sum"], dtype=np.int64))

    pred_sum = np.concatenate(pred_sums)
    target_sum = np.concatenate(target_sums)
    intersection = np.concatenate(intersections)
    denom = pred_sum + target_sum
    scores_array = np.where(denom == 0, 1.0, (2.0 * intersection + 1e-7) / (denom + 1e-7))

    target_positive_mask = target_sum > 0
    target_empty_mask = ~target_positive_mask
    pred_positive_mask = pred_sum > 0
    target_positive = int(target_positive_mask.sum())
    target_empty = int(target_empty_mask.sum())
    pred_positive_on_target_positive = int(np.logical_and(pred_positive_mask, target_positive_mask).sum())
    pred_positive_on_target_empty = int(np.logical_and(pred_positive_mask, target_empty_mask).sum())

    return {
        "dice_all_slices": float(scores_array.mean()),
        "dice_positive_slices": (
            float(scores_array[target_positive_mask].mean()) if target_positive else None
        ),
        "dice_empty_slices": float(scores_array[target_empty_mask].mean()) if target_empty else None,
        "target_positive_slices": int(target_positive),
        "target_empty_slices": int(target_empty),
        "positive_slice_detection_rate": (
            float(pred_positive_on_target_positive / target_positive) if target_positive else None
        ),
        "empty_slice_false_positive_rate": (
            float(pred_positive_on_target_empty / target_empty) if target_empty else None
        ),
    }


def build_channel_masks(
    probs: np.ndarray,
    cls_probs: np.ndarray | None,
    channel: int,
    mask_threshold: float,
    cls_threshold: float | None,
) -> np.ndarray:
    keep = np.ones(probs.shape[0], dtype=bool)
    if cls_probs is not None and cls_threshold is not None:
        keep = cls_probs[:, channel] >= cls_threshold
    masks = (probs[:, channel] > mask_threshold).astype(np.uint8)
    masks[~keep] = 0
    return masks


def apply_2d_min_area(masks: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return masks.astype(np.uint8)
    cleaned = np.zeros_like(masks, dtype=np.uint8)
    for idx in range(masks.shape[0]):
        cleaned[idx] = remove_small_components(masks[idx], min_area)
    return cleaned


def evaluate_channel(preds: np.ndarray, targets: np.ndarray) -> dict:
    pred_flat = preds.reshape(preds.shape[0], -1).astype(bool, copy=False)
    target_flat = targets.reshape(targets.shape[0], -1).astype(bool, copy=False)
    pred_sum = pred_flat.sum(axis=1)
    target_sum = target_flat.sum(axis=1)
    intersection = np.logical_and(pred_flat, target_flat).sum(axis=1)
    denom = pred_sum + target_sum
    scores_array = np.where(denom == 0, 1.0, (2.0 * intersection + 1e-7) / (denom + 1e-7))

    target_positive_mask = target_sum > 0
    target_empty_mask = ~target_positive_mask
    pred_positive_mask = pred_sum > 0
    target_positive = int(target_positive_mask.sum())
    target_empty = int(target_empty_mask.sum())
    pred_positive_on_target_positive = int(np.logical_and(pred_positive_mask, target_positive_mask).sum())
    pred_positive_on_target_empty = int(np.logical_and(pred_positive_mask, target_empty_mask).sum())

    return {
        "dice_all_slices": float(scores_array.mean()),
        "dice_positive_slices": (
            float(scores_array[target_positive_mask].mean()) if target_positive else None
        ),
        "dice_empty_slices": float(scores_array[target_empty_mask].mean()) if target_empty else None,
        "target_positive_slices": int(target_positive),
        "target_empty_slices": int(target_empty),
        "positive_slice_detection_rate": (
            float(pred_positive_on_target_positive / target_positive) if target_positive else None
        ),
        "empty_slice_false_positive_rate": (
            float(pred_positive_on_target_empty / target_empty) if target_empty else None
        ),
    }


def aggregate_channel_results(results: list[dict]) -> dict:
    total_slices = 0
    total_positive = 0
    total_empty = 0
    dice_all_sum = 0.0
    dice_positive_sum = 0.0
    dice_empty_sum = 0.0
    positive_detected = 0.0
    empty_false_positive = 0.0

    for result in results:
        target_positive = int(result["target_positive_slices"])
        target_empty = int(result["target_empty_slices"])
        slices = target_positive + target_empty
        total_slices += slices
        total_positive += target_positive
        total_empty += target_empty
        dice_all_sum += result["dice_all_slices"] * slices
        if result["dice_positive_slices"] is not None:
            dice_positive_sum += result["dice_positive_slices"] * target_positive
        if result["dice_empty_slices"] is not None:
            dice_empty_sum += result["dice_empty_slices"] * target_empty
        if result["positive_slice_detection_rate"] is not None:
            positive_detected += result["positive_slice_detection_rate"] * target_positive
        if result["empty_slice_false_positive_rate"] is not None:
            empty_false_positive += result["empty_slice_false_positive_rate"] * target_empty

    return {
        "dice_all_slices": float(dice_all_sum / max(total_slices, 1)),
        "dice_positive_slices": float(dice_positive_sum / total_positive) if total_positive else None,
        "dice_empty_slices": float(dice_empty_sum / total_empty) if total_empty else None,
        "target_positive_slices": int(total_positive),
        "target_empty_slices": int(total_empty),
        "positive_slice_detection_rate": float(positive_detected / total_positive) if total_positive else None,
        "empty_slice_false_positive_rate": float(empty_false_positive / total_empty) if total_empty else None,
    }


def search_class(
    fold_data: list[dict],
    channel: int,
    class_name: str,
    mask_threshold: float,
    cls_threshold: float | None,
    min_area_grid: list[int],
    z_min_run_grid: list[int],
    min_volume_grid: list[int],
    keep_largest_options: list[bool],
    connectivity: int,
) -> dict:
    best = None
    candidates = []
    for min_area in min_area_grid:
        area_masks = []
        for fold in fold_data:
            masks = build_channel_masks(
                fold["probs"],
                fold["cls_probs"],
                channel,
                mask_threshold,
                cls_threshold,
            )
            area_masks.append(apply_2d_min_area(masks, min_area))

        for z_min_run in z_min_run_grid:
            z_masks = []
            for masks, fold in zip(area_masks, fold_data):
                z_masks.append(apply_z_to_channel(masks, fold["meta"], z_min_run))

            for keep_largest in keep_largest_options:
                for min_volume in min_volume_grid:
                    if not keep_largest and min_volume <= 0:
                        candidate_name = "none"
                    elif keep_largest:
                        candidate_name = "keep_largest"
                    else:
                        candidate_name = "min_volume"
                    fold_results = []
                    for masks, fold in zip(z_masks, fold_data):
                        preds = apply_components_to_channel(
                            masks,
                            fold["meta"],
                            min_volume=min_volume,
                            keep_largest=keep_largest,
                            connectivity=connectivity,
                        )
                        fold_results.append(evaluate_channel(preds, fold["targets"][:, channel]))
                    result = aggregate_channel_results(fold_results)
                    candidate = {
                        "class": class_name,
                        "mask_threshold": mask_threshold,
                        "cls_threshold": cls_threshold,
                        "min_area": min_area,
                        "z_min_run": z_min_run,
                        "component_mode": candidate_name,
                        "min_volume": min_volume,
                        "keep_largest": keep_largest,
                        "connectivity": connectivity,
                        "metrics": result,
                    }
                    candidates.append(candidate)
                    if best is None or result["dice_all_slices"] > best["metrics"]["dice_all_slices"]:
                        best = candidate
    candidates.sort(key=lambda item: item["metrics"]["dice_all_slices"], reverse=True)
    assert best is not None
    best["top_candidates"] = candidates[:10]
    return best


def candidate_grid(
    class_name: str,
    mask_threshold: float,
    cls_threshold: float | None,
    min_area_grid: list[int],
    z_min_run_grid: list[int],
    min_volume_grid: list[int],
    keep_largest_options: list[bool],
    connectivity: int,
) -> list[dict]:
    candidates = []
    for min_area in min_area_grid:
        for z_min_run in z_min_run_grid:
            for keep_largest in keep_largest_options:
                for min_volume in min_volume_grid:
                    if not keep_largest and min_volume <= 0:
                        component_mode = "none"
                    elif keep_largest:
                        component_mode = "keep_largest"
                    else:
                        component_mode = "min_volume"
                    candidates.append(
                        {
                            "class": class_name,
                            "mask_threshold": mask_threshold,
                            "cls_threshold": cls_threshold,
                            "min_area": min_area,
                            "z_min_run": z_min_run,
                            "component_mode": component_mode,
                            "min_volume": min_volume,
                            "keep_largest": keep_largest,
                            "connectivity": connectivity,
                        }
                    )
    return candidates


def empty_accumulator() -> dict:
    return {
        "total_slices": 0,
        "target_positive": 0,
        "target_empty": 0,
        "dice_all_sum": 0.0,
        "dice_positive_sum": 0.0,
        "dice_empty_sum": 0.0,
        "positive_detected": 0.0,
        "empty_false_positive": 0.0,
    }


def add_to_accumulator(accumulator: dict, result: dict) -> None:
    target_positive = int(result["target_positive_slices"])
    target_empty = int(result["target_empty_slices"])
    slices = target_positive + target_empty
    accumulator["total_slices"] += slices
    accumulator["target_positive"] += target_positive
    accumulator["target_empty"] += target_empty
    accumulator["dice_all_sum"] += result["dice_all_slices"] * slices
    if result["dice_positive_slices"] is not None:
        accumulator["dice_positive_sum"] += result["dice_positive_slices"] * target_positive
    if result["dice_empty_slices"] is not None:
        accumulator["dice_empty_sum"] += result["dice_empty_slices"] * target_empty
    if result["positive_slice_detection_rate"] is not None:
        accumulator["positive_detected"] += result["positive_slice_detection_rate"] * target_positive
    if result["empty_slice_false_positive_rate"] is not None:
        accumulator["empty_false_positive"] += result["empty_slice_false_positive_rate"] * target_empty


def finalize_accumulator(accumulator: dict) -> dict:
    total_slices = accumulator["total_slices"]
    target_positive = accumulator["target_positive"]
    target_empty = accumulator["target_empty"]
    return {
        "dice_all_slices": float(accumulator["dice_all_sum"] / max(total_slices, 1)),
        "dice_positive_slices": (
            float(accumulator["dice_positive_sum"] / target_positive) if target_positive else None
        ),
        "dice_empty_slices": float(accumulator["dice_empty_sum"] / target_empty) if target_empty else None,
        "target_positive_slices": int(target_positive),
        "target_empty_slices": int(target_empty),
        "positive_slice_detection_rate": (
            float(accumulator["positive_detected"] / target_positive) if target_positive else None
        ),
        "empty_slice_false_positive_rate": (
            float(accumulator["empty_false_positive"] / target_empty) if target_empty else None
        ),
    }


def apply_candidate_to_channel(
    probs: np.ndarray,
    cls_probs: np.ndarray | None,
    meta,
    channel: int,
    candidate: dict,
) -> np.ndarray:
    masks = build_channel_masks(
        probs,
        cls_probs,
        channel,
        candidate["mask_threshold"],
        candidate["cls_threshold"],
    )
    masks = apply_2d_min_area(masks, candidate["min_area"])
    masks = apply_z_to_channel(masks, meta, candidate["z_min_run"])
    return apply_components_to_channel(
        masks,
        meta,
        min_volume=candidate["min_volume"],
        keep_largest=candidate["keep_largest"],
        connectivity=candidate["connectivity"],
    )


def update_candidate_scores_for_fold(
    candidates_by_class: dict[str, list[dict]],
    accumulators_by_class: dict[str, list[dict]],
    probs: np.ndarray,
    targets: np.ndarray,
    cls_probs: np.ndarray | None,
    meta,
) -> None:
    for channel, class_name in enumerate(CLASSES):
        candidates = candidates_by_class[class_name]
        accumulators = accumulators_by_class[class_name]
        base_masks = build_channel_masks(
            probs,
            cls_probs,
            channel,
            candidates[0]["mask_threshold"],
            candidates[0]["cls_threshold"],
        )
        area_cache = {}
        z_cache = {}
        component_stats_cache = {}
        for idx, candidate in enumerate(candidates):
            min_area = candidate["min_area"]
            z_min_run = candidate["z_min_run"]
            area_masks = area_cache.get(min_area)
            if area_masks is None:
                area_masks = apply_2d_min_area(base_masks, min_area)
                area_cache[min_area] = area_masks
            z_key = (min_area, z_min_run)
            z_masks = z_cache.get(z_key)
            if z_masks is None:
                z_masks = apply_z_to_channel(area_masks, meta, z_min_run)
                z_cache[z_key] = z_masks
            if candidate["min_volume"] <= 0 and not candidate["keep_largest"]:
                preds = z_masks
                result = evaluate_channel(preds, targets[:, channel])
            else:
                stats = component_stats_cache.get(z_key)
                if stats is None:
                    stats = build_component_stats_cache(z_masks, targets[:, channel], meta, candidate["connectivity"])
                    component_stats_cache[z_key] = stats
                result = evaluate_component_stats_cache(
                    stats,
                    min_volume=candidate["min_volume"],
                    keep_largest=candidate["keep_largest"],
                )
            add_to_accumulator(accumulators[idx], result)


def select_best_candidates(candidates_by_class: dict[str, list[dict]], accumulators_by_class: dict[str, list[dict]]) -> dict:
    best_by_class = {}
    for class_name in CLASSES:
        ranked = []
        for candidate, accumulator in zip(candidates_by_class[class_name], accumulators_by_class[class_name]):
            item = dict(candidate)
            item["metrics"] = finalize_accumulator(accumulator)
            ranked.append(item)
        ranked.sort(key=lambda item: item["metrics"]["dice_all_slices"], reverse=True)
        best = dict(ranked[0])
        best["top_candidates"] = [dict(item) for item in ranked[:10]]
        best_by_class[class_name] = best
    return best_by_class


def evaluate_selected_for_fold(
    selected: dict[str, dict],
    config_path: Path,
    checkpoint_name: str,
) -> dict:
    cfg = load_yaml(config_path)
    checkpoint = Path(cfg["train"]["output_dir"]) / checkpoint_name
    device = get_device(cfg["train"]["device"])
    probs, targets, cls_probs = collect_predictions(cfg, checkpoint, device)
    meta = valid_metadata(cfg)
    preds = np.zeros_like(targets, dtype=np.uint8)
    for channel, class_name in enumerate(CLASSES):
        preds[:, channel] = apply_candidate_to_channel(probs, cls_probs, meta, channel, selected[class_name])
    fold_result = evaluate_masks(preds, targets)
    fold_result["config"] = str(config_path)
    fold_result["fold"] = int(cfg["data"]["valid_fold"])
    return fold_result


def assemble_predictions(fold_data: list[dict], class_params: dict[str, dict]) -> list[dict]:
    fold_results = []
    for fold in fold_data:
        preds = np.zeros_like(fold["targets"], dtype=np.uint8)
        for channel, class_name in enumerate(CLASSES):
            params = class_params[class_name]
            masks = build_channel_masks(
                fold["probs"],
                fold["cls_probs"],
                channel,
                params["mask_threshold"],
                params["cls_threshold"],
            )
            masks = apply_2d_min_area(masks, params["min_area"])
            masks = apply_z_to_channel(masks, fold["meta"], params["z_min_run"])
            masks = apply_components_to_channel(
                masks,
                fold["meta"],
                min_volume=params["min_volume"],
                keep_largest=params["keep_largest"],
                connectivity=params["connectivity"],
            )
            preds[:, channel] = masks
        fold_result = evaluate_masks(preds, fold["targets"])
        fold_result["config"] = fold["config"]
        fold_result["fold"] = int(fold["valid_fold"])
        fold_results.append(fold_result)
    return fold_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-config-glob", required=True)
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-area-grid", default="0,8,16,24,48,96,192")
    parser.add_argument("--z-min-run-grid", default="1,2,3")
    parser.add_argument("--min-volume-grid", default="0,32,64,128,256,512")
    parser.add_argument("--keep-largest", action="store_true")
    parser.add_argument("--connectivity", type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()

    config_paths = sorted(Path(path) for path in glob.glob(args.fold_config_glob))
    if not config_paths:
        raise FileNotFoundError(f"No configs matched {args.fold_config_glob}")

    thresholds = json.loads(Path(args.thresholds_json).read_text(encoding="utf-8"))
    min_area_grid = parse_int_grid(args.min_area_grid)
    z_min_run_grid = parse_int_grid(args.z_min_run_grid)
    min_volume_grid = parse_int_grid(args.min_volume_grid)
    keep_largest_options = [False, True] if args.keep_largest else [False]

    candidates_by_class = {}
    accumulators_by_class = {}
    for class_name in CLASSES:
        class_thresholds = thresholds[class_name]
        candidates_by_class[class_name] = candidate_grid(
            class_name=class_name,
            mask_threshold=float(class_thresholds["mask_threshold"]),
            cls_threshold=(
                None
                if class_thresholds.get("cls_threshold") is None
                else float(class_thresholds["cls_threshold"])
            ),
            min_area_grid=min_area_grid,
            z_min_run_grid=z_min_run_grid,
            min_volume_grid=min_volume_grid,
            keep_largest_options=keep_largest_options,
            connectivity=args.connectivity,
        )
        accumulators_by_class[class_name] = [empty_accumulator() for _ in candidates_by_class[class_name]]

    for config_path in config_paths:
        print(f"Loading predictions for {config_path}", flush=True)
        cfg = load_yaml(config_path)
        checkpoint = Path(cfg["train"]["output_dir"]) / args.checkpoint_name
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for {config_path}: {checkpoint}")
        device = get_device(cfg["train"]["device"])
        probs, targets, cls_probs = collect_predictions(cfg, checkpoint, device)
        print(f"Scoring candidates for {config_path}", flush=True)
        update_candidate_scores_for_fold(
            candidates_by_class,
            accumulators_by_class,
            probs,
            targets,
            cls_probs,
            valid_metadata(cfg),
        )
        print(f"Finished candidate scoring for {config_path}", flush=True)

    class_params = select_best_candidates(candidates_by_class, accumulators_by_class)
    for class_name in CLASSES:
        print(json.dumps({class_name: class_params[class_name]}, indent=2))

    fold_results = []
    for config_path in config_paths:
        print(f"Evaluating selected postprocess for {config_path}", flush=True)
        fold_results.append(evaluate_selected_for_fold(class_params, config_path, args.checkpoint_name))
    aggregate = aggregate_fold_results(fold_results)
    results = {
        "summary": aggregate,
        "classes": class_params,
        "folds": fold_results,
        "num_folds": len(config_paths),
        "source_thresholds": str(args.thresholds_json),
        "search_space": {
            "min_area_grid": min_area_grid,
            "z_min_run_grid": z_min_run_grid,
            "min_volume_grid": min_volume_grid,
            "keep_largest_options": keep_largest_options,
            "connectivity": args.connectivity,
        },
    }

    out = Path(args.out)
    ensure_dir(out.parent)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))
    print(f"Saved OOF postprocess search: {out}")


if __name__ == "__main__":
    main()
