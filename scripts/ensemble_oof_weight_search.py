from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_oof_postprocess import (  # noqa: E402
    aggregate_fold_results,
    apply_3d_components,
    apply_z_continuity,
    evaluate_masks,
)
from scripts.threshold_search import collect_predictions  # noqa: E402
from src.uwgi.postprocess import postprocess_slice  # noqa: E402
from src.uwgi.train import load_or_build_metadata  # noqa: E402
from src.uwgi.utils import ensure_dir, get_device, load_yaml  # noqa: E402


def parse_float_grid(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def configs_by_fold(pattern: str) -> dict[int, Path]:
    paths = sorted(Path(path) for path in glob.glob(pattern))
    result = {}
    for path in paths:
        cfg = load_yaml(path)
        result[int(cfg["data"]["valid_fold"])] = path
    return result


def apply_postprocess(probs: np.ndarray, cls_probs: np.ndarray | None, cfg: dict, meta) -> np.ndarray:
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
    preds = apply_z_continuity(preds, meta, post_cfg.get("z_min_run", 1))
    preds = apply_3d_components(
        preds,
        meta,
        min_volume=post_cfg.get("min_volume", 0),
        keep_largest=post_cfg.get("keep_largest_component", False),
        connectivity=int(post_cfg.get("component_connectivity", 1)),
    )
    return preds


def valid_meta(cfg: dict):
    metadata = load_or_build_metadata(cfg)
    valid_fold = int(cfg["data"]["valid_fold"])
    meta = metadata[metadata["fold"] == valid_fold].reset_index(drop=True)
    if cfg["data"].get("limit_valid_samples"):
        meta = meta.head(int(cfg["data"]["limit_valid_samples"])).reset_index(drop=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a-glob", required=True)
    parser.add_argument("--model-b-glob", required=True)
    parser.add_argument("--model-a-checkpoint", default="best_postprocess.pt")
    parser.add_argument("--model-b-checkpoint", default="best_postprocess.pt")
    parser.add_argument("--weights-b", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--postprocess-source", choices=["a", "b"], default="b")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model_a = configs_by_fold(args.model_a_glob)
    model_b = configs_by_fold(args.model_b_glob)
    folds = sorted(set(model_a) & set(model_b))
    if not folds:
        raise FileNotFoundError("No overlapping folds between model config globs")
    missing = sorted((set(model_a) | set(model_b)) - set(folds))
    if missing:
        raise ValueError(f"Config fold mismatch; missing paired folds: {missing}")

    weights_b = parse_float_grid(args.weights_b)
    fold_results_by_weight = {weight_b: [] for weight_b in weights_b}
    for fold in folds:
        cfg_a = load_yaml(model_a[fold])
        cfg_b = load_yaml(model_b[fold])
        ckpt_a = Path(cfg_a["train"]["output_dir"]) / args.model_a_checkpoint
        ckpt_b = Path(cfg_b["train"]["output_dir"]) / args.model_b_checkpoint
        if not ckpt_a.exists():
            raise FileNotFoundError(f"Missing model A checkpoint: {ckpt_a}")
        if not ckpt_b.exists():
            raise FileNotFoundError(f"Missing model B checkpoint: {ckpt_b}")

        device = get_device(cfg_b["train"]["device"])
        probs_a, targets_a, cls_a = collect_predictions(cfg_a, ckpt_a, device)
        probs_b, targets_b, cls_b = collect_predictions(cfg_b, ckpt_b, device)
        if probs_a.shape != probs_b.shape or targets_a.shape != targets_b.shape:
            raise ValueError(f"Shape mismatch on fold{fold}: {probs_a.shape} vs {probs_b.shape}")
        if not np.array_equal(targets_a, targets_b):
            raise ValueError(f"Target mismatch on fold{fold}")

        post_cfg = cfg_b if args.postprocess_source == "b" else cfg_a
        meta = valid_meta(post_cfg)
        for weight_b in weights_b:
            weight_a = 1.0 - weight_b
            probs = weight_a * probs_a + weight_b * probs_b
            if cls_a is None or cls_b is None:
                cls_probs = None
            else:
                cls_probs = weight_a * cls_a + weight_b * cls_b
            preds = apply_postprocess(probs, cls_probs, post_cfg, meta)
            fold_result = evaluate_masks(preds, targets_a.astype(np.uint8))
            fold_result["fold"] = fold
            fold_result["weight_a"] = weight_a
            fold_result["weight_b"] = weight_b
            fold_results_by_weight[weight_b].append(fold_result)

    results = []
    for weight_b, fold_results in fold_results_by_weight.items():
        weight_a = 1.0 - weight_b
        aggregate = aggregate_fold_results(fold_results)
        results.append(
            {
                "weight_a": weight_a,
                "weight_b": weight_b,
                "summary": aggregate,
                "folds": fold_results,
            }
        )

    results = sorted(
        results,
        key=lambda item: item["summary"]["summary"]["mean_dice_all_slices"],
        reverse=True,
    )
    output = {
        "model_a_glob": args.model_a_glob,
        "model_b_glob": args.model_b_glob,
        "model_a_checkpoint": args.model_a_checkpoint,
        "model_b_checkpoint": args.model_b_checkpoint,
        "postprocess_source": args.postprocess_source,
        "best": results[0],
        "results": results,
    }
    out = Path(args.out)
    ensure_dir(out.parent)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["best"]["summary"]["summary"], indent=2))
    print(
        f"Best weights: model_a={output['best']['weight_a']:.3f}, "
        f"model_b={output['best']['weight_b']:.3f}"
    )
    print(f"Saved ensemble weight search: {out}")


if __name__ == "__main__":
    main()
