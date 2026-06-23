import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.uwgi.dataset import CLASSES, UWGIDataset
from src.uwgi.inference import predict_logits
from src.uwgi.models import build_model
from src.uwgi.postprocess import apply_classification_gate, remove_small_components_multiclass, threshold_masks
from src.uwgi.train import load_or_build_metadata
from src.uwgi.utils import ensure_dir, get_device, load_yaml


def dice_numpy(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    inter = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    return float((2.0 * inter + eps) / (denom + eps))


@torch.no_grad()
def collect_predictions(cfg: dict, checkpoint: Path, device) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    metadata = load_or_build_metadata(cfg)
    valid_fold = cfg["data"]["valid_fold"]
    valid_meta = metadata[metadata["fold"] == valid_fold].reset_index(drop=True)
    if cfg["data"].get("limit_valid_samples"):
        valid_meta = valid_meta.head(int(cfg["data"]["limit_valid_samples"])).reset_index(drop=True)

    dataset = UWGIDataset(
        cfg["data"]["root"],
        valid_meta,
        cfg["data"]["train_csv"],
        image_size=cfg["data"]["image_size"],
        slice_window=cfg["data"].get("slice_window", cfg["model"]["in_channels"]),
        crop_mode=cfg["data"].get("crop_mode", "none"),
        crop_margin=cfg["data"].get("crop_margin", 12),
        center_crop_ratio=cfg["data"].get("center_crop_ratio", 0.9),
        augment=False,
        normalization=cfg["data"].get("normalization"),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("inference", {}).get("batch_size", cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        cfg["model"]["name"],
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        encoder_weights=None,
        classification_head=cfg["model"].get("classification_head", False),
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("ema_model") or ckpt["model"]
    model.load_state_dict(state)
    model.eval()

    all_probs, all_masks, all_cls = [], [], []
    for batch in tqdm(loader, desc="predict", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        logits, cls_logits = predict_logits(model, images, tta=cfg.get("inference", {}).get("tta", False))
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_masks.append(batch["mask"].numpy())
        if cls_logits is not None:
            all_cls.append(torch.sigmoid(cls_logits).cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    cls_probs = np.concatenate(all_cls, axis=0) if all_cls else None
    return probs, masks, cls_probs


def search_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    cls_probs: np.ndarray | None,
    mask_grid: list[float],
    cls_grid: list[float],
    min_area: int,
) -> dict:
    if min_area <= 0:
        return search_thresholds_fast(probs, targets, cls_probs, mask_grid, cls_grid)

    results = {}
    for channel, cls_name in enumerate(CLASSES):
        best = {"dice": -1.0, "mask_threshold": 0.5, "cls_threshold": None}
        for mask_thr in mask_grid:
            cls_thresholds = [0.0, 0.0, 0.0]
            candidate_cls_grid = cls_grid if cls_probs is not None else [None]
            for cls_thr in candidate_cls_grid:
                if cls_thr is not None:
                    cls_thresholds[channel] = cls_thr
                dices = []
                for idx in range(probs.shape[0]):
                    prob = probs[idx].copy()
                    if cls_thr is not None:
                        prob = apply_classification_gate(prob, cls_probs[idx], cls_thresholds)
                    pred = threshold_masks(prob, [0.0, 0.0, 0.0])
                    pred[channel] = (prob[channel] > mask_thr).astype(np.uint8)
                    pred = remove_small_components_multiclass(pred, min_area)
                    dices.append(dice_numpy(pred[channel], targets[idx, channel]))
                score = float(np.mean(dices))
                if score > best["dice"]:
                    best = {"dice": score, "mask_threshold": float(mask_thr), "cls_threshold": None if cls_thr is None else float(cls_thr)}
        results[cls_name] = best
    results["mean_dice"] = float(np.mean([value["dice"] for value in results.values()]))
    return results


def search_thresholds_fast(
    probs: np.ndarray,
    targets: np.ndarray,
    cls_probs: np.ndarray | None,
    mask_grid: list[float],
    cls_grid: list[float],
) -> dict:
    results = {}
    for channel, cls_name in enumerate(CLASSES):
        target = targets[:, channel] > 0.5
        target_sum = target.reshape(target.shape[0], -1).sum(axis=1)
        best = {"dice": -1.0, "mask_threshold": 0.5, "cls_threshold": None}
        candidate_cls_grid = cls_grid if cls_probs is not None else [None]
        for mask_thr in mask_grid:
            pred = probs[:, channel] > mask_thr
            pred_sum = pred.reshape(pred.shape[0], -1).sum(axis=1)
            intersection = (pred & target).reshape(pred.shape[0], -1).sum(axis=1)
            for cls_thr in candidate_cls_grid:
                if cls_thr is None:
                    gated_pred_sum = pred_sum
                    gated_intersection = intersection
                else:
                    keep = cls_probs[:, channel] >= cls_thr
                    gated_pred_sum = pred_sum * keep
                    gated_intersection = intersection * keep
                denom = gated_pred_sum + target_sum
                dice = np.where(denom > 0, (2.0 * gated_intersection + 1e-7) / (denom + 1e-7), 1.0)
                score = float(dice.mean())
                if score > best["dice"]:
                    best = {
                        "dice": score,
                        "mask_threshold": float(mask_thr),
                        "cls_threshold": None if cls_thr is None else float(cls_thr),
                    }
        results[cls_name] = best
    results["mean_dice"] = float(np.mean([value["dice"] for value in results.values()]))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--mask-grid", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--cls-grid", default="0.30,0.40,0.50,0.60,0.70")
    parser.add_argument("--min-area", type=int, default=24)
    parser.add_argument("--disable-cls-gate", action="store_true", help="Search only mask thresholds even if the model has a classification head.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = get_device(cfg["train"]["device"])
    probs, targets, cls_probs = collect_predictions(cfg, Path(args.checkpoint), device)
    if args.disable_cls_gate:
        cls_probs = None
    mask_grid = [float(x) for x in args.mask_grid.split(",") if x]
    cls_grid = [float(x) for x in args.cls_grid.split(",") if x]
    results = search_thresholds(probs, targets, cls_probs, mask_grid, cls_grid, args.min_area)

    out = Path(args.out) if args.out else Path(cfg["train"]["output_dir"]) / "thresholds.json"
    ensure_dir(out.parent)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved thresholds: {out}")


if __name__ == "__main__":
    main()
