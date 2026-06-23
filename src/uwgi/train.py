import argparse
import copy
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .dataset import CLASSES, UWGIDataset, build_metadata
from .losses import segmentation_loss
from .metrics import dice_score
from .models import build_model
from .postprocess import postprocess_slice
from .utils import ensure_dir, get_device, load_yaml, seed_everything


def split_outputs(outputs):
    if isinstance(outputs, tuple):
        return outputs[0], outputs[1]
    return outputs, None


def compute_configured_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    cls_logits: torch.Tensor | None,
    cls_targets: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    loss = segmentation_loss(logits, masks, cfg["loss"])
    if cls_logits is not None and cfg["loss"].get("cls_weight", 0.0) > 0:
        loss = loss + cfg["loss"]["cls_weight"] * F.binary_cross_entropy_with_logits(cls_logits, cls_targets)
    return loss


def update_ema(model, ema_model, decay: float) -> None:
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, ema_model=None):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    accumulation_steps = int(cfg["train"].get("gradient_accumulation_steps", 1))
    optimizer.zero_grad(set_to_none=True)
    num_batches = len(loader)
    for step_idx, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        cls_targets = batch["cls"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=cfg["train"]["amp"] and device.type == "cuda"):
            logits, cls_logits = split_outputs(model(images))
            loss = compute_configured_loss(logits, masks, cls_logits, cls_targets, cfg)
            scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()
        if step_idx % accumulation_steps == 0 or step_idx == num_batches:
            scaler.step(optimizer)
            scaler.update()
            if ema_model is not None:
                update_ema(model, ema_model, cfg["train"]["ema"].get("decay", 0.999))
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * images.size(0)
        total_dice += dice_score(logits.detach(), masks).item() * images.size(0)
    n = len(loader.dataset)
    return {"loss": total_loss / n, "dice": total_dice / n}


@torch.no_grad()
def validate(model, loader, device, cfg):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_postprocess_dice = 0.0
    use_postprocess_eval = cfg["train"].get("postprocess_eval", {}).get("enabled", False)
    for batch in tqdm(loader, desc="valid", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        cls_targets = batch["cls"].to(device, non_blocking=True)
        logits, cls_logits = split_outputs(model(images))
        loss = compute_configured_loss(logits, masks, cls_logits, cls_targets, cfg)
        total_loss += loss.item() * images.size(0)
        total_dice += dice_score(logits, masks).item() * images.size(0)
        if use_postprocess_eval:
            total_postprocess_dice += postprocess_dice_score(logits, cls_logits, masks, cfg) * images.size(0)
    n = len(loader.dataset)
    metrics = {"loss": total_loss / n, "dice": total_dice / n}
    if use_postprocess_eval:
        metrics["postprocess_dice"] = total_postprocess_dice / n
    return metrics


def dice_numpy(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    intersection = float((pred * target).sum())
    denominator = float(pred.sum() + target.sum())
    if denominator == 0:
        return 1.0
    return (2.0 * intersection + eps) / (denominator + eps)


def postprocess_dice_score(logits: torch.Tensor, cls_logits: torch.Tensor | None, masks: torch.Tensor, cfg: dict) -> float:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    targets = masks.detach().cpu().numpy().astype(np.uint8)
    cls_probs = None
    if cls_logits is not None:
        cls_probs = torch.sigmoid(cls_logits).detach().cpu().numpy()
    post_cfg = cfg.get("postprocess", {})
    scores = []
    for idx in range(probs.shape[0]):
        pred = postprocess_slice(
            probs[idx],
            None if cls_probs is None else cls_probs[idx],
            mask_thresholds=post_cfg.get("mask_thresholds", 0.5),
            cls_thresholds=post_cfg.get("cls_thresholds", 0.5),
            min_area=post_cfg.get("min_area", 0),
        )
        for channel in range(pred.shape[0]):
            scores.append(dice_numpy(pred[channel], targets[idx, channel]))
    return float(np.mean(scores))


def load_or_build_metadata(cfg: dict) -> pd.DataFrame:
    data_root = Path(cfg["data"]["root"])
    metadata_path = data_root / "metadata_folds.csv"
    if metadata_path.exists():
        meta = pd.read_csv(metadata_path)
        required_class_columns = {f"has_{name}" for name in CLASSES}
        if required_class_columns.issubset(meta.columns):
            return meta
        labels = pd.read_csv(data_root / cfg["data"]["train_csv"])
        class_presence = (
            labels.assign(is_positive=labels["segmentation"].notna().astype(int))
            .pivot_table(index="id", columns="class", values="is_positive", aggfunc="max", fill_value=0)
            .reset_index()
        )
        class_presence = class_presence.rename(columns={name: f"has_{name}" for name in CLASSES})
        enriched = meta.merge(class_presence[["id", *required_class_columns]], on="id", how="left")
        for column in required_class_columns:
            enriched[column] = enriched[column].fillna(0).astype(int)
        enriched.to_csv(metadata_path, index=False)
        return enriched
    meta = build_metadata(data_root, cfg["data"]["train_csv"], cfg["data"]["num_folds"])
    meta.to_csv(metadata_path, index=False)
    return meta


def filter_metadata(meta: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all_slices":
        return meta.reset_index(drop=True)
    if mode == "positive_slices_only":
        return meta[meta["has_mask"] == 1].reset_index(drop=True)
    if mode in ("balanced_positive_negative", "organ_balanced"):
        return meta.reset_index(drop=True)
    raise ValueError(f"Unknown sampling mode: {mode}")


def limit_metadata(meta: pd.DataFrame, limit: int | None, stratified: bool = True) -> pd.DataFrame:
    if not limit:
        return meta.reset_index(drop=True)
    limit = int(limit)
    if not stratified or "has_mask" not in meta.columns:
        return meta.head(limit).reset_index(drop=True)
    pos = meta[meta["has_mask"] == 1]
    neg = meta[meta["has_mask"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return meta.head(limit).reset_index(drop=True)
    num_pos = min(len(pos), max(1, limit // 2))
    num_neg = min(len(neg), limit - num_pos)
    limited = pd.concat([pos.head(num_pos), neg.head(num_neg)], axis=0)
    if len(limited) < limit:
        rest = meta.drop(limited.index).head(limit - len(limited))
        limited = pd.concat([limited, rest], axis=0)
    return limited.sample(frac=1.0, random_state=42).reset_index(drop=True)


def build_sampler(meta: pd.DataFrame, mode: str):
    if mode == "organ_balanced":
        class_cols = [f"has_{name}" for name in CLASSES]
        missing = [col for col in class_cols if col not in meta.columns]
        if missing:
            raise ValueError(f"organ_balanced sampling requires metadata columns: {missing}")
        empty = (meta["has_mask"].astype(int) == 0).to_numpy()
        empty_count = max(int(empty.sum()), 1)
        class_counts = {col: max(int(meta[col].astype(int).sum()), 1) for col in class_cols}
        weights = []
        for _, row in meta.iterrows():
            if int(row["has_mask"]) == 0:
                weights.append(1.0 / empty_count)
                continue
            present = [col for col in class_cols if int(row[col]) == 1]
            weights.append(sum(1.0 / class_counts[col] for col in present) / max(len(present), 1))
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    if mode != "balanced_positive_negative":
        return None
    pos = meta["has_mask"].astype(int).to_numpy()
    num_pos = max(int(pos.sum()), 1)
    num_neg = max(int((1 - pos).sum()), 1)
    weights = [0.5 / num_pos if value == 1 else 0.5 / num_neg for value in pos]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def build_scheduler(optimizer, cfg: dict):
    scheduler_name = cfg["train"].get("scheduler")
    if scheduler_name in (None, "none"):
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["train"]["epochs"],
            eta_min=cfg["train"].get("min_lr", 1e-6),
        )
    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def append_metrics(output_dir: Path, row: dict) -> None:
    metrics_path = output_dir / "metrics.csv"
    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "train_dice",
        "valid_loss",
        "valid_dice",
        "valid_postprocess_dice",
        "best_dice",
        "best_postprocess_dice",
    ]
    write_header = not metrics_path.exists()
    with open(metrics_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_checkpoint(
    path: Path,
    epoch: int,
    cfg: dict,
    model,
    ema_model,
    optimizer,
    scheduler,
    scaler,
    valid_dice: float,
    best_dice: float,
    valid_postprocess_dice: float | None = None,
    best_postprocess_dice: float | None = None,
) -> None:
    ckpt = {
        "epoch": epoch,
        "config": cfg,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict() if ema_model is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "valid_dice": valid_dice,
        "best_dice": best_dice,
        "valid_postprocess_dice": valid_postprocess_dice,
        "best_postprocess_dice": best_postprocess_dice,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str | Path, model, ema_model, optimizer, scheduler, scaler, device) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if ema_model is not None and ckpt.get("ema_model") is not None:
        ema_model.load_state_dict(ckpt["ema_model"])
    if ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)) + 1, float(ckpt.get("best_dice", ckpt.get("valid_dice", -1.0)))


def init_model_weights(path: str | Path, model, device, use_ema: bool = True) -> None:
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("ema_model") if use_ema else None
    if state is None:
        state = ckpt["model"]
    model.load_state_dict(state)
    print(f"Initialized model weights from {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unet_baseline.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(cfg["seed"])
    output_dir = ensure_dir(cfg["train"]["output_dir"])
    device = get_device(cfg["train"]["device"])

    metadata = load_or_build_metadata(cfg)
    valid_fold = cfg["data"]["valid_fold"]
    train_meta = metadata[metadata["fold"] != valid_fold].reset_index(drop=True)
    valid_meta = metadata[metadata["fold"] == valid_fold].reset_index(drop=True)
    sampling_mode = cfg["data"].get("sampling", "all_slices")
    train_meta = filter_metadata(train_meta, sampling_mode)
    limit_strategy = cfg["data"].get("limit_strategy", "stratified")
    stratified_limit = limit_strategy == "stratified"
    train_meta = limit_metadata(train_meta, cfg["data"].get("limit_train_samples"), stratified=stratified_limit)
    valid_meta = limit_metadata(valid_meta, cfg["data"].get("limit_valid_samples"), stratified=stratified_limit)

    train_ds = UWGIDataset(
        cfg["data"]["root"],
        train_meta,
        cfg["data"]["train_csv"],
        image_size=cfg["data"]["image_size"],
        slice_window=cfg["data"].get("slice_window", cfg["model"]["in_channels"]),
        crop_mode=cfg["data"].get("crop_mode", "none"),
        crop_margin=cfg["data"].get("crop_margin", 12),
        center_crop_ratio=cfg["data"].get("center_crop_ratio", 0.9),
        augment=True,
        augmentation=cfg["data"].get("augmentation"),
        normalization=cfg["data"].get("normalization"),
    )
    valid_ds = UWGIDataset(
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

    train_sampler = build_sampler(train_meta, sampling_mode)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        cfg["model"]["name"],
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        encoder_weights=cfg["model"].get("encoder_weights"),
        classification_head=cfg["model"].get("classification_head", False),
    ).to(device)
    if cfg["train"].get("init_from_checkpoint"):
        init_model_weights(
            cfg["train"]["init_from_checkpoint"],
            model,
            device,
            use_ema=cfg["train"].get("init_from_ema", True),
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda")
    ema_model = None
    if cfg["train"].get("ema", {}).get("enabled", False):
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)

    best_dice = -1.0
    best_postprocess_dice = -1.0
    start_epoch = 1
    resume_path = args.resume or cfg["train"].get("resume_from_checkpoint")
    if resume_path:
        start_epoch, best_dice = load_checkpoint(resume_path, model, ema_model, optimizer, scheduler, scaler, device)
        print(f"Resumed from {resume_path}: start_epoch={start_epoch}, best_dice={best_dice:.4f}")

    patience = cfg["train"].get("early_stopping_patience")
    bad_epochs = 0
    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, ema_model=ema_model)
        eval_model = ema_model if ema_model is not None else model
        valid_metrics = validate(eval_model, valid_loader, device, cfg)
        lr = current_lr(optimizer)
        if scheduler is not None:
            scheduler.step()
        print(
            f"epoch={epoch:03d} "
            f"lr={lr:.6g} "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} valid_dice={valid_metrics['dice']:.4f}"
            + (
                f" valid_postprocess_dice={valid_metrics['postprocess_dice']:.4f}"
                if "postprocess_dice" in valid_metrics
                else ""
            )
        )
        append_metrics(
            output_dir,
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "valid_loss": valid_metrics["loss"],
                "valid_dice": valid_metrics["dice"],
                "valid_postprocess_dice": valid_metrics.get("postprocess_dice"),
                "best_dice": max(best_dice, valid_metrics["dice"]),
                "best_postprocess_dice": (
                    max(best_postprocess_dice, valid_metrics["postprocess_dice"])
                    if "postprocess_dice" in valid_metrics
                    else ""
                ),
            },
        )
        valid_postprocess_dice = valid_metrics.get("postprocess_dice")
        save_checkpoint(
            output_dir / "last.pt",
            epoch,
            cfg,
            model,
            ema_model,
            optimizer,
            scheduler,
            scaler,
            valid_metrics["dice"],
            max(best_dice, valid_metrics["dice"]),
            valid_postprocess_dice=valid_postprocess_dice,
            best_postprocess_dice=(
                max(best_postprocess_dice, valid_postprocess_dice)
                if valid_postprocess_dice is not None
                else None
            ),
        )
        if valid_metrics["dice"] > best_dice:
            best_dice = valid_metrics["dice"]
            save_checkpoint(
                output_dir / "best.pt",
                epoch,
                cfg,
                model,
                ema_model,
                optimizer,
                scheduler,
                scaler,
                valid_metrics["dice"],
                best_dice,
                valid_postprocess_dice=valid_postprocess_dice,
                best_postprocess_dice=(
                    max(best_postprocess_dice, valid_postprocess_dice)
                    if valid_postprocess_dice is not None
                    else None
                ),
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
        if valid_postprocess_dice is not None and valid_postprocess_dice > best_postprocess_dice:
            best_postprocess_dice = valid_postprocess_dice
            save_checkpoint(
                output_dir / "best_postprocess.pt",
                epoch,
                cfg,
                model,
                ema_model,
                optimizer,
                scheduler,
                scaler,
                valid_metrics["dice"],
                best_dice,
                valid_postprocess_dice=valid_postprocess_dice,
                best_postprocess_dice=best_postprocess_dice,
            )
        if patience is not None and bad_epochs >= int(patience):
            print(f"Early stopping: no improvement for {bad_epochs} epochs.")
            break


if __name__ == "__main__":
    main()
