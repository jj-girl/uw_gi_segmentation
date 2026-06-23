import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from .dataset import UWGIDataset, build_metadata
from .inference import predict_logits
from .models import build_model
from .postprocess import postprocess_slice
from .utils import get_device, load_yaml


COLORS = np.array(
    [
        [255, 64, 64],
        [64, 220, 120],
        [64, 128, 255],
    ],
    dtype=np.float32,
)


def colorize(mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.float32)
    for channel, color in zip(mask, COLORS):
        canvas[channel > 0.5] = color
    return canvas / 255.0


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unet_baseline.yaml")
    parser.add_argument("--checkpoint", default="outputs/unet_baseline/best.pt")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", default="outputs/preview.png")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = get_device(cfg["train"]["device"])
    metadata = build_metadata(cfg["data"]["root"], cfg["data"]["train_csv"], cfg["data"]["num_folds"])
    valid_meta = metadata[metadata["fold"] == cfg["data"]["valid_fold"]].reset_index(drop=True)
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
    sample = dataset[args.index]

    model = build_model(
        cfg["model"]["name"],
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        encoder_weights=cfg["model"].get("encoder_weights"),
        classification_head=cfg["model"].get("classification_head", False),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("ema_model") or ckpt["model"])
    model.eval()

    image = sample["image"].unsqueeze(0).to(device)
    logits, cls_logits = predict_logits(model, image, tta=cfg.get("inference", {}).get("tta", False))
    pred_probs = torch.sigmoid(logits)[0].cpu().numpy()
    cls_probs = torch.sigmoid(cls_logits)[0].cpu().numpy() if cls_logits is not None else None
    post_cfg = cfg.get("postprocess", {})
    pred = postprocess_slice(
        pred_probs,
        cls_probs=cls_probs,
        mask_thresholds=post_cfg.get("mask_thresholds", 0.5),
        cls_thresholds=post_cfg.get("cls_thresholds", 0.5),
        min_area=post_cfg.get("min_area", 0),
    ).astype(np.float32)
    truth = sample["mask"].numpy()
    base = sample["image"][sample["image"].shape[0] // 2].numpy()
    base_rgb = np.repeat(base[:, :, None], 3, axis=2)

    pred_overlay = np.clip(base_rgb * 0.65 + colorize(pred) * 0.55, 0, 1)
    truth_overlay = np.clip(base_rgb * 0.65 + colorize(truth) * 0.55, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(base, cmap="gray")
    axes[0].set_title(str(sample["id"]))
    axes[1].imshow(truth_overlay)
    axes[1].set_title("Ground truth")
    axes[2].imshow(pred_overlay)
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"Saved preview: {args.out}")


if __name__ == "__main__":
    main()
