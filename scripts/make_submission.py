import argparse
import glob
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_oof_postprocess import apply_3d_components, apply_z_continuity
from src.uwgi.dataset import CLASSES, parse_id, parse_scan_filename
from src.uwgi.inference import predict_logits
from src.uwgi.models import build_model
from src.uwgi.postprocess import postprocess_slice
from src.uwgi.rle import rle_encode
from src.uwgi.utils import ensure_dir, get_device, load_yaml


def extract_image_id(value: str) -> str:
    text = str(value)
    for cls_name in CLASSES:
        suffix = f"_{cls_name}"
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def find_scan_path(data_root: Path, image_id: str, split: str) -> Path:
    case, day, slice_id = parse_id(image_id)
    candidates = []
    if split != "auto":
        candidates.append(split)
    else:
        candidates.extend(["test", "train"])
    for candidate_split in candidates:
        scan_dir = data_root / candidate_split / case / f"{case}_{day}" / "scans"
        matches = sorted(scan_dir.glob(f"slice_{slice_id:04d}_*.png"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No scan found for {image_id} under {data_root} split={split}")


def build_submission_metadata(sample: pd.DataFrame, data_root: Path, split: str) -> pd.DataFrame:
    image_ids = [extract_image_id(value) for value in sample["id"].drop_duplicates()]
    rows = []
    for image_id in image_ids:
        scan_path = find_scan_path(data_root, image_id, split)
        height, width, spacing_h, spacing_w = parse_scan_filename(scan_path)
        case, day, slice_id = parse_id(image_id)
        rows.append(
            {
                "id": image_id,
                "case": case,
                "day": day,
                "slice": slice_id,
                "image_path": str(scan_path),
                "height": height,
                "width": width,
                "spacing_h": spacing_h,
                "spacing_w": spacing_w,
            }
        )
    return pd.DataFrame(rows)


class SubmissionDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        image_size: int,
        slice_window: int,
        normalization: dict | None = None,
    ):
        if slice_window < 1 or slice_window % 2 == 0:
            raise ValueError("slice_window must be a positive odd integer.")
        self.metadata = metadata.reset_index(drop=True)
        self.image_size = image_size
        self.slice_window = slice_window
        self.normalization = normalization or {}
        self._scan_cache: dict[str, dict[int, Path]] = {}
        self._volume_stats_cache: dict[str, tuple[float, float]] = {}

    def __len__(self) -> int:
        return len(self.metadata)

    def _scan_files(self, scan_dir: str | Path) -> dict[int, Path]:
        scan_dir = str(scan_dir)
        if scan_dir not in self._scan_cache:
            files = {}
            for path in Path(scan_dir).glob("slice_*.png"):
                match = re.match(r"^slice_(\d+)_", path.name)
                if match:
                    files[int(match.group(1))] = path
            if not files:
                raise FileNotFoundError(f"No slice images found under {scan_dir}")
            self._scan_cache[scan_dir] = files
        return self._scan_cache[scan_dir]

    def _volume_stats(self, scan_dir: str | Path) -> tuple[float, float]:
        scan_dir = str(scan_dir)
        if scan_dir not in self._volume_stats_cache:
            lower_percentile = float(self.normalization.get("lower_percentile", 1.0))
            upper_percentile = float(self.normalization.get("upper_percentile", 99.0))
            sample_stride = max(int(self.normalization.get("sample_stride", 1)), 1)
            values = []
            files = self._scan_files(scan_dir)
            for idx, slice_id in enumerate(sorted(files)):
                if idx % sample_stride != 0:
                    continue
                image = cv2.imread(str(files[slice_id]), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise FileNotFoundError(files[slice_id])
                values.append(image.reshape(-1))
            pixels = np.concatenate(values).astype(np.float32)
            lower = float(np.percentile(pixels, lower_percentile))
            upper = float(np.percentile(pixels, upper_percentile))
            if upper <= lower:
                lower = float(pixels.min())
                upper = float(pixels.max())
            self._volume_stats_cache[scan_dir] = (lower, upper)
        return self._volume_stats_cache[scan_dir]

    def _load_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(path)
        image = image.astype(np.float32)
        if self.normalization.get("mode") == "volume_percentile":
            lower, upper = self._volume_stats(path.parent)
            image = np.clip(image, lower, upper)
            denominator = upper - lower
            return ((image - lower) / denominator).astype(np.float32) if denominator > 0 else image * 0.0
        max_value = image.max()
        return image / max_value if max_value > 0 else image

    def _neighbor_path(self, row: pd.Series, offset: int) -> Path:
        scan_dir = Path(row.image_path).parent
        files = self._scan_files(scan_dir)
        target = int(row.slice) + offset
        if target in files:
            return files[target]
        closest = min(files, key=lambda key: abs(key - target))
        return files[closest]

    def __getitem__(self, index: int) -> dict:
        row = self.metadata.iloc[index]
        half = self.slice_window // 2
        images = []
        for offset in range(-half, half + 1):
            image = self._load_image(self._neighbor_path(row, offset))
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            images.append(image)
        stack = np.stack(images, axis=0).astype(np.float32)
        return {"image": torch.from_numpy(stack), "id": row.id}


def load_model(cfg: dict, checkpoint: Path, device: torch.device) -> torch.nn.Module:
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
    return model


@torch.no_grad()
def predict_ensemble(config_paths: list[Path], checkpoint_name: str, metadata: pd.DataFrame, device: torch.device):
    first_cfg = load_yaml(config_paths[0])
    dataset = SubmissionDataset(
        metadata,
        image_size=int(first_cfg["data"]["image_size"]),
        slice_window=int(first_cfg["data"].get("slice_window", first_cfg["model"]["in_channels"])),
        normalization=first_cfg["data"].get("normalization"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(first_cfg.get("inference", {}).get("batch_size", first_cfg["train"]["batch_size"])),
        shuffle=False,
        num_workers=int(first_cfg["data"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    prob_sum = None
    cls_sum = None
    for config_path in config_paths:
        cfg = load_yaml(config_path)
        checkpoint = Path(cfg["train"]["output_dir"]) / checkpoint_name
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        model = load_model(cfg, checkpoint, device)
        fold_probs, fold_cls = [], []
        for batch in tqdm(loader, desc=f"predict {config_path.stem}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            logits, cls_logits = predict_logits(model, images, tta=cfg.get("inference", {}).get("tta", False))
            fold_probs.append(torch.sigmoid(logits).cpu().numpy())
            if cls_logits is not None:
                fold_cls.append(torch.sigmoid(cls_logits).cpu().numpy())
        probs = np.concatenate(fold_probs, axis=0)
        cls_probs = np.concatenate(fold_cls, axis=0) if fold_cls else None
        prob_sum = probs if prob_sum is None else prob_sum + probs
        if cls_probs is not None:
            cls_sum = cls_probs if cls_sum is None else cls_sum + cls_probs
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    probs = prob_sum / len(config_paths)
    cls_probs = cls_sum / len(config_paths) if cls_sum is not None else None
    return probs, cls_probs, first_cfg


def postprocess_predictions(probs: np.ndarray, cls_probs: np.ndarray | None, metadata: pd.DataFrame, cfg: dict) -> np.ndarray:
    post_cfg = cfg.get("postprocess", {})
    masks = []
    for idx in range(probs.shape[0]):
        masks.append(
            postprocess_slice(
                probs[idx],
                cls_probs=cls_probs[idx] if cls_probs is not None else None,
                mask_thresholds=post_cfg.get("mask_thresholds", 0.5),
                cls_thresholds=post_cfg.get("cls_thresholds", 0.5),
                min_area=post_cfg.get("min_area", 0),
            )
        )
    masks = np.stack(masks, axis=0)
    masks = apply_z_continuity(masks, metadata, post_cfg.get("z_min_run", 1))
    masks = apply_3d_components(
        masks,
        metadata,
        min_volume=post_cfg.get("min_volume", 0),
        keep_largest=post_cfg.get("keep_largest_component", False),
        connectivity=int(post_cfg.get("component_connectivity", 1)),
    )
    return masks


def build_submission(sample: pd.DataFrame, metadata: pd.DataFrame, masks: np.ndarray) -> pd.DataFrame:
    mask_by_id = {image_id: masks[idx] for idx, image_id in enumerate(metadata["id"].tolist())}
    shape_by_id = {
        row.id: (int(row.height), int(row.width))
        for row in metadata.itertuples(index=False)
    }
    rows = []
    for _, row in sample.iterrows():
        image_id = extract_image_id(row["id"])
        class_name = row["class"]
        channel = CLASSES.index(class_name)
        height, width = shape_by_id[image_id]
        mask = mask_by_id[image_id][channel]
        resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        rows.append({"id": row["id"], "class": class_name, "predicted": rle_encode(resized)})
    return pd.DataFrame(rows, columns=["id", "class", "predicted"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-config-glob", required=True)
    parser.add_argument("--checkpoint-name", default="best_postprocess.pt")
    parser.add_argument("--sample-submission", default="data/raw/uw-madison-gi-tract-image-segmentation/sample_submission.csv")
    parser.add_argument("--data-root", default="data/raw/uw-madison-gi-tract-image-segmentation")
    parser.add_argument("--split", default="auto", choices=["auto", "test", "train"])
    parser.add_argument("--out", default="outputs/submissions/strategy_e_minarea_z_submission.csv")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    sample_path = Path(args.sample_submission)
    sample = pd.read_csv(sample_path)
    out = Path(args.out)
    ensure_dir(out.parent)
    manifest_path = Path(args.manifest) if args.manifest else out.with_suffix(".manifest.json")

    config_paths = sorted(Path(path) for path in glob.glob(args.fold_config_glob))
    if not config_paths:
        raise FileNotFoundError(f"No configs matched {args.fold_config_glob}")

    manifest = {
        "sample_submission": str(sample_path),
        "data_root": args.data_root,
        "fold_configs": [str(path) for path in config_paths],
        "checkpoint_name": args.checkpoint_name,
        "output": str(out),
        "num_sample_rows": int(len(sample)),
        "status": "not_started",
    }

    if sample.empty:
        pd.DataFrame(columns=["id", "class", "predicted"]).to_csv(out, index=False)
        manifest["status"] = "empty_sample_submission"
        manifest["num_images"] = 0
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Sample submission is empty; wrote empty submission: {out}")
        print(f"Saved manifest: {manifest_path}")
        return

    if set(sample.columns) != {"id", "class", "predicted"}:
        raise ValueError(f"Unexpected sample columns: {sample.columns.tolist()}")

    metadata = build_submission_metadata(sample, Path(args.data_root), args.split)
    device = get_device(args.device)
    probs, cls_probs, cfg = predict_ensemble(config_paths, args.checkpoint_name, metadata, device)
    masks = postprocess_predictions(probs, cls_probs, metadata, cfg)
    submission = build_submission(sample, metadata, masks)
    submission.to_csv(out, index=False)

    manifest.update(
        {
            "status": "completed",
            "num_images": int(len(metadata)),
            "num_non_empty_predictions": int((submission["predicted"].fillna("") != "").sum()),
            "postprocess": cfg.get("postprocess", {}),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved submission: {out}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
