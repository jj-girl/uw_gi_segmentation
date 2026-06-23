import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

from .rle import decode_multiclass


CLASSES = ("large_bowel", "small_bowel", "stomach")


def parse_id(image_id: str) -> tuple[str, str, int]:
    match = re.match(r"^(case\d+)_(day\d+)_slice_(\d+)$", image_id)
    if not match:
        raise ValueError(f"Unexpected image id: {image_id}")
    case, day, slice_id = match.groups()
    return case, day, int(slice_id)


def parse_scan_filename(path: Path) -> tuple[int, int, float, float]:
    parts = path.stem.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected scan filename: {path.name}")
    height = int(parts[-4])
    width = int(parts[-3])
    spacing_h = float(parts[-2])
    spacing_w = float(parts[-1])
    return height, width, spacing_h, spacing_w


def find_scan_path(data_root: Path, image_id: str) -> Path:
    case, day, slice_id = parse_id(image_id)
    scan_dir = data_root / "train" / case / f"{case}_{day}" / "scans"
    matches = list(scan_dir.glob(f"slice_{slice_id:04d}_*.png"))
    if not matches:
        raise FileNotFoundError(f"No scan found for {image_id} under {scan_dir}")
    return matches[0]


def build_metadata(data_root: str | Path, train_csv: str = "train.csv", num_folds: int = 5) -> pd.DataFrame:
    data_root = Path(data_root)
    csv_path = data_root / train_csv
    df = pd.read_csv(csv_path)
    required = {"id", "class", "segmentation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    rows = []
    class_presence = (
        df.assign(is_positive=df["segmentation"].notna().astype(int))
        .pivot_table(index="id", columns="class", values="is_positive", aggfunc="max", fill_value=0)
        .reset_index()
    )
    for image_id, group in df.groupby("id", sort=False):
        scan_path = find_scan_path(data_root, image_id)
        height, width, spacing_h, spacing_w = parse_scan_filename(scan_path)
        case, day, slice_id = parse_id(image_id)
        presence_row = class_presence[class_presence["id"] == image_id]
        presence = presence_row.iloc[0].to_dict() if len(presence_row) else {}
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
                "has_mask": int(group["segmentation"].notna().any()),
                **{f"has_{name}": int(presence.get(name, 0)) for name in CLASSES},
            }
        )
    meta = pd.DataFrame(rows)
    meta["fold"] = -1
    splitter = GroupKFold(n_splits=num_folds)
    for fold, (_, valid_idx) in enumerate(splitter.split(meta, groups=meta["case"])):
        meta.loc[valid_idx, "fold"] = fold
    return meta


class UWGIDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        metadata: pd.DataFrame,
        train_csv: str = "train.csv",
        image_size: int = 256,
        slice_window: int = 1,
        crop_mode: str = "none",
        crop_margin: int = 12,
        center_crop_ratio: float = 0.9,
        augment: bool = False,
        augmentation: dict | None = None,
        normalization: dict | None = None,
    ):
        if slice_window < 1 or slice_window % 2 == 0:
            raise ValueError("slice_window must be a positive odd integer.")
        self.data_root = Path(data_root)
        self.metadata = metadata.reset_index(drop=True)
        self.labels = pd.read_csv(self.data_root / train_csv)
        self.image_size = image_size
        self.slice_window = slice_window
        self.crop_mode = crop_mode
        self.crop_margin = crop_margin
        self.center_crop_ratio = center_crop_ratio
        self.augment = augment
        self.augmentation = augmentation or {}
        self.normalization = normalization or {}
        self._scan_cache: dict[str, dict[int, Path]] = {}
        self._volume_stats_cache: dict[str, tuple[float, float]] = {}

    def __len__(self) -> int:
        return len(self.metadata)

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
            if not values:
                raise FileNotFoundError(f"No images found under {scan_dir}")
            pixels = np.concatenate(values).astype(np.float32)
            lower = float(np.percentile(pixels, lower_percentile))
            upper = float(np.percentile(pixels, upper_percentile))
            if upper <= lower:
                lower = float(pixels.min())
                upper = float(pixels.max())
            self._volume_stats_cache[scan_dir] = (lower, upper)
        return self._volume_stats_cache[scan_dir]

    def _load_image(self, path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(path)
        image = image.astype(np.float32)
        mode = self.normalization.get("mode")
        if mode == "volume_percentile":
            lower, upper = self._volume_stats(Path(path).parent)
            image = np.clip(image, lower, upper)
            denominator = upper - lower
            if denominator > 0:
                image = (image - lower) / denominator
            else:
                image = image * 0.0
            return image.astype(np.float32)
        max_value = image.max()
        if max_value > 0:
            image = image / max_value
        return image

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

    def _neighbor_path(self, row: pd.Series, offset: int) -> Path:
        scan_dir = Path(row.image_path).parent
        files = self._scan_files(scan_dir)
        target = int(row.slice) + offset
        if target in files:
            return files[target]
        closest = min(files, key=lambda key: abs(key - target))
        return files[closest]

    def _load_image_stack(self, row: pd.Series) -> np.ndarray:
        half = self.slice_window // 2
        images = []
        for offset in range(-half, half + 1):
            image = self._load_image(str(self._neighbor_path(row, offset)))
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            images.append(image)
        return np.stack(images, axis=0).astype(np.float32)

    def _crop_box(self, image_stack: np.ndarray) -> tuple[int, int, int, int]:
        _, height, width = image_stack.shape
        if self.crop_mode == "none":
            return 0, 0, width, height
        if self.crop_mode == "center":
            crop_h = int(height * self.center_crop_ratio)
            crop_w = int(width * self.center_crop_ratio)
            x1 = max((width - crop_w) // 2, 0)
            y1 = max((height - crop_h) // 2, 0)
            return x1, y1, x1 + crop_w, y1 + crop_h
        if self.crop_mode == "foreground":
            center = image_stack[image_stack.shape[0] // 2]
            threshold = max(float(center.mean() + 0.15 * center.std()), float(np.percentile(center, 55)))
            ys, xs = np.where(center > threshold)
            if len(xs) == 0 or len(ys) == 0:
                return 0, 0, width, height
            x1 = max(int(xs.min()) - self.crop_margin, 0)
            x2 = min(int(xs.max()) + self.crop_margin + 1, width)
            y1 = max(int(ys.min()) - self.crop_margin, 0)
            y2 = min(int(ys.max()) + self.crop_margin + 1, height)
            if x2 <= x1 or y2 <= y1:
                return 0, 0, width, height
            return x1, y1, x2, y2
        raise ValueError(f"Unknown crop_mode: {self.crop_mode}")

    def _apply_crop(self, image_stack: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = self._crop_box(image_stack)
        return image_stack[:, y1:y2, x1:x2], mask[:, y1:y2, x1:x2]

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:
            image = np.ascontiguousarray(np.flip(image, axis=2))
            mask = np.ascontiguousarray(np.flip(mask, axis=2))

        if np.random.rand() < float(self.augmentation.get("vertical_flip_prob", 0.0)):
            image = np.ascontiguousarray(np.flip(image, axis=1))
            mask = np.ascontiguousarray(np.flip(mask, axis=1))

        affine_prob = float(self.augmentation.get("affine_prob", 0.0))
        if affine_prob > 0 and np.random.rand() < affine_prob:
            _, height, width = image.shape
            rotate_limit = float(self.augmentation.get("rotate_limit", 0.0))
            scale_limit = float(self.augmentation.get("scale_limit", 0.0))
            shift_limit = float(self.augmentation.get("shift_limit", 0.0))
            angle = np.random.uniform(-rotate_limit, rotate_limit)
            scale = np.random.uniform(1.0 - scale_limit, 1.0 + scale_limit)
            dx = np.random.uniform(-shift_limit, shift_limit) * width
            dy = np.random.uniform(-shift_limit, shift_limit) * height
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
            matrix[0, 2] += dx
            matrix[1, 2] += dy
            warped_images = [
                cv2.warpAffine(channel, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                for channel in image
            ]
            warped_masks = [
                cv2.warpAffine(channel, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                for channel in mask
            ]
            image = np.stack(warped_images, axis=0).astype(np.float32)
            mask = np.stack(warped_masks, axis=0).astype(np.float32)

        distortion_prob = float(self.augmentation.get("distortion_prob", 0.0))
        if distortion_prob > 0 and np.random.rand() < distortion_prob:
            image, mask = self._sinusoidal_distortion(image, mask)

        if np.random.rand() < 0.5:
            alpha = np.random.uniform(0.9, 1.1)
            beta = np.random.uniform(-0.05, 0.05)
            image = np.clip(image * alpha + beta, 0.0, 1.0)

        gamma_prob = float(self.augmentation.get("gamma_prob", 0.0))
        if gamma_prob > 0 and np.random.rand() < gamma_prob:
            gamma_limit = self.augmentation.get("gamma_limit", [0.8, 1.25])
            gamma = np.random.uniform(float(gamma_limit[0]), float(gamma_limit[1]))
            image = np.clip(image, 0.0, 1.0) ** gamma

        noise_prob = float(self.augmentation.get("noise_prob", 0.0))
        if noise_prob > 0 and np.random.rand() < noise_prob:
            noise_std = float(self.augmentation.get("noise_std", 0.025))
            image = np.clip(image + np.random.normal(0.0, noise_std, size=image.shape).astype(np.float32), 0.0, 1.0)

        blur_prob = float(self.augmentation.get("blur_prob", 0.0))
        if blur_prob > 0 and np.random.rand() < blur_prob:
            kernel_size = int(self.augmentation.get("blur_kernel", 3))
            if kernel_size % 2 == 0:
                kernel_size += 1
            image = np.stack([cv2.GaussianBlur(channel, (kernel_size, kernel_size), 0) for channel in image], axis=0)

        z_dropout_prob = float(self.augmentation.get("z_dropout_prob", 0.0))
        if z_dropout_prob > 0 and image.shape[0] > 1 and np.random.rand() < z_dropout_prob:
            image = self._z_context_dropout(image)

        return image, mask

    def _sinusoidal_distortion(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, height, width = image.shape
        magnitude = float(self.augmentation.get("distortion_magnitude", 6.0))
        min_period = float(self.augmentation.get("distortion_min_period", 80.0))
        max_period = float(self.augmentation.get("distortion_max_period", 160.0))
        period_x = np.random.uniform(min_period, max_period)
        period_y = np.random.uniform(min_period, max_period)
        phase_x = np.random.uniform(0.0, 2.0 * np.pi)
        phase_y = np.random.uniform(0.0, 2.0 * np.pi)
        yy, xx = np.indices((height, width), dtype=np.float32)
        map_x = xx + magnitude * np.sin(2.0 * np.pi * yy / period_y + phase_x)
        map_y = yy + magnitude * np.sin(2.0 * np.pi * xx / period_x + phase_y)
        warped_images = [
            cv2.remap(channel, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            for channel in image
        ]
        warped_masks = [
            cv2.remap(channel, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            for channel in mask
        ]
        return np.stack(warped_images, axis=0).astype(np.float32), np.stack(warped_masks, axis=0).astype(np.float32)

    def _z_context_dropout(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        center_idx = image.shape[0] // 2
        max_drop = int(self.augmentation.get("z_dropout_max_channels", 2))
        candidates = [idx for idx in range(image.shape[0]) if idx != center_idx]
        if not candidates:
            return result
        num_drop = np.random.randint(1, min(max_drop, len(candidates)) + 1)
        for idx in np.random.choice(candidates, size=num_drop, replace=False):
            result[idx] = image[center_idx]
        return result

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.metadata.iloc[index]
        image = self._load_image_stack(row)
        label_rows = self.labels[self.labels["id"] == row.id]
        mask = decode_multiclass(label_rows, (int(row.height), int(row.width)), CLASSES)
        resized_masks = []
        for channel in mask:
            resized = cv2.resize(channel, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            resized_masks.append(resized)
        mask = np.stack(resized_masks, axis=0).astype(np.float32)
        image, mask = self._apply_crop(image, mask)

        resized_images = []
        for channel in image:
            resized = cv2.resize(channel, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            resized_images.append(resized)
        image = np.stack(resized_images, axis=0).astype(np.float32)

        resized_masks = []
        for channel in mask:
            resized = cv2.resize(channel, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            resized_masks.append(resized)
        mask = np.stack(resized_masks, axis=0).astype(np.float32)

        if self.augment:
            image, mask = self._augment(image, mask)
        cls = (mask.reshape(mask.shape[0], -1).max(axis=1) > 0).astype(np.float32)

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        cls_tensor = torch.from_numpy(cls)
        return {"image": image_tensor, "mask": mask_tensor, "cls": cls_tensor, "id": row.id}
