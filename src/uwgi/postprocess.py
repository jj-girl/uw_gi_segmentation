from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def apply_classification_gate(
    probs: np.ndarray,
    cls_probs: np.ndarray | None,
    cls_thresholds: list[float] | tuple[float, ...] | float = 0.5,
) -> np.ndarray:
    """Suppress organ probability maps whose classification score is too low."""
    if cls_probs is None:
        return probs
    thresholds = np.asarray(cls_thresholds, dtype=np.float32)
    if thresholds.ndim == 0:
        thresholds = np.repeat(thresholds, probs.shape[0])
    gated = probs.copy()
    for channel, threshold in enumerate(thresholds):
        if cls_probs[channel] < threshold:
            gated[channel] = 0
    return gated


def threshold_masks(
    probs: np.ndarray,
    thresholds: list[float] | tuple[float, ...] | float = 0.5,
) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.ndim == 0:
        thresholds = np.repeat(thresholds, probs.shape[0])
    masks = np.zeros_like(probs, dtype=np.uint8)
    for channel, threshold in enumerate(thresholds):
        masks[channel] = (probs[channel] > threshold).astype(np.uint8)
    return masks


def remove_small_components(mask: np.ndarray, min_area: int = 32) -> np.ndarray:
    """Remove small connected components from a single binary mask."""
    if min_area <= 0:
        return mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1
    return cleaned


def remove_small_components_multiclass(masks: np.ndarray, min_area: int | list[int] | tuple[int, ...] = 32) -> np.ndarray:
    min_areas = np.asarray(min_area, dtype=np.int32)
    if min_areas.ndim == 0:
        min_areas = np.repeat(min_areas, masks.shape[0])
    cleaned = np.zeros_like(masks, dtype=np.uint8)
    for channel, area in enumerate(min_areas):
        cleaned[channel] = remove_small_components(masks[channel], int(area))
    return cleaned


def enforce_z_continuity(volume_masks: np.ndarray, min_run: int = 2) -> np.ndarray:
    """Remove isolated positive slices in a CxZxHxW volume mask."""
    if min_run <= 1:
        return volume_masks.astype(np.uint8)
    cleaned = volume_masks.copy().astype(np.uint8)
    for channel in range(cleaned.shape[0]):
        positive = cleaned[channel].reshape(cleaned.shape[1], -1).sum(axis=1) > 0
        start = None
        for idx, value in enumerate(np.r_[positive, False]):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                end = idx
                if end - start < min_run:
                    cleaned[channel, start:end] = 0
                start = None
    return cleaned


def remove_small_components_3d(
    volume_mask: np.ndarray,
    min_volume: int = 0,
    keep_largest: bool = False,
    connectivity: int = 1,
) -> np.ndarray:
    """Filter connected components in a ZxHxW binary volume."""
    volume = volume_mask.astype(np.uint8)
    if min_volume <= 0 and not keep_largest:
        return volume
    structure = ndimage.generate_binary_structure(rank=3, connectivity=connectivity)
    labels, num_labels = ndimage.label(volume, structure=structure)
    if num_labels == 0:
        return volume

    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    keep = np.zeros(num_labels + 1, dtype=bool)
    if keep_largest:
        largest = int(component_sizes.argmax())
        if largest > 0 and component_sizes[largest] >= max(min_volume, 1):
            keep[largest] = True
    else:
        keep = component_sizes >= max(min_volume, 1)
        keep[0] = False
    return keep[labels].astype(np.uint8)


def postprocess_slice(
    probs: np.ndarray,
    cls_probs: np.ndarray | None = None,
    mask_thresholds: list[float] | tuple[float, ...] | float = 0.5,
    cls_thresholds: list[float] | tuple[float, ...] | float = 0.5,
    min_area: int | list[int] | tuple[int, ...] = 32,
) -> np.ndarray:
    probs = apply_classification_gate(probs, cls_probs, cls_thresholds)
    masks = threshold_masks(probs, mask_thresholds)
    return remove_small_components_multiclass(masks, min_area)
