import numpy as np


def rle_decode(rle: str | float | None, shape: tuple[int, int]) -> np.ndarray:
    """Decode Kaggle RLE string into a binary mask with shape (height, width)."""
    height, width = shape
    mask = np.zeros(height * width, dtype=np.uint8)
    if rle is None or (isinstance(rle, float) and np.isnan(rle)):
        return mask.reshape((height, width), order="C")

    rle = str(rle).strip()
    if not rle:
        return mask.reshape((height, width), order="C")

    values = np.asarray(rle.split(), dtype=np.int64)
    starts = values[0::2] - 1
    lengths = values[1::2]
    ends = starts + lengths
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    return mask.reshape((height, width), order="C")


def rle_encode(mask: np.ndarray) -> str:
    """Encode a binary mask using Kaggle RLE format."""
    pixels = np.asarray(mask, dtype=np.uint8).flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(str(x) for x in runs)


def decode_multiclass(
    rows,
    shape: tuple[int, int],
    classes: tuple[str, ...] = ("large_bowel", "small_bowel", "stomach"),
) -> np.ndarray:
    """Decode grouped dataframe rows into a CxHxW mask."""
    masks = np.zeros((len(classes), shape[0], shape[1]), dtype=np.uint8)
    by_class = {row["class"]: row["segmentation"] for _, row in rows.iterrows()}
    for idx, cls in enumerate(classes):
        masks[idx] = rle_decode(by_class.get(cls), shape)
    return masks
