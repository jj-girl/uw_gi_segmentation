from __future__ import annotations

import torch

from .train import split_outputs


@torch.no_grad()
def predict_logits(model, images: torch.Tensor, tta: bool = False):
    """Return segmentation logits and optional classification logits."""
    outputs = model(images)
    logits, cls_logits = split_outputs(outputs)
    if not tta:
        return logits, cls_logits

    flipped = torch.flip(images, dims=[-1])
    flip_outputs = model(flipped)
    flip_logits, flip_cls_logits = split_outputs(flip_outputs)
    flip_logits = torch.flip(flip_logits, dims=[-1])
    logits = (logits + flip_logits) * 0.5
    if cls_logits is not None and flip_cls_logits is not None:
        cls_logits = (cls_logits + flip_cls_logits) * 0.5
    return logits, cls_logits

