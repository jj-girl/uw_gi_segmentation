import torch
import torch.nn.functional as F

from .metrics import soft_dice_loss


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1 - probs)
    alpha_t = torch.where(targets > 0.5, alpha, 1 - alpha)
    return (alpha_t * (1 - pt).pow(gamma) * bce).mean()


def tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-7,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    dims = (0, 2, 3)
    tp = torch.sum(probs * targets, dims)
    fp = torch.sum(probs * (1 - targets), dims)
    fn = torch.sum((1 - probs) * targets, dims)
    score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1 - score.mean()


def focal_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
) -> torch.Tensor:
    return tversky_loss(logits, targets, alpha=alpha, beta=beta).pow(gamma)


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor, cfg: dict) -> torch.Tensor:
    name = cfg.get("name", "dice_bce")
    bce_weight = cfg.get("bce_weight", 0.5)
    dice_weight = cfg.get("dice_weight", 0.5)
    focal_weight = cfg.get("focal_weight", 0.5)
    tversky_weight = cfg.get("tversky_weight", 1.0)

    if name == "dice_bce":
        return bce_weight * F.binary_cross_entropy_with_logits(logits, targets) + dice_weight * soft_dice_loss(logits, targets)
    if name == "dice_focal":
        return dice_weight * soft_dice_loss(logits, targets) + focal_weight * focal_loss_with_logits(logits, targets)
    if name == "tversky":
        return tversky_weight * tversky_loss(logits, targets, alpha=cfg.get("tversky_alpha", 0.3), beta=cfg.get("tversky_beta", 0.7))
    if name == "focal_tversky":
        return tversky_weight * focal_tversky_loss(
            logits,
            targets,
            alpha=cfg.get("tversky_alpha", 0.3),
            beta=cfg.get("tversky_beta", 0.7),
            gamma=cfg.get("focal_tversky_gamma", 0.75),
        )
    raise ValueError(f"Unknown loss name: {name}")

