import torch


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()
    dims = (2, 3)
    intersection = torch.sum(preds * targets, dims)
    cardinality = torch.sum(preds + targets, dims)
    dice = torch.where(cardinality > 0, (2.0 * intersection + eps) / (cardinality + eps), torch.ones_like(cardinality))
    return dice.mean()


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets, dims)
    cardinality = torch.sum(probs + targets, dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()
