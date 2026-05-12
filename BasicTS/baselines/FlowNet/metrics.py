import torch
from torch.nn import functional as F


def plain_mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(prediction, target)


def plain_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target)


def plain_rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(plain_mse(prediction, target))
