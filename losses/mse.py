import torch
import torch.nn.functional as F


def mse_loss_per_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean squared error loss per batch.
    """
    loss = F.mse_loss(target, pred)

    return loss
