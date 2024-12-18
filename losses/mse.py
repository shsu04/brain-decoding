import torch
import torch.nn.functional as F


def mse_loss_per_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean squared error loss per batch to avoid averaging causing
    homogeneous predictions.
    """
    weights = {
        "mse": 1.0,
    }

    losses = {}

    # Basic MSE loss
    losses["mse"] = F.mse_loss(target, pred)

    # Compute weighted sum
    total_loss = sum(weights[k] * v for k, v in losses.items())

    return total_loss
