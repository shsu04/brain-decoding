import torch

def mse_loss_per_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean squared error loss per batch to avoid averaging causing
    homogeneous predictions.
    """
    loss = ((pred - target) ** 2).mean(dim=[1,2]).mean()

    # Take the mean across the batch: scalar
    return loss.mean()