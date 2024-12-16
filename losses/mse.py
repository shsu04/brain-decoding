import torch

def mse_loss_per_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean squared error loss per batch to avoid averaging causing
    homogeneous predictions.
    """
    
    #[B, C, T] -> [B, C * T]
    pred_flat = pred.view(pred.size(0), -1)       # Shape: [B, C * T]
    target_flat = target.view(target.size(0), -1) # Shape: [B, C * T]

    # Compute the sum of squared errors across C * T for each batch: [B]
    mse_per_batch = torch.sum((pred_flat - target_flat) ** 2, dim=1)

    # Take the mean across the batch: scalar
    return torch.mean(mse_per_batch)