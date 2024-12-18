import torch
import torch.nn.functional as F


def mse_loss_per_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean squared error loss per batch to avoid averaging causing
    homogeneous predictions.
    """
    weights = {
        "mse": 1.0,
        # "wasserstein": 0.5,
    }

    losses = {}

    # Basic MSE loss
    losses["mse"] = F.mse_loss(target, pred)

    # Wasserstein distance using sorted values
    # This helps match the overall distribution shape
    def wasserstein_approx(p, t):
        # Reshape to [batch, -1] to handle each batch independently
        p_flat = p.reshape(p.shape[0], -1)
        t_flat = t.reshape(t.shape[0], -1)

        # Sort values for each batch
        p_sort = torch.sort(p_flat, dim=-1)[0]
        t_sort = torch.sort(t_flat, dim=-1)[0]

        # Compute distance between sorted distributions
        return F.l1_loss(p_sort, t_sort)  # Using L1 for better gradient properties

    # losses["wasserstein"] = wasserstein_approx(pred, target)

    # Compute weighted sum
    total_loss = sum(weights[k] * v for k, v in losses.items())

    return total_loss
