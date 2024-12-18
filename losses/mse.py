import torch
import torch.nn.functional as F


def mse_loss_per_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean squared error loss per batch to avoid averaging causing
    homogeneous predictions.
    """
    # loss = ((pred - target) ** 2).mean(dim=[1, 2]).mean()

    weights = {
        "mse": 1.0,
        "temporal_grad": 0.3,
        # "freq_grad": 0.1,
        # "energy_bands": 0.1,
        # "local_patterns": 0.1,
        # "correlation": 0.1,
        # "peaks": 0.1,
    }

    losses = {}

    # Basic MSE loss
    losses["mse"] = F.mse_loss(target, pred)

    # Temporal gradient loss, preserves vertical stripes in the spectrograms
    t_grad_orig = target[:, :, 1:] - target[:, :, :-1]
    t_grad_pred = pred[:, :, 1:] - pred[:, :, :-1]
    losses["temporal_grad"] = F.mse_loss(t_grad_orig, t_grad_pred)

    # # Frequency gradient loss, preserves horizontal stripes in the spectrograms
    # f_grad_orig = target[:, 1:, :] - target[:, :-1, :]
    # f_grad_pred = pred[:, 1:, :] - pred[:, :-1, :]
    # losses["freq_grad"] = F.mse_loss(f_grad_orig, f_grad_pred)

    # # Energy bands loss, preserves the overall energy distribution across frequencies
    # # Maintains the overall energy distribution across frequencies
    # losses["energy_bands"] = (
    #     F.mse_loss(target[:, :32, :].mean(1), pred[:, :32, :].mean(1))  # Low freqs
    #     + F.mse_loss(
    #         target[:, 32:80, :].mean(1), pred[:, 32:80, :].mean(1)
    #     )  # Mid freqs
    #     + F.mse_loss(target[:, 80:, :].mean(1), pred[:, 80:, :].mean(1))  # High freqs
    # )

    # # Local pattern structure loss (3x3 patches)
    # patches_orig = F.unfold(target.unsqueeze(1), kernel_size=3)
    # patches_pred = F.unfold(pred.unsqueeze(1), kernel_size=3)
    # losses["local_patterns"] = F.mse_loss(patches_orig, patches_pred)

    # # Frame-wise correlation loss, preserves the overall shape of the spectrogram
    # # across time, emphasizing less on magnitude
    # orig_norm = F.normalize(target, dim=1)
    # pred_norm = F.normalize(pred, dim=1)
    # correlation = torch.sum(orig_norm * pred_norm, dim=1)
    # losses["correlation"] = 1.0 - correlation.mean()

    # # Peak preservation loss
    # threshold = 0.8 * target.max()
    # peaks_mask = target > threshold
    # losses["peaks"] = F.mse_loss(target[peaks_mask], pred[peaks_mask])

    # Compute weighted sum
    total_loss = sum(weights[k] * v for k, v in losses.items())

    # Take the mean across the batch: scalar
    return total_loss.mean()
