import torch
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_mel_spectrogram(
    x: torch.Tensor,
    max_plots: int = 4,
    x_pred: torch.Tensor = None,
    sample_rate: int = 16000,
    hop_length: int = 160,
    frequency_scale: str = "mel",
):
    """Plot mel spectrogram with correct time axis and normalized dB scale.

    Arguments:
        x -- mel spectrogram tensor of shape [B, mel_bins, T]
        max_plots -- maximum number of plots to display (default: {4})
        x_pred -- predicted mel spectrogram tensor of shape [B, mel_bins, T] (default: {None})
        sample_rate -- sample rate of the original audio (default: {16000})
        hop_length -- hop length used in creating the mel spectrogram (default: {512})
        frequency_scale -- y-axis scale ('mel' or 'hz') (default: {'mel'})
    """
    # Convert tensor to numpy array
    if torch.is_tensor(x):
        specs = x.detach().cpu().numpy()
    if x_pred is not None and torch.is_tensor(x_pred):
        specs_pred = x_pred.detach().cpu().numpy()

    else:
        specs, specs_pred = x, x_pred

    # Find global min and max values across both spectrograms
    vmin = specs.min()
    vmax = specs.max()

    if x_pred is not None:
        vmin = min(vmin, specs_pred.min())
        vmax = max(vmax, specs_pred.max())

    num_specs = min(max_plots, specs.shape[0])
    cols = 2 if x_pred is not None else 1
    fig, axes = plt.subplots(num_specs, cols, figsize=(7 * cols, 2 * num_specs))

    for i in range(num_specs):
        if num_specs == 1:
            ax = axes if cols == 1 else axes[0]
            ax_pred = None if cols == 1 else axes[1]
        else:
            ax = axes[i, 0] if cols == 2 else axes[i]
            ax_pred = None if cols == 1 else axes[i, 1]

        # Calculate time axis values
        time_steps = specs[i].shape[1]
        times = np.arange(time_steps) * hop_length / sample_rate

        # Plot original spectrogram with normalized scale
        img = ax.imshow(
            specs[i],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )

        # Set correct time axis
        ax.set_xticks(np.linspace(0, time_steps - 1, 5))
        ax.set_xticklabels([f"{t:.2f}" for t in np.linspace(0, times[-1], 5)])

        plt.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_xlabel("Time (s)")
        y_label = "Frequency (Hz)" if frequency_scale == "hz" else "Mel Frequency Bins"
        ax.set_ylabel(y_label)
        ax.set_title(f"Original Mel Spectrogram {i+1}")

        # Plot predicted spectrogram if available, using same scale
        if x_pred is not None:
            img_pred = ax_pred.imshow(
                specs_pred[i],
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )

            # Set correct time axis
            ax_pred.set_xticks(np.linspace(0, time_steps - 1, 5))
            ax_pred.set_xticklabels([f"{t:.2f}" for t in np.linspace(0, times[-1], 5)])

            plt.colorbar(img_pred, ax=ax_pred, format="%+2.0f dB")
            ax_pred.set_xlabel("Time (s)")
            ax_pred.set_ylabel(y_label)
            ax_pred.set_title(f"Predicted Mel Spectrogram {i+1}")

    plt.tight_layout()
    plt.show()
