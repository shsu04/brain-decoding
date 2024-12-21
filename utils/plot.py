import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import typing as tp


def moving_average(values: tp.List[float], window_size: int) -> tp.List[float]:
    """Compute a simple moving average of a list of values."""
    if window_size <= 1:
        return values  # No smoothing
    cumsum = np.cumsum(np.insert(values, 0, 0))
    smoothed = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
    return smoothed.tolist()


def display_metrics(
    studies: tp.Dict[str, str],
    train_metrics: tp.Optional[tp.List[str]] = None,
    test_metrics: tp.Optional[tp.List[str]] = None,
    test_subset: str = None,
    smooth_window: int = 1,  # Increase to >1 to apply smoothing
):
    """
    Displays specified training and test metrics for multiple experiments side by side.
    Each metric is shown in its own subplot. If multiple metrics are provided, multiple
    subplots are created. Smoothing can be applied to reduce noise.

    Only one legend is displayed for all graphs, on the right side.

    Arguments:
        studies: A dictionary where keys are titles (labels) for the experiments and
                 values are paths to the corresponding saved training sessions.
        train_metrics: A list of training metrics to plot. If None or empty, no training plots.
        test_metrics: A list of test metrics to plot. If None or empty, no test plots.
        test_subset: If provided, select a particular test subset from metrics["test"].
                     If None, the first available test subset is chosen.
        smooth_window: Window size for moving average smoothing. 1 means no smoothing.
    """
    if train_metrics is None:
        train_metrics = []
    if test_metrics is None:
        test_metrics = []

    total_metrics = len(train_metrics) + len(test_metrics)
    if total_metrics == 0:
        print("No metrics provided for either training or testing.")
        return

    # Configure fonts and style for a compact, paper-friendly figure
    plt.rc("font", size=8)
    plt.rc("axes", titlesize=8)
    plt.rc("axes", labelsize=8)
    plt.rc("xtick", labelsize=6)
    plt.rc("ytick", labelsize=6)
    plt.rc("legend", fontsize=6)

    # Try to use seaborn-whitegrid style; if not available, revert to default
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        print(
            "Warning: 'seaborn-whitegrid' style is not available. Using default style."
        )
        plt.style.use("default")

    fig, axes = plt.subplots(1, total_metrics, figsize=(4 * total_metrics, 4))
    # If there's only one metric total, axes will not be a list, so make it a list.
    if total_metrics == 1:
        axes = [axes]

    # Assign axes to training and testing metrics
    train_axes = axes[: len(train_metrics)]
    test_axes = axes[len(train_metrics) :]

    # Track which studies got plotted so we can build a single combined legend
    study_lines = {}  # {study_title: line_handle}

    # We'll store the chosen test_subset after we pick it once, so we can use it in titles
    chosen_test_subset = None

    for title, save_path in studies.items():
        metrics_path = os.path.join(save_path, "metrics.pt")
        if not os.path.exists(metrics_path):
            print(
                f"Warning: No metrics found at {metrics_path} for '{title}'. Skipping."
            )
            continue

        data = torch.load(metrics_path)
        metrics = data.get("metrics", {})

        # Load training data
        train_data = metrics.get("train", [])

        # Load testing data
        test_dict = metrics.get("test", {})
        # Determine which test subset to use
        if len(test_dict) > 0:
            available_test_subsets = list(test_dict.keys())
            if test_subset is None:
                selected_test = available_test_subsets[0]
            else:
                if test_subset not in test_dict:
                    print(
                        f"Test subset '{test_subset}' not found for '{title}'. "
                        f"Available test subsets: {available_test_subsets}"
                    )
                    selected_test = None
                else:
                    selected_test = test_subset
        else:
            selected_test = None

        if selected_test is not None and len(test_dict[selected_test]) == 0:
            print(
                f"No test data for subset '{selected_test}' in '{title}'. "
                f"Available subsets: {list(test_dict.keys())}"
            )
            selected_test = None

        # Remember the chosen subset for titles (if we have test metrics)
        if (
            chosen_test_subset is None
            and selected_test is not None
            and len(test_metrics) > 0
        ):
            chosen_test_subset = selected_test

        # Plot training metrics
        if len(train_data) > 0:
            available_train_metrics = (
                train_data[0].keys() if len(train_data) > 0 else []
            )
            for i, tm in enumerate(train_metrics):
                if tm not in available_train_metrics:
                    print(
                        f"Training metric '{tm}' not found for '{title}'. "
                        f"Available training metrics: {list(available_train_metrics)}"
                    )
                    continue

                train_values = [m[tm] for m in train_data if tm in m]
                if smooth_window > 1:
                    train_values = moving_average(train_values, smooth_window)
                train_x = range(1, len(train_values) + 1)

                line = train_axes[i].plot(
                    train_x, train_values, label=title, linewidth=1
                )
                study_lines[title] = line[0]

        else:
            if len(train_metrics) > 0:
                print(
                    f"No training data found for '{title}'. Available sets: {list(metrics.keys())}. "
                    "Skipping training metrics for this study."
                )

        # Plot testing metrics
        if selected_test is not None:
            test_data_list = test_dict[selected_test]
            if len(test_data_list) > 0:
                available_test_metrics = (
                    test_data_list[0].keys() if len(test_data_list) > 0 else []
                )
                for j, tm in enumerate(test_metrics):
                    if tm not in available_test_metrics:
                        print(
                            f"Test metric '{tm}' not found for '{title}' in subset '{selected_test}'. "
                            f"Available test metrics: {list(available_test_metrics)}"
                        )
                        continue
                    test_values = [m[tm] for m in test_data_list if tm in m]
                    if smooth_window > 1:
                        test_values = moving_average(test_values, smooth_window)
                    test_x = range(1, len(test_values) + 1)

                    line = test_axes[j].plot(
                        test_x, test_values, label=title, linewidth=1
                    )
                    study_lines[title] = line[0]
            else:
                if len(test_metrics) > 0:
                    print(
                        f"No test data found for '{title}' in subset '{selected_test}'."
                    )
        else:
            if len(test_metrics) > 0 and len(test_dict) == 0:
                print(f"No test metrics available for '{title}'.")

    # Set labels and titles for training metrics
    for i, tm in enumerate(train_metrics):
        ax = train_axes[i]
        ax.set_ylabel(tm.replace("_", " ").title())
        ax.set_xlabel("Step (Train Recordings)")
        ax.set_title("Training " + tm.replace("_", " ").title())

    # Set labels and titles for testing metrics
    for j, tm in enumerate(test_metrics):
        ax = test_axes[j]
        ax.set_ylabel(tm.replace("_", " ").title())
        ax.set_xlabel("Step (Test Recordings)")
        # Include the chosen subset in the title if available
        title_str = "Testing " + tm.replace("_", " ").title()
        if chosen_test_subset is not None:
            title_str += f" - {chosen_test_subset}"
        ax.set_title(title_str)

    # Create a single combined legend to the right of all plots
    handles = list(study_lines.values())
    labels = list(study_lines.keys())

    # Adjust layout to make space on the right for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    # Place the legend outside the plotting area
    plt.tight_layout(pad=2.0)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.0, 0.5))

    plt.show()
