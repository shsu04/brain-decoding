import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import typing as tp
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns


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

    # Try to use seaborn-whitegrid style; if not available, revert to default
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        print(
            "Warning: 'seaborn-whitegrid' style is not available. Using default style."
        )
        plt.style.use("default")

    sns.set_palette("tab20", len(studies))

    # Configure fonts and style for a compact, paper-friendly figure
    mpl.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )

    mpl.rcParams.update()

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

        enable_fine_grid(ax)
        ax.set_axisbelow(True)

        ax.set_ylabel(tm.replace("_", " ").title())
        ax.set_xlabel("Epoch (Train Recordings)")
        ax.set_title("Training " + tm.replace("_", " ").title())

    # Set labels and titles for testing metrics
    for j, tm in enumerate(test_metrics):
        ax = test_axes[j]

        enable_fine_grid(ax)
        ax.set_axisbelow(True)

        ax.set_ylabel(tm.replace("_", " ").title())
        ax.set_xlabel(f"Epoch {chosen_test_subset}")
        # Include the chosen subset in the title if available
        title_str = "Testing " + tm.replace("_", " ").title()
        ax.set_title(title_str)

    # Create a single combined legend to the right of all plots
    handles = list(study_lines.values())
    labels = list(study_lines.keys())

    # Adjust layout to make space on the right for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    # Place the legend outside the plotting area
    plt.tight_layout(pad=2.0)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))

    plt.show()


def enable_fine_grid(ax):
    """Enable major and minor axis grid"""
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.75)
    ax.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.5)

    # Turn on minor ticks for both axes
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def display_best_performance_barchart(
    studies: tp.Dict[str, str],
    test_metrics: tp.List[str],
    test_subsets: tp.List[str],
    top_percent: bool = False,
):
    """
    Displays a bar chart of the best (maximum) test performance across different test subsets
    for multiple experiments (studies). Each test metric gets its own subplot. The x-axis
    represents the given test subsets, and each study's best score is shown side-by-side in a grouped bar.

    Now updated to replicate the training logic:
    We first determine the "best epoch" by summing a chosen metric (e.g., top_10_accuracy)
    across all test subsets and identifying the epoch with the highest sum. We then take that epoch's values
    for all requested test metrics.

    Arguments:
        studies: A dictionary where keys are titles (labels) for the experiments and
                 values are paths to the corresponding saved training sessions (containing metrics.pt).
        test_metrics: A list of test metrics to consider for plotting.
        test_subsets: A list of test subsets (categories) to plot on the x-axis.
    """

    if len(test_metrics) == 0 or len(test_subsets) == 0:
        print("No test metrics or test subsets provided.")
        return

    # This is the metric we use to determine the best epoch, similar to the training loop.
    # You can change it if you use a different metric to select the best epoch.
    summation_metric = "top_10_accuracy"

    # Try to use seaborn-whitegrid style; if not available, revert to default
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        print(
            "Warning: 'seaborn-whitegrid' style is not available. Using default style."
        )
        plt.style.use("default")

    sns.set_palette("tab20", len(studies))

    # Configure fonts and style
    mpl.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 12,
            "ytick.labelsize": 16,
            "legend.fontsize": 18,
        }
    )
    mpl.rcParams.update()

    # Prepare figure with one subplot per test metric
    n_metrics = len(test_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), squeeze=False)
    axes = axes[0]  # (1, n) shape, select the row

    n_categories = len(test_subsets)
    study_titles = list(studies.keys())
    n_studies = len(study_titles)

    # We'll store the best epoch performance for each study here:
    # best_perf[metric][subset][study_title] = value at best epoch
    best_perf = {m: {s: {} for s in test_subsets} for m in test_metrics}

    for title, save_path in studies.items():
        metrics_path = os.path.join(save_path, "metrics.pt")
        if not os.path.exists(metrics_path):
            print(
                f"Warning: No metrics found at {metrics_path} for '{title}'. Skipping."
            )
            for tm in test_metrics:
                for subset in test_subsets:
                    best_perf[tm][subset][title] = np.nan
            continue

        data = torch.load(metrics_path)
        metrics = data.get("metrics", {})
        test_dict = metrics.get("test", {})

        # Check if we have data for all subsets
        # We'll only consider epochs that appear consistently across subsets
        # If a subset doesn't exist, mark as NaN
        subset_data_lists = []
        for subset in test_subsets:
            if subset in test_dict and len(test_dict[subset]) > 0:
                subset_data_lists.append(test_dict[subset])
            else:
                # Missing or empty subset data
                subset_data_lists.append([])

        # If any subset is empty, we can't find a proper best epoch; just NaN them out
        if any(len(lst) == 0 for lst in subset_data_lists):
            for tm in test_metrics:
                for subset in test_subsets:
                    best_perf[tm][subset][title] = np.nan
            continue

        # All subsets have data; assume they're aligned by epoch index
        # Find the best epoch by summing summation_metric across subsets
        # Number of epochs:
        n_epochs = len(subset_data_lists[0])

        # Verify all subsets have the same number of epochs
        if not all(len(lst) == n_epochs for lst in subset_data_lists):
            print(
                f"Warning: Inconsistent number of epochs across subsets for '{title}'. "
                "This might indicate mismatched runs. Setting values to NaN."
            )
            for tm in test_metrics:
                for subset in test_subsets:
                    best_perf[tm][subset][title] = np.nan
            continue

        # Compute sum over subsets for each epoch
        best_sum = -np.inf
        best_epoch = 0
        for e in range(n_epochs):
            # Sum the summation_metric across all subsets at epoch e
            current_sum = 0
            valid = True
            for lst in subset_data_lists:
                val = lst[e].get(summation_metric, None)
                if val is None:
                    valid = False
                    break
                current_sum += val
            if valid and current_sum > best_sum:
                best_sum = current_sum
                best_epoch = e

        # Now best_epoch is the epoch with the highest sum of summation_metric
        # Extract values for requested test_metrics from that epoch
        for subset_i, subset in enumerate(test_subsets):
            epoch_data = test_dict[subset][best_epoch]
            for tm in test_metrics:
                val = epoch_data.get(tm, np.nan)
                best_perf[tm][subset][title] = val

    # Now plot the bar charts using best_perf
    bar_width = 0.8 / n_studies
    x_positions = np.arange(n_categories)

    for i, tm in enumerate(test_metrics):
        ax = axes[i]

        enable_fine_grid(ax)
        ax.set_axisbelow(True)

        for j, study_title in enumerate(study_titles):
            vals = [
                best_perf[tm][subset].get(study_title, np.nan)
                for subset in test_subsets
            ]
            offset = (j - (n_studies - 1) / 2) * bar_width
            bar_positions = x_positions + offset
            ax.bar(bar_positions, vals, width=bar_width, label=study_title)

        ax.set_xticks(x_positions)
        formatted_subsets = [s.replace("_", " ").title() for s in test_subsets]
        ax.set_xticklabels(formatted_subsets, rotation=0)

        title = tm.replace("_", " ").title()

        # Fix the top % vs absolute values
        if top_percent and title != "Accuracy":
            title += " %"

        ax.set_title(title)
        ax.set_ylabel(title)

        if i == n_metrics - 1:
            ax.legend(study_titles, loc="best", bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()
    plt.show()
