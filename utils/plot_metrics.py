import os
import typing as tp
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator


def moving_average(values: tp.List[float], window_size: int) -> tp.List[float]:
    """
    Compute a simple moving average of a list of values.
    If window_size <= 1, returns the original values (no smoothing).
    """
    if window_size <= 1:
        return values
    cumsum = np.cumsum(np.insert(values, 0, 0))
    smoothed = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
    return smoothed.tolist()


def expand_available_metrics(metric_dict: tp.Dict) -> tp.Set[str]:
    """
    Return all top-level keys plus any sub-keys in "final_layer_losses"
    (with "final_layer_losses_" prefixed).
    """
    keys = set(metric_dict.keys())
    if "final_layer_losses" in metric_dict and isinstance(
        metric_dict["final_layer_losses"], dict
    ):
        for subkey in metric_dict["final_layer_losses"].keys():
            keys.add("final_layer_losses_" + subkey)
    return keys


def get_metric_value(metric_dict: tp.Dict, key: str):
    """
    Retrieve the value for `key` from `metric_dict`. If `key` starts with
    "final_layer_losses_", look inside the nested "final_layer_losses" dictionary.
    Otherwise, return metric_dict[key] if it exists, else np.nan.
    """
    if key in metric_dict:
        return metric_dict[key]
    elif key.startswith("final_layer_losses_"):
        subkey = key[len("final_layer_losses_") :]
        return metric_dict.get("final_layer_losses", {}).get(subkey, np.nan)
    else:
        return np.nan


def beautify_metric_name(metric_name: str, top_percent: bool = False) -> str:
    """
    Convert a metric key into a more readable form for plotting:
      - If metric_name starts with "final_layer_losses_", it becomes something
        like "Final Layer MSE Loss" or "Final Layer Cosine Similarity Loss".
      - Otherwise, underscores are replaced with spaces, and it is title-cased.
      - If top_percent=True and it’s not "Accuracy", optionally append '%'.
    """
    if metric_name.startswith("final_layer_losses_"):
        # Remove the prefix
        subkey = metric_name[len("final_layer_losses_") :]
        if subkey.endswith("_loss"):
            subkey = subkey[: -len("_loss")]  # remove trailing "_loss"
        subkey_title = subkey.replace("_", " ").title()

        # Special replacements
        subkey_title = subkey_title.replace("Mse", "MSE")
        if "Cosine Similarity" in subkey_title:
            subkey_title = subkey_title.replace(
                "Cosine Similarity", "Cosine Similarity Loss"
            )
        elif not subkey_title.endswith("Loss"):
            subkey_title += " Loss"

        title_str = "Final Layer " + subkey_title
        if top_percent and "Accuracy" not in title_str and not title_str.endswith("%"):
            title_str += " %"
        return title_str
    elif metric_name.startswith("mse"):
        title_str = "MSE"
        return title_str
    else:
        title_str = metric_name.replace("_", " ").title()
        if top_percent and title_str.lower() != "accuracy":
            title_str += " %"
        return title_str


def enable_fine_grid(ax):
    """Enable major and minor axis grid lines on the given Axes object."""
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.75)
    ax.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def display_metrics(
    studies: tp.Dict[str, str],
    train_metrics: tp.Optional[tp.List[str]] = None,
    test_metrics: tp.Optional[tp.List[str]] = None,
    test_subset: str = None,
    smooth_window: int = 1,
):
    """
    Displays specified training and test metrics for multiple experiments side by side.
    Each metric is shown in its own subplot. If multiple metrics are provided, multiple
    subplots are created. Smoothing can be applied to reduce noise.
    Only one legend is displayed for all graphs, on the right side.
    """
    if train_metrics is None:
        train_metrics = []
    if test_metrics is None:
        test_metrics = []

    total_metrics = len(train_metrics) + len(test_metrics)
    if total_metrics == 0:
        print("No metrics provided for either training or testing.")
        return

    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        print("Warning: 'seaborn-whitegrid' style not available. Using default style.")
        plt.style.use("default")

    sns.set_palette("tab20", len(studies))
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
    if total_metrics == 1:
        axes = [axes]

    train_axes = axes[: len(train_metrics)]
    test_axes = axes[len(train_metrics) :]
    study_lines = {}
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
        train_data = metrics.get("train", [])
        test_dict = metrics.get("test", {})

        # Decide which test subset to use
        if len(test_dict) > 0:
            available_test_subsets = list(test_dict.keys())
            if test_subset is None:
                selected_test = available_test_subsets[0]
            else:
                if test_subset not in test_dict:
                    print(
                        f"Test subset '{test_subset}' not found for '{title}'. "
                        f"Available: {available_test_subsets}"
                    )
                    selected_test = None
                else:
                    selected_test = test_subset
        else:
            selected_test = None

        if selected_test is not None and len(test_dict[selected_test]) == 0:
            print(
                f"No test data for subset '{selected_test}' in '{title}'. "
                f"Available: {list(test_dict.keys())}"
            )
            selected_test = None

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
                        f"Available: {list(available_train_metrics)}"
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
                    f"No training data found for '{title}'. "
                    f"Skipping training metrics."
                )

        # Plot testing metrics
        if selected_test is not None:
            test_data_list = test_dict[selected_test]
            if len(test_data_list) > 0:
                available_test_metrics = test_data_list[0].keys()
                for j, tm in enumerate(test_metrics):
                    if tm not in available_test_metrics:
                        print(
                            f"Test metric '{tm}' not found for '{title}' in subset '{selected_test}'. "
                            f"Available: {list(available_test_metrics)}"
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

    # Training axes labels
    for i, tm in enumerate(train_metrics):
        ax = train_axes[i]
        enable_fine_grid(ax)
        ax.set_axisbelow(True)
        ax.set_ylabel(tm.replace("_", " ").title())
        ax.set_xlabel("Epoch (Train Recordings)")
        ax.set_title("Training " + tm.replace("_", " ").title())

    # Testing axes labels
    for j, tm in enumerate(test_metrics):
        ax = test_axes[j]
        enable_fine_grid(ax)
        ax.set_axisbelow(True)
        ax.set_ylabel(tm.replace("_", " ").title())
        if chosen_test_subset is not None:
            ax.set_xlabel(f"Epoch {chosen_test_subset}")
        else:
            ax.set_xlabel("Epoch (Test)")
        ax.set_title("Testing " + tm.replace("_", " ").title())

    handles = list(study_lines.values())
    labels = list(study_lines.keys())

    # Layout adjustments
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.tight_layout(pad=2.0)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.show()


def display_best_performance_barchart(
    studies: tp.Dict[str, str],
    test_metrics: tp.List[str],
    test_subsets: tp.List[str],
    top_percent: bool = False,
):
    """
    Displays a bar chart of the best (maximum) test performance across different test subsets
    for multiple experiments (studies). Each test metric gets its own subplot. The x-axis
    represents the given test subsets, and each study's best score is shown side-by-side
    in a grouped bar chart.
    """
    if len(test_metrics) == 0 or len(test_subsets) == 0:
        print("No test metrics or test subsets provided.")
        return

    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        print("Warning: 'seaborn-whitegrid' style not available. Using default style.")
        plt.style.use("default")

    sns.set_palette("tab20", len(studies))
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

    n_metrics = len(test_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), squeeze=False)
    axes = axes[0]  # (1, n_metrics) shape
    n_categories = len(test_subsets)
    study_titles = list(studies.keys())
    n_studies = len(study_titles)

    # best_perf[metric][subset][study_title]
    best_perf = {m: {s: {} for s in test_subsets} for m in test_metrics}

    for title, save_path in studies.items():
        metrics_path = os.path.join(save_path, "metrics.pt")
        if not os.path.exists(metrics_path):
            print(
                f"Warning: No metrics found at {metrics_path} for '{title}'. Skipping."
            )
            # Fill in NaNs to keep it consistent
            for tm in test_metrics:
                for subset in test_subsets:
                    best_perf[tm][subset][title] = np.nan
            continue

        data = torch.load(metrics_path)
        highest_metrics = data.get("highest_metrics", {})

        for subset in test_subsets:
            for tm in test_metrics:
                val = get_metric_value(highest_metrics.get(subset, {}), tm)
                best_perf[tm][subset][title] = val

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
        ax.set_xticklabels(
            [s.replace("_", " ").title() for s in test_subsets], rotation=0
        )

        pretty_name = beautify_metric_name(tm, top_percent)
        ax.set_title(pretty_name)
        ax.set_ylabel(pretty_name)

        # Legend on the last plot
        if i == n_metrics - 1:
            ax.legend(study_titles, loc="best", bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()
    plt.show()
