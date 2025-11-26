# Plot file - Federated Convergence Comparison

import matplotlib.pyplot as plt
import json
import os
import argparse
import numpy as np
from typing import Dict, List, Tuple

# Dictionary mapping run directories to labels
# Update these paths to point to your run directories.
# The code will attempt to read either <run_dir>/baseline/final_info.json
# or <run_dir>/final_info.json if the baseline subdir is not used.
labels = {
    "run_non-iid_label_skew_(heterogeneity)": "non-iid_label_skew_(heterogeneity)",
    "run_robustness_to_noise_and_outliers": "robustness_to_noise_and_outliers",
    "run_scalability:_high-dimensional,_large-scale": "scalability:_high-dimensional,_large-scale",
    "run_hyperparameter_sensitivity:_momentum_and_subspace": "hyperparameter_sensitivity:_momentum_and_subspace",
    "run_edge_case:_few_clients_selected,_many_local_steps": "edge_case:_few_clients_selected,_many_local_steps"
}


def load_results(run_dir: str) -> Dict[str, Dict[str, List[float]]]:
    baseline_path = os.path.join(run_dir, "baseline", "final_info.json")
    direct_path = os.path.join(run_dir, "final_info.json")
    path = direct_path if os.path.exists(direct_path) else baseline_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find results JSON at {direct_path} or {baseline_path}")
    with open(path, "r") as f:
        data = json.load(f)
    return data


def collect_metrics(runs: Dict[str, str]) -> Tuple[List[str], Dict[str, Dict[str, List[float]]]]:
    # Get union of metric names across runs
    all_metrics = set()
    run_data = {}
    for run_dir, _ in runs.items():
        try:
            data = load_results(run_dir)
            run_data[run_dir] = data
            for k in data.keys():
                all_metrics.add(k)
        except FileNotFoundError:
            # Skip missing runs
            continue
    metric_names = sorted(list(all_metrics))
    return metric_names, run_data


def main():
    parser = argparse.ArgumentParser(description="Plot convergence curves for federated experiments.")
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save plots')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Initialize matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')

    metric_names, run_data = collect_metrics(labels)
    if not run_data:
        # Nothing to plot
        empty_plot_path = os.path.join(args.out_dir, "comparison.png")
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.text(0.5, 0.5, "No runs found.\nUpdate the `labels` dictionary in plot.py.", ha='center', va='center')
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(empty_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved placeholder plot to: {empty_plot_path}")
        return

    # Determine subplot layout
    n_metrics = len(metric_names)
    ncols = 2 if n_metrics > 1 else 1
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols + 2, 4.5 * nrows))
    if n_metrics == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    fig.suptitle("Federated Convergence Comparison Across Scenarios", fontsize=14, fontweight='bold')

    # Colors per run
    color_map = plt.cm.tab20(np.linspace(0, 1, max(1, len(labels))))

    # Plot each metric on its own axis
    for mi, metric in enumerate(metric_names):
        r = mi // ncols
        c = mi % ncols
        ax = axes[r, c]
        for ri, (run_dir, label) in enumerate(labels.items()):
            if run_dir not in run_data:
                continue
            data = run_data[run_dir]
            if metric not in data:
                continue
            means = np.array(data[metric].get("means", []), dtype=float)
            stds = np.array(data[metric].get("stds", []), dtype=float)
            if means.size == 0:
                continue
            x = np.arange(means.size)
            color = color_map[ri % len(color_map)]
            ax.plot(x, means, label=label, color=color, linewidth=2)
            if stds.size == means.size:
                ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.2)
        ax.set_xlabel('Rounds', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3)
        # Legend is handled at the figure level for clarity across many scenarios.

    # Hide any unused subplots
    total_axes = nrows * ncols
    for extra in range(n_metrics, total_axes):
        r = extra // ncols
        c = extra % ncols
        axes[r, c].set_visible(False)

    # Save plot
    plot_path = os.path.join(args.out_dir, "comparison.png")

    # Build a combined legend across scenarios for clearer comparison
    try:
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color=color_map[ri % len(color_map)], lw=2, label=label)
            for ri, (_, label) in enumerate(labels.items())
        ]
        fig.legend(
            handles=legend_handles,
            loc='lower center',
            ncol=min(3, len(legend_handles)),
            fontsize=10,
            frameon=True,
            bbox_to_anchor=(0.5, -0.02)
        )
    except Exception:
        # If legend construction fails, skip gracefully
        pass

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
