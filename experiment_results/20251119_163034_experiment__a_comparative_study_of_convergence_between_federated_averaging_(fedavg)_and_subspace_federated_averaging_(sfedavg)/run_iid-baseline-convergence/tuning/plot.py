# Plot file - Comparative visualization for FedAvg vs SFedAvg

import argparse
import json
import os
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


def find_final_info_path(run_dir: str) -> str:
    """
    Return the path to final_info.json given a run directory.
    Checks:
      - run_dir if it is already a file ending with .json
      - run_dir/baseline/final_info.json
      - run_dir/final_info.json
    """
    if run_dir.endswith(".json") and os.path.isfile(run_dir):
        return run_dir
    p1 = os.path.join(run_dir, "baseline", "final_info.json")
    p2 = os.path.join(run_dir, "final_info.json")
    if os.path.isfile(p1):
        return p1
    if os.path.isfile(p2):
        return p2
    return ""


def load_results(run_dir: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Load the results dictionary from final_info.json for a single run.
    """
    path = find_final_info_path(run_dir)
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def ensure_out_dir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)


def save_placeholder_plot(out_dir: str,
                          filename: str = "comparison.png",
                          title: str = "No Results",
                          message: str = "No valid runs found to plot."):
    """
    Save a placeholder PNG so the script always generates at least one figure.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('off')
    ax.text(0.5, 0.6, title, fontsize=16, ha='center', va='center', weight='bold')
    ax.text(0.5, 0.4, message, fontsize=12, ha='center', va='center')
    path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {path}")


def plot_metric_series(all_runs: Dict[str, Dict[str, Dict[str, List[float]]]],
                       out_dir: str):
    """
    Given mapping run_label -> results (metric_name -> {means, stds}), save separate figures per base metric.
    Metric names may include a method suffix separated by '/', e.g., 'train_loss/FedAvg'.
    """
    # Gather all metric base names
    metric_groups: Dict[str, List[Tuple[str, str]]] = {}  # base_metric -> list of (run_label, metric_name)
    for run_label, results in all_runs.items():
        for metric_name in results.keys():
            base = metric_name.split('/')[0] if '/' in metric_name else metric_name
            metric_groups.setdefault(base, []).append((run_label, metric_name))

    plt.style.use('seaborn-v0_8-whitegrid')

    for base_metric, items in metric_groups.items():
        # Build a figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, len(items)))

        for i, (run_label, metric_name) in enumerate(items):
            series = all_runs[run_label][metric_name]
            means = np.array(series.get("means", []), dtype=float)
            stds = np.array(series.get("stds", []), dtype=float)
            if means.size == 0:
                continue
            iters = np.arange(len(means))
            # Label includes run label and method if present
            method = metric_name.split('/')[1] if '/' in metric_name else ""
            label = f"{run_label}" + (f" [{method}]" if method else "")
            ax.plot(iters, means, label=label, color=colors[i], linewidth=2)
            if stds.size == means.size:
                ax.fill_between(iters, means - stds, means + stds, color=colors[i], alpha=0.2)

        ax.set_xlabel('Rounds', fontsize=12)
        ax.set_ylabel(base_metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f"{base_metric.replace('_', ' ').title()} Comparison", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save figure
        fname = base_metric.lower().replace(' ', '_') + ".png"
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot results from one or more runs.")
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save plots')
    # The labels dictionary maps run directories to human-friendly labels.
    # Update this mapping to add more runs.
    parser.add_argument('--runs', type=str, nargs='*', default=None,
                        help='Optional list of run_dir=Label entries; if omitted, uses a default example.')
    args = parser.parse_args()

    ensure_out_dir(args.out_dir)

    # Build labels mapping
    labels: Dict[str, str] = {}
    if args.runs:
        for entry in args.runs:
            if '=' in entry:
                run_dir, label = entry.split('=', 1)
                labels[run_dir] = label
    else:
        # No default runs; will generate a placeholder plot if none provided.
        labels = {}

    # Load all runs
    all_runs: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for run_dir, label in labels.items():
        results = load_results(run_dir)
        if results:
            all_runs[label] = results
        else:
            print(f"Warning: No results found in '{run_dir}'. Expected final_info.json under 'baseline/' or run root.")

    if not all_runs:
        print("No valid runs to plot. Generating a placeholder figure.")
        save_placeholder_plot(args.out_dir, filename="comparison.png",
                              title="No Results", message="No valid runs found to plot.")
        return

    plot_metric_series(all_runs, args.out_dir)


if __name__ == "__main__":
    main()
