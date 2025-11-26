# Plotting utilities for comparing Federated Averaging (FedAvg) and Subspace Federated Averaging (SFedAvg)

import matplotlib.pyplot as plt
import json
import os
import argparse
import numpy as np
from typing import Dict, List, Optional

# Dictionary mapping run directories to labels.
# Update this with your run directories, e.g.:
# labels = {
#     "runs/test_run": "Baseline",
#     "runs/variant_run": "SFedAvg Variant",
# }
labels: Dict[str, str] = {}


def find_result_path(run_dir: str) -> Optional[str]:
    """
    Prefer {run_dir}/baseline/final_info.json, fallback to {run_dir}/final_info.json.
    """
    baseline_path = os.path.join(run_dir, "baseline", "final_info.json")
    root_path = os.path.join(run_dir, "final_info.json")
    if os.path.exists(baseline_path):
        return baseline_path
    if os.path.exists(root_path):
        return root_path
    return None


def load_results(run_dir: str) -> Optional[Dict]:
    path = find_result_path(run_dir)
    if path is None:
        return None
    with open(path, "r") as f:
        return json.load(f)


def gather_metric_keys(run_results: Dict[str, Dict]) -> List[str]:
    keys = set()
    for data in run_results.values():
        if data is None:
            continue
        keys.update(data.keys())
    # Our JSON structure: {metric_name: {means: [...], stds: [...]}}
    # So keys are metric names themselves
    return list(run_results.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save plots')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load all available run results
    run_data: Dict[str, Dict] = {}
    for run_dir, label in labels.items():
        data = load_results(run_dir)
        if data is not None:
            run_data[run_dir] = data

    # If no runs specified or found, still create an informative empty plot
    plt.style.use('seaborn-v0_8-whitegrid')

    # Determine union of metric names across runs
    metric_names = set()
    for data in run_data.values():
        for metric_name in data.keys():
            metric_names.add(metric_name)

    if not metric_names:
        # Create an empty figure with instructions
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.set_title("No runs found. Update 'labels' in plot.py to point to run directories.")
        ax.set_xlabel("Iterations/Rounds")
        ax.set_ylabel("Metric Value")
        ax.grid(True, alpha=0.3)
        out_path = os.path.join(args.out_dir, "empty.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved empty plot to: {out_path}")
        return

    # For each metric, produce a standalone high-quality plot
    for metric_name in sorted(metric_names):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(run_data))))
        color_idx = 0

        for run_dir, label in labels.items():
            data = run_data.get(run_dir, None)
            if data is None or metric_name not in data:
                continue

            means = np.array(data[metric_name].get("means", []), dtype=np.float64)
            stds = np.array(data[metric_name].get("stds", []), dtype=np.float64)
            iterations = np.arange(len(means))

            ax.plot(
                iterations,
                means,
                label=f"{label}",
                color=colors[color_idx % len(colors)],
                linewidth=2,
            )

            if stds.size == means.size and np.any(stds > 0):
                ax.fill_between(
                    iterations,
                    means - stds,
                    means + stds,
                    alpha=0.2,
                    color=colors[color_idx % len(colors)],
                )

            color_idx += 1

        ax.set_xlabel('Rounds', fontsize=12)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Comparison: {metric_name.replace("_", " ").title()}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = os.path.join(args.out_dir, f"{metric_name}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
