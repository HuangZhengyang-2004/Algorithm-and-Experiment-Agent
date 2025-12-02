# Plotting script for Federated Learning experiments
# Reads final_info.json from one or more run directories and produces comparison plots.

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Dictionary mapping run directories to labels. Keys can be either:
# - the directory that directly contains final_info.json, or
# - the parent run directory that contains a 'baseline/final_info.json'.
# If left empty, the script will try to auto-discover immediate subdirectories
# in the current working directory that contain 'baseline/final_info.json'.
labels: Dict[str, str] = {}


def load_results(run_dir: str) -> Dict[str, Dict[str, List[float]]]:
    # Accept run_dir as either path to final_info.json directory or parent run dir
    json_path = os.path.join(run_dir, "final_info.json")
    if not os.path.exists(json_path):
        # try baseline
        json_path = os.path.join(run_dir, "baseline", "final_info.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"final_info.json not found in {run_dir} or {os.path.join(run_dir, 'baseline')}")
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def discover_runs() -> Dict[str, str]:
    discovered = {}
    for name in os.listdir("."):
        if not os.path.isdir(name):
            continue
        candidate = os.path.join(name, "baseline", "final_info.json")
        if os.path.exists(candidate):
            discovered[os.path.join(name, "baseline")] = name
    return discovered


def plot_metric(metric_name: str, runs_data: Dict[str, Dict[str, Dict[str, List[float]]]], out_path: str):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    if not runs_data:
        ax.set_title(f"No runs available - {metric_name}")
        ax.set_xlabel("Iterations/Rounds")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(runs_data))))

    for i, (run_dir, data) in enumerate(runs_data.items()):
        if metric_name not in data:
            continue
        means = np.array(data[metric_name].get("means", []), dtype=float)
        stds = np.array(data[metric_name].get("stds", np.zeros_like(means)), dtype=float)
        if means.size == 0:
            continue
        x = np.arange(len(means))
        label = runs_data[run_dir].get("_label", os.path.basename(run_dir))
        # Plot mean with std shading
        ax.plot(x, means, label=label, color=colors[i], linewidth=2)
        low = means - stds
        high = means + stds
        ax.fill_between(x, low, high, color=colors[i], alpha=0.2)

    ax.set_xlabel('Iterations/Rounds', fontsize=12)
    ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=12)
    ax.set_title(f'Comparison: {metric_name.replace("_", " ").title()}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save plots')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    run_labels = dict(labels)  # copy

    if not run_labels:
        # Try to discover runs automatically in current directory
        run_labels = discover_runs()

    # Load data
    runs_data: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for run_dir, label in run_labels.items():
        try:
            data = load_results(run_dir)
        except FileNotFoundError:
            # Skip missing runs
            continue
        data["_label"] = label  # attach label for plotting
        runs_data[run_dir] = data

    # Determine all metric names across runs
    metric_names = set()
    for rd, data in runs_data.items():
        metric_names.update([k for k in data.keys() if not k.startswith("_")])

    if not metric_names:
        # Create a placeholder plot so the script still outputs a PNG
        out_path = os.path.join(args.out_dir, "comparison.png")
        plot_metric("no_metrics_found", {}, out_path)
        print(f"Saved placeholder plot to: {out_path}")
        return

    # Plot each metric into its own PNG
    for metric in sorted(metric_names):
        out_path = os.path.join(args.out_dir, f"{metric}.png")
        plot_metric(metric, runs_data, out_path)
        print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
