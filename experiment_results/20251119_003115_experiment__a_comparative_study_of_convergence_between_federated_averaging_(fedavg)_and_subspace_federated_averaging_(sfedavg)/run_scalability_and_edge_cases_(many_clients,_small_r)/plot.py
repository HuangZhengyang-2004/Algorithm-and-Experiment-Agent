# Plot file - compare convergence curves across runs

import matplotlib.pyplot as plt
import json
import os
import argparse
import numpy as np


def load_run_metrics(run_dir: str) -> dict:
    path = os.path.join(run_dir, "final_info.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def plot_metric_across_runs(metric_name: str, runs: dict, out_dir: str):
    """
    Plot a single metric across multiple runs.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    for i, (run_dir, label) in enumerate(runs.items()):
        data = load_run_metrics(run_dir)
        if not data or metric_name not in data:
            continue
        means = np.array(data[metric_name]["means"])
        stds = np.array(data[metric_name]["stds"])
        iters = np.arange(len(means))
        ax.plot(iters, means, label=f"{label}", color=colors[i], linewidth=2)
        if stds.shape == means.shape:
            ax.fill_between(iters, means - stds, means + stds, color=colors[i], alpha=0.2)

    ax.set_xlabel("Rounds", fontsize=12)
    ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"{metric_name.replace('_', ' ').title()} Comparison", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"{metric_name}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save plots')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Auto-discover runs: include current directory if it has final_info.json,
    # and any immediate subdirectories that contain final_info.json.
    labels = {}
    if os.path.exists("final_info.json"):
        labels["."] = "Current Run"
    for entry in sorted(os.listdir(".")):
        if os.path.isdir(entry) and os.path.exists(os.path.join(entry, "final_info.json")):
            labels[entry] = entry

    # Determine all metrics present across discovered runs
    all_metrics = set()
    for run_dir in labels.keys():
        data = load_run_metrics(run_dir)
        for k in data.keys():
            all_metrics.add(k)

    if not all_metrics:
        # No runs found: generate a placeholder PNG so the script still outputs something useful.
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, "No runs with final_info.json found.\nPlace your run directories here and re-run.",
                ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        placeholder_path = os.path.join(args.out_dir, "no_runs_found.png")
        plt.tight_layout()
        plt.savefig(placeholder_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved placeholder plot to: {placeholder_path}")
        return

    # Plot each metric separately
    for metric in sorted(all_metrics):
        plot_metric_across_runs(metric, labels, args.out_dir)


if __name__ == "__main__":
    main()
