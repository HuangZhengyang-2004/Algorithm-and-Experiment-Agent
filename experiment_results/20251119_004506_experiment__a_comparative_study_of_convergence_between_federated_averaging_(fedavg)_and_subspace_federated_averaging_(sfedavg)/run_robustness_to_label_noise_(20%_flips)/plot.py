# Plotting script for comparing FedAvg vs SFedAvg results across multiple runs.
# Reads {run_dir}/final_info.json and saves professional-looking PNG plots.

import matplotlib.pyplot as plt
import json
import os
import argparse
import numpy as np
from typing import Dict, List


def load_run_metrics(run_dir: str) -> Dict[str, Dict[str, List[float]]]:
    path = os.path.join(run_dir, "final_info.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save plots')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Dictionary mapping run directories to labels
    # Edit this mapping to include your run directories; auto-detect any dirs with final_info.json.
    labels: Dict[str, str] = {}

    # Auto-detect runs in current directory
    for entry in os.listdir("."):
        run_path = os.path.join(".", entry)
        if os.path.isdir(run_path) and os.path.exists(os.path.join(run_path, "final_info.json")):
            labels[run_path] = entry

    if not labels:
        print("No runs detected with final_info.json. Plots will be empty but saved.")
        # Create a dummy mapping to proceed
        labels = {}

    # Initialize matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Prepare figures: loss, accuracy, communication
    figs = {
        "loss_comparison.png": plt.subplots(1, 1, figsize=(10, 6)),
        "accuracy_comparison.png": plt.subplots(1, 1, figsize=(10, 6)),
        "communication_comparison.png": plt.subplots(1, 1, figsize=(10, 6))
    }

    # Colors per run
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(labels))))

    # Iterate runs and plot metrics
    for i, (run_dir, label) in enumerate(labels.items()):
        data = load_run_metrics(run_dir)
        if not data:
            continue

        # Collect series
        # Metrics naming convention expected from experiment.py:
        # "FedAvg/train_loss", "FedAvg/test_accuracy", "FedAvg/communication_bytes",
        # "SFedAvg/train_loss", "SFedAvg/test_accuracy", "SFedAvg/communication_bytes"
        for metric_name, metric_data in data.items():
            means = metric_data.get("means", [])
            stds = metric_data.get("stds", [0.0] * len(means))
            x = np.arange(len(means))
            means_arr = np.array(means, dtype=np.float64)
            stds_arr = np.array(stds, dtype=np.float64)

            if "train_loss" in metric_name:
                fig, ax = figs["loss_comparison.png"]
                ax.plot(x, means_arr, label=f"{label} - {metric_name}", color=colors[i], linewidth=2)
                ax.fill_between(x, means_arr - stds_arr, means_arr + stds_arr, alpha=0.2, color=colors[i])
            elif "test_accuracy" in metric_name:
                fig, ax = figs["accuracy_comparison.png"]
                ax.plot(x, means_arr, label=f"{label} - {metric_name}", color=colors[i], linewidth=2)
                ax.fill_between(x, means_arr - stds_arr, means_arr + stds_arr, alpha=0.2, color=colors[i])
            elif "communication" in metric_name:
                fig, ax = figs["communication_comparison.png"]
                ax.plot(x, means_arr, label=f"{label} - {metric_name}", color=colors[i], linewidth=2)
                ax.fill_between(x, means_arr - stds_arr, means_arr + stds_arr, alpha=0.2, color=colors[i])

    # Customize and save each plot
    for filename, (fig, ax) in figs.items():
        if "loss" in filename:
            ax.set_xlabel('Rounds', fontsize=12)
            ax.set_ylabel('Training Loss', fontsize=12)
            ax.set_title('Training Loss Convergence', fontsize=14)
        elif "accuracy" in filename:
            ax.set_xlabel('Rounds', fontsize=12)
            ax.set_ylabel('Test Accuracy', fontsize=12)
            ax.set_title('Test Accuracy over Rounds', fontsize=14)
        elif "communication" in filename:
            ax.set_xlabel('Rounds', fontsize=12)
            ax.set_ylabel('Bytes (approx.)', fontsize=12)
            ax.set_title('Communication per Round', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = os.path.join(args.out_dir, filename)
        plt.tight_layout()
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
