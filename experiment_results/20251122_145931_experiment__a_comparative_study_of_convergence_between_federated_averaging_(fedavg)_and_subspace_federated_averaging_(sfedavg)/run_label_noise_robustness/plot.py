# Plot file - Federated Averaging vs Subspace Federated Averaging comparison

import matplotlib.pyplot as plt
import json
import os
import argparse
import numpy as np

# Dictionary mapping run directories to labels
# Users can update this to include multiple experiment outputs
labels = {
    # Example:
    # "runs/test_run_1": "Baseline",
    # "runs/test_run_2": "SFedAvg Variant",
}


def load_results(run_dir: str):
    """
    Load results from a run directory expecting baseline/final_info.json.
    Returns dict or None if not found.
    """
    path = os.path.join(run_dir, "baseline", "final_info.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save plots')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Initialize matplotlib with professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Colors for different runs
    num_runs = max(1, len(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, num_runs))

    plotted_any = False
    for i, (run_dir, label) in enumerate(labels.items()):
        data = load_results(run_dir)
        if data is None:
            continue

        # Plot each metric as a separate curve
        run_color = colors[i]
        for metric_name, metric_data in data.items():
            means = np.array(metric_data.get("means", []), dtype=np.float64)
            stds = np.array(metric_data.get("stds", []), dtype=np.float64)
            if means.size == 0:
                continue
            iterations = np.arange(means.size)

            ax.plot(iterations, means, label=f"{label} - {metric_name}", color=run_color, linewidth=2)
            if stds.size == means.size and np.any(stds > 0):
                ax.fill_between(iterations, means - stds, means + stds, alpha=0.2, color=run_color)
            plotted_any = True

    # Customize plot
    ax.set_xlabel('Rounds', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Convergence Comparison: FedAvg vs SFedAvg', fontsize=14)
    ax.grid(True, alpha=0.3)
    if plotted_any:
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No data to plot.\nUpdate "labels" with run directories.',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # Save plot
    plot_path = os.path.join(args.out_dir, "comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
