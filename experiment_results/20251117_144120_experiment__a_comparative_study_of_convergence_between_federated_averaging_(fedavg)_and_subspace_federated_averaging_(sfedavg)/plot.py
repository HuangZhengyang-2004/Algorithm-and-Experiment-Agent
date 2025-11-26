import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def load_run_metrics(run_dir):
    """
    Load metrics from a run directory containing final_info.json.
    Returns a dict: metric_name -> {"means": list, "stds": list}
    """
    path = os.path.join(run_dir, "final_info.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

def save_placeholder_plot(out_dir: str, message: str):
    """Save a placeholder PNG to out_dir with the provided message."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12)
    ax.set_axis_off()
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {plot_path}")

def make_subplots_for_metrics(metric_names):
    """
    Create one subplot per metric name.
    """
    n = len(metric_names)
    plt.style.use('seaborn-v0_8-whitegrid')
    if n == 0:
        # No metrics: caller should handle separately
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        return fig, [ax]
    fig, axes = plt.subplots(n, 1, figsize=(10, 5 + 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    return fig, axes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save plots')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Dictionary mapping run directories to labels.
    # Update these paths to your actual run directories before plotting.
    labels = {
        "run_fedavg": "FedAvg",
        "run_sfedavg": "SFedAvg",
    }

    # Load all runs
    runs_data = {}
    for run_dir, label in labels.items():
        data = load_run_metrics(run_dir)
        if data is not None:
            runs_data[run_dir] = {"label": label, "data": data}

    if not runs_data:
        # Save a placeholder plot instead of exiting without creating a PNG
        save_placeholder_plot(
            args.out_dir,
            "No valid runs found.\nPlace final_info.json in run directories (e.g., run_fedavg, run_sfedavg) and re-run.",
        )
        return

    # Determine metric names (union across runs)
    metric_names = set()
    for r in runs_data.values():
        metric_names.update(r["data"].keys())
    metric_names = sorted(metric_names)

    if len(metric_names) == 0:
        save_placeholder_plot(args.out_dir, "No metrics found in final_info.json files.")
        return

    fig, axes = make_subplots_for_metrics(metric_names)
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs_data)))

    # Plot each metric
    for ax_idx, metric_name in enumerate(metric_names):
        ax = axes[ax_idx]
        for color_idx, (run_dir, rinfo) in enumerate(runs_data.items()):
            data = rinfo["data"]
            if metric_name not in data:
                continue
            means = np.asarray(data[metric_name].get("means", []), dtype=float)
            stds = np.asarray(data[metric_name].get("stds", []), dtype=float)
            if means.size == 0:
                continue
            iterations = np.arange(len(means))
            lbl = f"{rinfo['label']}"

            ax.plot(iterations, means, label=lbl, color=colors[color_idx], linewidth=2)
            if stds.shape == means.shape:
                ax.fill_between(iterations, means - stds, means + stds, color=colors[color_idx], alpha=0.2)

        # Axis labels per metric
        ax.set_xlabel('Rounds', fontsize=12)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle('Federated Learning Comparison', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path = os.path.join(args.out_dir, "comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {plot_path}")

if __name__ == "__main__":
    main()
