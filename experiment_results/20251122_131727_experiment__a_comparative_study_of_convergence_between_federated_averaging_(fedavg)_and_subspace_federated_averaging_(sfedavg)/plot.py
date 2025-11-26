# Plotting for SFedAvg vs FedAvg results

import matplotlib.pyplot as plt
import json
import os
import argparse
import numpy as np

def load_run_metrics(run_baseline_dir):
    json_path = os.path.join(run_baseline_dir, "final_info.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def plot_metrics_across_runs(metric_name, runs_data, labels_map, save_path):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(runs_data))))
    markers = ['o', 's', '^', 'D', '*', 'P', 'X', 'h', 'v', '>', '<']
    for i, (run_dir, metrics) in enumerate(runs_data.items()):
        if metric_name not in metrics:
            continue
        means = np.array(metrics[metric_name]["means"], dtype=float)
        stds = np.array(metrics[metric_name]["stds"], dtype=float)
        iters = np.arange(len(means))
        label = labels_map.get(run_dir, os.path.basename(os.path.dirname(run_dir)))
        ax.plot(
            iters,
            means,
            label=label,
            color=colors[i],
            linewidth=2.5,
            marker=markers[i % len(markers)],
            markevery=max(1, len(means) // 10),
            alpha=0.9
        )
        if stds.size == means.size and np.any(stds > 0):
            ax.fill_between(iters, means - stds, means + stds, alpha=0.15, color=colors[i])

    ax.set_xlabel('Rounds', fontsize=13)
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=13)
    ax.set_title(f'Scenario Comparison: {metric_name.replace("_", " ").title()}', fontsize=15)
    ax.legend(title='Scenarios', loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot SFedAvg vs FedAvg metrics from baseline results")
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save plots')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Map scenario run directories to human-readable labels
    labels = {
        "run_non-iid_label_skew_(dirichlet_α=0.1)": "non-iid_label_skew_(dirichlet_α=0.1)",
        "run_robustness_to_label_noise_(20%_flips)": "robustness_to_label_noise_(20%_flips)",
        "run_scalability:_large_n_and_high-d_features": "scalability:_large_n_and_high-d_features",
        "run_hyperparameter_sensitivity:_no_momentum": "hyperparameter_sensitivity:_no_momentum",
        "run_edge_case:_full-rank_sfedavg_with_aggressive_local_training": "edge_case:_full-rank_sfedavg_with_aggressive_local_training"
    }

    # Load metrics for all runs
    runs_data = {}
    for run_baseline_dir, _label in labels.items():
        data = load_run_metrics(run_baseline_dir)
        if data is not None:
            runs_data[run_baseline_dir] = data

    if not runs_data:
        # No runs found; create an empty placeholder plot
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.set_title('No runs found. Please update labels in plot.py.', fontsize=14)
        ax.set_xlabel('Rounds', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        out_path = os.path.join(args.out_dir, "empty.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved placeholder plot to: {out_path}")
        return

    # Gather union of metric names across runs
    metric_names = set()
    for metrics in runs_data.values():
        metric_names.update(metrics.keys())

    # Plot each metric across runs
    for metric_name in sorted(metric_names):
        save_path = os.path.join(args.out_dir, f"{metric_name}.png")
        plot_metrics_across_runs(metric_name, runs_data, labels_map=labels, save_path=save_path)
        print(f"Saved plot: {save_path}")

if __name__ == "__main__":
    main()
