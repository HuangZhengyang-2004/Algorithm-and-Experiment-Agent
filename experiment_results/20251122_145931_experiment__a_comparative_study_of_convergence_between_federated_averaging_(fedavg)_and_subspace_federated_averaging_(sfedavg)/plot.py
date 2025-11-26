# Plot file - Federated Averaging vs Subspace Federated Averaging comparison

import matplotlib.pyplot as plt
import json
import os
import argparse
import numpy as np

# Dictionary mapping run directories to labels
# Users can update this to include multiple experiment outputs
labels = {
    "run_non-iid_label_skew_stress_test": "non-iid_label_skew_stress_test",
    "run_label_noise_robustness": "label_noise_robustness",
    "run_scalability:_many_clients_and_high-dimensional_features": "scalability:_many_clients_and_high-dimensional_features",
    "run_hyperparameter_stress:_aggressive_lr_and_deep_local_steps": "hyperparameter_stress:_aggressive_lr_and_deep_local_steps",
    "run_projection_rank_edge_case_(r=1)": "projection_rank_edge_case_(r=1)",
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

    # Colors per scenario (consistent across figures)
    num_runs = max(1, len(labels))
    base_colors = plt.cm.tab10(np.linspace(0, 1, min(10, num_runs)))
    colors = np.vstack([base_colors] * int(np.ceil(num_runs / base_colors.shape[0])))[:num_runs]

    # Metric groups and styles for methods
    metric_groups = [
        ("train_loss", "Train Loss", {"FedAvg": "train_loss_FedAvg", "SFedAvg": "train_loss_SFedAvg"}),
        ("test_accuracy", "Test Accuracy", {"FedAvg": "test_accuracy_FedAvg", "SFedAvg": "test_accuracy_SFedAvg"}),
        ("comm_floats", "Communication (floats)", {"FedAvg": "comm_floats_FedAvg", "SFedAvg": "comm_floats_SFedAvg"}),
    ]
    method_styles = {"FedAvg": "-", "SFedAvg": "--"}

    any_plotted_overall = False
    saved_paths = []

    for group_key, y_label, keys in metric_groups:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        plotted_any = False

        for i, (run_dir, label) in enumerate(labels.items()):
            data = load_results(run_dir)
            if data is None:
                continue

            run_color = colors[i]
            for method_name, metric_key in keys.items():
                if metric_key not in data:
                    continue

                metric_data = data[metric_key]
                means = np.array(metric_data.get("means", []), dtype=np.float64)
                stds = np.array(metric_data.get("stds", []), dtype=np.float64)
                if means.size == 0:
                    continue

                iterations = np.arange(means.size)
                ax.plot(
                    iterations,
                    means,
                    label=f"{label} - {method_name}",
                    color=run_color,
                    linestyle=method_styles.get(method_name, "-"),
                    linewidth=2
                )
                if stds.size == means.size and np.any(stds > 0):
                    ax.fill_between(iterations, means - stds, means + stds, alpha=0.15, color=run_color)
                plotted_any = True

        # Customize plot aesthetics
        ax.set_xlabel('Rounds', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'Convergence Comparison: FedAvg vs SFedAvg - {y_label}', fontsize=14)
        ax.grid(True, alpha=0.3)
        if plotted_any:
            ax.legend(fontsize=9, ncol=1)
            any_plotted_overall = True
        else:
            ax.text(0.5, 0.5, 'No data to plot.\nEnsure "labels" maps run dirs containing baseline/final_info.json.',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        # Save per-metric comparison plot
        plot_path = os.path.join(args.out_dir, f"comparison_{group_key}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths.append(plot_path)

    # Report saved plots
    if any_plotted_overall:
        for p in saved_paths:
            print(f"Saved plot to: {p}")
    else:
        print("No plots generated: no data found in specified run directories.")


if __name__ == "__main__":
    main()
