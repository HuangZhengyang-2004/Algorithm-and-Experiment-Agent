# Plotting for federated experiment comparisons
# Reads multiple run directories and plots metrics with shaded std regions.

import matplotlib.pyplot as plt
import json
import os
import argparse
import numpy as np
from typing import Dict

def load_metrics(run_dir: str) -> Dict:
    """
    Load final_info.json from a run directory. Supports either:
      - {run_dir}/final_info.json
      - {run_dir}/baseline/final_info.json
    Returns parsed JSON dict or None if not found.
    """
    cand1 = os.path.join(run_dir, "final_info.json")
    cand2 = os.path.join(run_dir, "baseline", "final_info.json")
    path = cand1 if os.path.exists(cand1) else cand2
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

    # Dictionary mapping run directories to labels.
    # Update this mapping to include more runs as needed.
    labels = {
        "run_non-iid_label_skew_(dirichlet)_–_convergence_stability": "non-iid_label_skew_(dirichlet)_–_convergence_stability",
        "run_robustness_to_label_noise_–_noisy_clients": "robustness_to_label_noise_–_noisy_clients",
        "run_scalability_–_many_clients,_sparse_participation": "scalability_–_many_clients,_sparse_participation",
        "run_hyperparameter_sensitivity_–_learning_rate_and_momentum": "hyperparameter_sensitivity_–_learning_rate_and_momentum",
        "run_edge_case_–_deep_local_training_with_tiny_participation": "edge_case_–_deep_local_training_with_tiny_participation",
    }

    # Auto-discover run directories containing results to ensure plots are generated
    # Scans immediate subdirectories for final_info.json
    discovered = {}
    for entry in os.listdir("."):
        if os.path.isdir(entry):
            data_path1 = os.path.join(entry, "final_info.json")
            data_path2 = os.path.join(entry, "baseline", "final_info.json")
            if os.path.exists(data_path1) or os.path.exists(data_path2):
                discovered[entry] = f"{entry}"
    # Merge discovered with predefined labels (predefined entries take precedence)
    for run_dir, label in discovered.items():
        if run_dir not in labels:
            labels[run_dir] = label

    # Initialize matplotlib with professional style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Aggregate data by metric across runs
    # metrics_data[metric_name] = list of (iterations, means, stds, label, color)
    metrics_data: Dict[str, list] = {}

    # Colors for different runs
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(labels))))

    for i, (run_dir, label) in enumerate(labels.items()):
        data = load_metrics(run_dir)
        if data is None:
            print(f"Warning: No results found in {run_dir} (expected final_info.json). Skipping.")
            continue

        for metric_name, metric in data.items():
            means = np.array(metric.get("means", []), dtype=float)
            stds = np.array(metric.get("stds", []), dtype=float)
            iterations = np.arange(len(means))
            metrics_data.setdefault(metric_name, []).append((iterations, means, stds, label, colors[i]))

    # If no data was found, still produce a placeholder plot to satisfy requirements
    if len(metrics_data) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, "No data found to plot.\nEnsure final_info.json exists in run_dir or run_dir/baseline.", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        plot_path = os.path.join(args.out_dir, "comparison.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved empty plot to: {plot_path}")
        return

    # Create a separate figure per metric
    for metric_name, series in metrics_data.items():
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for (iterations, means, stds, label, color) in series:
            ax.plot(iterations, means, label=label, color=color, linewidth=2)
            if stds is not None and len(stds) == len(means):
                ax.fill_between(iterations, means - stds, means + stds, alpha=0.2, color=color)

        # Customize plot
        ax.set_xlabel('Rounds', fontsize=12)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Comparison: {metric_name}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save plot
        safe_name = metric_name.replace('/', '_').replace(' ', '_')
        plot_path = os.path.join(args.out_dir, f"{safe_name}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to: {plot_path}")

if __name__ == "__main__":
    main()
