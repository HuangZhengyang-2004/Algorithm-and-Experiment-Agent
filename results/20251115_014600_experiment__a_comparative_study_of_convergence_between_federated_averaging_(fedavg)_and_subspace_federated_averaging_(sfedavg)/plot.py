import matplotlib.pyplot as plt
import json
import os
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save plots')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Dictionary mapping run directories to labels
    labels = {
        "run_1": "FedAvg (IID)",
        "run_2": "SFedAvg (IID, r=100)",
        "run_3": "FedAvg (Non-IID)",
        "run_4": "SFedAvg (Non-IID, r=100)",
        "run_5": "SFedAvg (Non-IID, r=200)"
    }
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    
    for i, (run_dir, label) in enumerate(labels.items()):
        results_path = os.path.join(run_dir, "final_info.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                data = json.load(f)
            
            # Plot test accuracy
            if "test_accuracy" in data:
                means = data["test_accuracy"]["means"]
                stds = data["test_accuracy"]["stds"]
                iterations = range(len(means))
                ax1.plot(iterations, means, label=label, 
                         color=colors[i], linewidth=2)
                ax1.fill_between(iterations, 
                                np.array(means) - np.array(stds),
                                np.array(means) + np.array(stds),
                                alpha=0.2, color=colors[i])
            
            # Plot training loss
            if "training_loss" in data:
                means = data["training_loss"]["means"]
                stds = data["training_loss"]["stds"]
                iterations = range(len(means))
                ax2.plot(iterations, means, label=label, 
                         color=colors[i], linewidth=2)
                ax2.fill_between(iterations, 
                                np.array(means) - np.array(stds),
                                np.array(means) + np.array(stds),
                                alpha=0.2, color=colors[i])
    
    ax1.set_xlabel('Communication Rounds', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Test Accuracy vs Communication Rounds', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    ax2.set_xlabel('Communication Rounds', fontsize=12)
    ax2.set_ylabel('Training Loss', fontsize=12)
    ax2.set_title('Training Loss vs Communication Rounds', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "convergence_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a bar chart for final performance comparison
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prepare data for bar chart
    final_accuracies = []
    final_losses = []
    algorithm_names = []
    
    for run_dir, label in labels.items():
        results_path = os.path.join(run_dir, "final_info.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                data = json.load(f)
            
            if "test_accuracy" in data and "training_loss" in data:
                final_acc = data["test_accuracy"]["means"][-1]
                final_loss = data["training_loss"]["means"][-1]
                final_accuracies.append(final_acc)
                final_losses.append(final_loss)
                algorithm_names.append(label)
    
    # Colors for bars
    bar_colors = plt.cm.tab10(np.linspace(0, 1, len(algorithm_names)))
    
    # Plot final accuracies
    bars1 = ax3.bar(algorithm_names, final_accuracies, color=bar_colors, alpha=0.7)
    ax3.set_xlabel('Algorithm Configuration', fontsize=12)
    ax3.set_ylabel('Final Test Accuracy', fontsize=12)
    ax3.set_title('Final Test Accuracy Comparison', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot final losses
    bars2 = ax4.bar(algorithm_names, final_losses, color=bar_colors, alpha=0.7)
    ax4.set_xlabel('Algorithm Configuration', fontsize=12)
    ax4.set_ylabel('Final Training Loss', fontsize=12)
    ax4.set_title('Final Training Loss Comparison', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Rotate x-axis labels for better readability
    ax3.tick_params(axis='x', rotation=45)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    bar_plot_path = os.path.join(args.out_dir, "final_performance_comparison.png")
    plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create speed comparison chart (rounds to reach accuracy milestones)
    fig3, ax5 = plt.subplots(1, 1, figsize=(12, 6))
    
    accuracy_milestones = [0.5, 0.6, 0.7, 0.8]
    milestone_data = {label: [] for label in algorithm_names}
    
    for run_dir, label in labels.items():
        results_path = os.path.join(run_dir, "final_info.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                data = json.load(f)
            
            if "test_accuracy" in data:
                accuracies = data["test_accuracy"]["means"]
                for milestone in accuracy_milestones:
                    rounds_to_milestone = None
                    for round_num, acc in enumerate(accuracies):
                        if acc >= milestone:
                            rounds_to_milestone = round_num
                            break
                    milestone_data[label].append(rounds_to_milestone if rounds_to_milestone is not None else 200)
    
    x_pos = np.arange(len(accuracy_milestones))
    width = 0.15
    for i, label in enumerate(algorithm_names):
        ax5.bar(x_pos + i*width, milestone_data[label], width, label=label, alpha=0.7)
    
    ax5.set_xlabel('Accuracy Milestone', fontsize=12)
    ax5.set_ylabel('Rounds to Reach Milestone', fontsize=12)
    ax5.set_title('Convergence Speed: Rounds to Reach Accuracy Milestones', fontsize=14)
    ax5.set_xticks(x_pos + width*2)
    ax5.set_xticklabels([f'{milestone*100:.0f}%' for milestone in accuracy_milestones])
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    speed_plot_path = os.path.join(args.out_dir, "convergence_speed_comparison.png")
    plt.savefig(speed_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create relative performance comparison (FedAvg vs SFedAvg)
    fig4, ax6 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Get FedAvg IID as baseline
    baseline_accuracies = None
    for run_dir, label in labels.items():
        if "FedAvg (IID)" in label:
            results_path = os.path.join(run_dir, "final_info.json")
            if os.path.exists(results_path):
                with open(results_path) as f:
                    data = json.load(f)
                baseline_accuracies = data["test_accuracy"]["means"]
                break
    
    if baseline_accuracies is not None:
        for run_dir, label in labels.items():
            results_path = os.path.join(run_dir, "final_info.json")
            if os.path.exists(results_path) and "SFedAvg" in label:
                with open(results_path) as f:
                    data = json.load(f)
                sfedavg_accuracies = data["test_accuracy"]["means"]
                # Calculate performance gap
                performance_gap = [baseline - sfedavg for baseline, sfedavg in 
                                 zip(baseline_accuracies[:len(sfedavg_accuracies)], sfedavg_accuracies)]
                iterations = range(len(performance_gap))
                ax6.plot(iterations, performance_gap, label=label, linewidth=2)
    
    ax6.set_xlabel('Communication Rounds', fontsize=12)
    ax6.set_ylabel('Accuracy Gap (FedAvg IID - Algorithm)', fontsize=12)
    ax6.set_title('Performance Gap: FedAvg IID vs SFedAvg Variants', fontsize=14)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    gap_plot_path = os.path.join(args.out_dir, "performance_gap_comparison.png")
    plt.savefig(gap_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved convergence plot to: {plot_path}")
    print(f"Saved performance comparison plot to: {bar_plot_path}")
    print(f"Saved convergence speed plot to: {speed_plot_path}")
    print(f"Saved performance gap plot to: {gap_plot_path}")

if __name__ == "__main__":
    main()
