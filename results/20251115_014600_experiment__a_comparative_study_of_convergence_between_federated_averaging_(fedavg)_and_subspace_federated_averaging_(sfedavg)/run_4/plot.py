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
        "run_fedavg": "FedAvg",
        "run_sfedavg": "SFedAvg",
    }
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
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
                ax1.plot(iterations, means, label=label, color=colors[i], linewidth=2)
                ax1.fill_between(iterations, 
                                np.array(means) - np.array(stds),
                                np.array(means) + np.array(stds),
                                alpha=0.2, color=colors[i])
            
            # Plot training loss
            if "training_loss" in data:
                means = data["training_loss"]["means"]
                stds = data["training_loss"]["stds"]
                iterations = range(len(means))
                ax2.plot(iterations, means, label=label, color=colors[i], linewidth=2)
                ax2.fill_between(iterations, 
                                np.array(means) - np.array(stds),
                                np.array(means) + np.array(stds),
                                alpha=0.2, color=colors[i])
    
    ax1.set_xlabel('Rounds', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Test Accuracy vs Rounds', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Rounds', fontsize=12)
    ax2.set_ylabel('Training Loss', fontsize=12)
    ax2.set_title('Training Loss vs Rounds', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to: {plot_path}")

if __name__ == "__main__":
    main()
