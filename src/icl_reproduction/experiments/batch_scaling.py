import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Relative imports assuming running as module: python -m icl_reproduction.experiments.batch_size_custom
from ..models import LinearTransformer
from ..training import train_model
from ..evaluation import print_experiment_summary

def main():
    # --- Experiment Settings ---
    BATCH_SIZES = [5, 10, 20, 50, 100, 200, 500, 1000]
    d = 20           # Input dimension
    N = 10           # Context length
    R = 5.0          # Signal strength
    flip_p = 0.1     # Label flipping probability
    steps = 500      # Optimization steps
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Running Batch Size Scaling Experiment on {device}...")
    print(f"Fixed parameters: d={d}, N={N}, R={R}, noise={flip_p}")

    test_accuracies = []
    ic_accuracies = []

    for B in BATCH_SIZES:
        print(f"\nTraining with Batch Size B={B}...")
        
        # 1. Initialize Model
        model = LinearTransformer(d=d).to(device)
        
        # 2. Train
        metrics = train_model(
            model=model,
            d=d, N=N, B=B,
            R_train=R, R_val=R,
            flip_train=0.0,      # Usually clean pre-training
            flip_val=flip_p,     # Noisy test
            steps=steps,
            device=device,
            return_metrics=True
        )
        
        # 3. Collect final metrics (averaging last 5 steps for stability)
        final_test_acc = np.mean(metrics['val_acc'][-5:])
        final_ic_acc = np.mean(metrics['ic_acc'][-5:])
        
        test_accuracies.append(final_test_acc)
        ic_accuracies.append(final_ic_acc)
        
        print(f"-> Test Acc: {final_test_acc:.3f} | In-Context Acc: {final_ic_acc:.3f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    # Plot Test Accuracy
    plt.plot(BATCH_SIZES, test_accuracies, 'o-', label='Test Accuracy (Generalization)', linewidth=2, color='blue')
    
    # Plot In-Context Train Accuracy
    plt.plot(BATCH_SIZES, ic_accuracies, 's--', label='In-Context Train Accuracy', linewidth=2, color='red', alpha=0.7)
    
    # Reference line for random guessing
    plt.axhline(y=0.5, color='gray', linestyle=':', label='Random Guessing')

    plt.xscale('log')
    plt.xlabel('Number of Pre-training Tasks (Batch Size B)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Effect of Batch Size on ICL Performance (d={d}, N={N})', fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Save plot
    output_path = 'batch_size_scaling_custom.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    main()