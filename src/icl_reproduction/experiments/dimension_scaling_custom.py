import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from ..models import LinearTransformer
from ..training import train_model

def main():
    # --- Experiment Settings ---
    DIMENSIONS = [10, 20, 50, 100, 200, 500, 1000, 2000]
    N = 20              # Context length
    B = 100            # Fixed large batch size to ensure learning happens
    flip_p = 0.2        # Label flipping probability
    steps = 300         # Optimization steps
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Running Dimensionality Scaling Experiment on {device}...")
    print(f"Fixed parameters: N={N}, B={B}, noise={flip_p}")

    test_accuracies = []
    ic_accuracies = []

    for d in DIMENSIONS:
        # We scale R with sqrt(d) to keep problem difficulty relative to dimension consistent
        # Alternatively, the paper uses R = d^0.3 or d^0.6 to show specific regimes.
        # Here we use a standard scaling R = 2 * sqrt(d)
        R = d ** 0.3
        
        print(f"\nTraining with Dimension d={d} (R={R:.2f})...")
        
        # 1. Initialize Model
        model = LinearTransformer(d=d).to(device)
        
        # 2. Train
        metrics = train_model(
            model=model,
            d=d, N=N, B=B,
            R_train=R, R_val=R,
            flip_train=0.0,
            flip_val=flip_p,
            steps=steps,
            device=device,
            return_metrics=True
        )
        
        # 3. Collect final metrics
        final_test_acc = np.mean(metrics['val_acc'][-5:])
        final_ic_acc = np.mean(metrics['ic_acc'][-5:])
        
        test_accuracies.append(final_test_acc)
        ic_accuracies.append(final_ic_acc)
        
        print(f"-> Test Acc: {final_test_acc:.3f} | In-Context Acc: {final_ic_acc:.3f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(DIMENSIONS, test_accuracies, 'o-', label='Test Accuracy', color='purple', linewidth=2)
    plt.plot(DIMENSIONS, ic_accuracies, 'x--', label='In-Context Train Accuracy', color='orange', alpha=0.8)
    
    plt.xscale('log')
    plt.xlabel('Input Dimension (d)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Effect of Dimensionality on ICL Performance (B={B}, N={N})', fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    output_path = 'dimension_scaling_custom.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    main()