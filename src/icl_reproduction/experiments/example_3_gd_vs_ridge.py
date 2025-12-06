"""
Example 3: GD vs Ridge Regression Comparison

This script demonstrates:
1. Comparing GD and Ridge regression on classification tasks
2. Analyzing prediction agreement
3. Plotting comparison visualizations
"""

import numpy as np
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from ..evaluation import (
    compare_gd_and_ridge, 
    plot_gd_vs_ridge_comparison,
    plot_accuracy_comparison,
    compute_metrics,
    print_experiment_summary
)

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


def main():
    """Compare GD and Ridge regression"""
    
    print("\n" + "="*70)
    print("GD vs Ridge Regression Comparison")
    print("="*70 + "\n")
    
    # Parameters
    n_tasks = 200
    d = 20
    N = 40
    R = 2.0

    print(f"Comparing on {n_tasks} tasks:")
    print(f"  Dimension: d={d}")
    print(f"  Context size: N={N}")
    print(f"  Signal strength: R={R}\n")

    # Run comparison
    print("Running comparison...")
    results = compare_gd_and_ridge(n_tasks=n_tasks, d=d, N=N, R=R)
    
    # Compute metrics
    gd_metrics = compute_metrics(results['gd_preds'], results['gt'])
    ridge_metrics = compute_metrics(results['ridge_preds'], results['gt'])
    
    print("\nResults:")
    print(f"  GD Accuracy: {gd_metrics['accuracy']:.3f}")
    print(f"  GD MSE: {gd_metrics['mse']:.4f}")
    print(f"\n  Ridge Accuracy: {ridge_metrics['accuracy']:.3f}")
    print(f"  Ridge MSE: {ridge_metrics['mse']:.4f}")
    
    # Compute agreement
    gd_binary = (results['gd_preds'] > 0).astype(int)
    ridge_binary = (results['ridge_preds'] > 0).astype(int)
    agreement = np.mean(gd_binary == ridge_binary)
    print(f"\n  GD-Ridge Prediction Agreement: {agreement:.3f}")
    
    # Plot comparisons
    print("\nGenerating plots...")
    
    fig1, axes1 = plot_gd_vs_ridge_comparison(
        results['gd_preds'],
        results['ridge_preds'],
        results['gt'],
        save_path='gd_vs_ridge_predictions.png'
    )
    print("✓ Saved gd_vs_ridge_predictions.png")
    
    fig2, ax2 = plot_accuracy_comparison(
        ['GD', 'Ridge Regression'],
        [gd_metrics['accuracy'], ridge_metrics['accuracy']],
        colors=['orange', 'green'],
        save_path='gd_vs_ridge_accuracy.png'
    )
    print("✓ Saved gd_vs_ridge_accuracy.png")
    
    # Print summary
    summary = {
        'GD Accuracy': gd_metrics['accuracy'],
        'Ridge Accuracy': ridge_metrics['accuracy'],
        'GD MSE': gd_metrics['mse'],
        'Ridge MSE': ridge_metrics['mse'],
        'Prediction Agreement': agreement
    }
    print_experiment_summary(summary, "GD vs Ridge Regression")


if __name__ == "__main__":
    main()
