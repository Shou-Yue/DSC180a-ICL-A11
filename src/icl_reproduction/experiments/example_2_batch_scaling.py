"""
Example 2: Batch Size Scaling Experiment

This script demonstrates:
1. Training LinearTransformer with varying batch sizes
2. Evaluating performance on test set
3. Computing in-context training accuracy
4. Plotting scaling results
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np

from ..models import LinearTransformer
from ..training import GaussianMixtureDataset, train_linear_transformer
from ..evaluation import evaluate_linear_transformer, plot_batch_size_scaling, print_experiment_summary

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main():
    """Run batch size scaling experiment"""
    
    # Parameters
    d = 1000
    N = 20
    R = d ** 0.4
    label_flip_p = 0.2
    batch_sizes = [10, 20, 50, 100, 200, 500, 1000]
    num_epochs = 50
    learning_rate = 0.01
    num_test_tasks = 200

    print("\n" + "="*70)
    print("Batch Size Scaling Experiment")
    print("="*70)
    print(f"Parameters: d={d}, N={N}, R={R}, Label Flip={label_flip_p}")
    print(f"Optimal accuracy: {1.0 - label_flip_p:.2f}\n")

    results = {
        'batch_sizes': batch_sizes,
        'test_acc': [],
        'in_context_acc': [],
    }

    for B in batch_sizes:
        print(f"Training with B={B} tasks...")
        
        # Create datasets
        train_dataset = GaussianMixtureDataset(
            d=d, N=N, B=B, R=R,
            is_validation=False, label_flip_p=0.0
        )
        
        test_dataset = GaussianMixtureDataset(
            d=d, N=N, B=num_test_tasks, R=R,
            is_validation=True, label_flip_p=label_flip_p
        )
        
        # Train model
        model = LinearTransformer(d=d).to(device)
        train_linear_transformer(
            model, train_dataset, num_epochs=num_epochs,
            learning_rate=learning_rate, device=device, verbose=False
        )
        
        # Evaluate
        test_acc, in_context_acc = evaluate_linear_transformer(model, test_dataset, device)
        
        results['test_acc'].append(test_acc)
        results['in_context_acc'].append(in_context_acc)
        
        print(f"  Test Acc: {test_acc:.3f}, In-Context Train Acc: {in_context_acc:.3f}\n")

    # Plot results
    print("Generating plot...")
    fig, ax = plot_batch_size_scaling(
        results['batch_sizes'],
        results['test_acc'],
        results['in_context_acc'],
        optimal_acc=1.0 - label_flip_p,
        title=f"Performance vs Batch Size (d={d}, N={N}, R={R})",
        save_path='batch_size_scaling.png'
    )
    print("✓ Saved batch_size_scaling.png")

    # Print summary
    print_experiment_summary(results, "Batch Size Scaling")


if __name__ == "__main__":
    main()
