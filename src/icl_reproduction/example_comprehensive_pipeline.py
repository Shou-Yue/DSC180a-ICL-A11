"""
Comprehensive Example: Full ICL Pipeline

This script demonstrates a complete pipeline:
1. Data generation
2. Model training
3. Baseline evaluation
4. Results visualization
5. Metrics computation
"""

import torch
import numpy as np
from torch.utils.data import DataLoader

from models import LinearTransformer, LinearClassifier
from training import (
    BinaryClassificationDataset, 
    GaussianMixtureDataset,
    train_epoch,
    evaluate,
    train_linear_transformer,
    data_gen
)
from evaluation import (
    evaluate_linear_transformer,
    compare_gd_and_ridge,
    plot_training_curves,
    plot_accuracy_comparison,
    plot_gd_vs_ridge_comparison,
    compute_metrics,
    print_experiment_summary
)

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def experiment_1_simple_training():
    """Experiment 1: Simple LinearTransformer training"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Simple LinearTransformer Training")
    print("="*70)
    
    # Parameters
    config = {
        'd_input': 50,
        'N': 20,
        'num_tasks': 500,
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 1e-2,
        'R': 1.0,
        'flip_prob': 0.1
    }
    
    print(f"\nConfig: {config}\n")
    
    # Create datasets
    train_dataset = BinaryClassificationDataset(
        d=config['d_input'],
        N=config['N'],
        num_tasks=config['num_tasks'],
        R=config['R'],
        flip_prob=config['flip_prob']
    )
    
    val_dataset = BinaryClassificationDataset(
        d=config['d_input'],
        N=config['N'],
        num_tasks=200,
        R=config['R'],
        flip_prob=0.0
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Create and train model
    model = LinearTransformer(d=config['d_input']).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
    
    train_losses = []
    train_accs = []
    val_accs = []
    
    print("Training...")
    for epoch in range(config['num_epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                  f"Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
    
    # Plot
    print("\nPlotting...")
    plot_training_curves(train_losses, train_accs, val_accs, 
                        save_path='exp1_training_curves.png')
    print("✓ Saved exp1_training_curves.png")
    
    return {
        'final_train_loss': train_losses[-1],
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1],
        'max_val_acc': max(val_accs)
    }


def experiment_2_gd_vs_ridge():
    """Experiment 2: Compare GD and Ridge regression"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: GD vs Ridge Regression Comparison")
    print("="*70)
    
    print("\nComparing on 100 tasks (d=20, N=40, R=2.0)...")
    
    results = compare_gd_and_ridge(n_tasks=100, d=20, N=40, R=2.0)
    
    gd_metrics = compute_metrics(results['gd_preds'], results['gt'])
    ridge_metrics = compute_metrics(results['ridge_preds'], results['gt'])
    
    print(f"\nGD Results:")
    print(f"  Accuracy: {gd_metrics['accuracy']:.3f}")
    print(f"  MSE: {gd_metrics['mse']:.4f}")
    
    print(f"\nRidge Regression Results:")
    print(f"  Accuracy: {ridge_metrics['accuracy']:.3f}")
    print(f"  MSE: {ridge_metrics['mse']:.4f}")
    
    # Plot
    print("\nPlotting...")
    plot_gd_vs_ridge_comparison(
        results['gd_preds'],
        results['ridge_preds'],
        results['gt'],
        save_path='exp2_gd_vs_ridge.png'
    )
    print("✓ Saved exp2_gd_vs_ridge.png")
    
    plot_accuracy_comparison(
        ['GD', 'Ridge'],
        [gd_metrics['accuracy'], ridge_metrics['accuracy']],
        save_path='exp2_accuracy_comparison.png'
    )
    print("✓ Saved exp2_accuracy_comparison.png")
    
    return {
        'GD Accuracy': gd_metrics['accuracy'],
        'Ridge Accuracy': ridge_metrics['accuracy'],
        'GD MSE': gd_metrics['mse'],
        'Ridge MSE': ridge_metrics['mse']
    }


def experiment_3_scaling():
    """Experiment 3: Batch size scaling"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Batch Size Scaling")
    print("="*70)
    
    d = 500
    N = 20
    R = 0.5
    label_flip_p = 0.2
    batch_sizes = [50, 100, 200, 500]
    num_epochs = 30
    
    print(f"\nParameters: d={d}, N={N}, R={R}, Label Flip={label_flip_p}")
    print(f"Testing batch sizes: {batch_sizes}\n")
    
    test_accs = []
    ic_accs = []
    
    for B in batch_sizes:
        print(f"Training with B={B}...")
        
        train_dataset = GaussianMixtureDataset(
            d=d, N=N, B=B, R=R,
            is_validation=False, label_flip_p=0.0
        )
        
        test_dataset = GaussianMixtureDataset(
            d=d, N=N, B=100, R=R,
            is_validation=True, label_flip_p=label_flip_p
        )
        
        model = LinearTransformer(d=d).to(device)
        train_linear_transformer(model, train_dataset, num_epochs=num_epochs, device=device)
        
        test_acc, ic_acc = evaluate_linear_transformer(model, test_dataset, device)
        test_accs.append(test_acc)
        ic_accs.append(ic_acc)
        
        print(f"  Test Acc: {test_acc:.3f}, In-Context Acc: {ic_acc:.3f}")
    
    from evaluation import plot_batch_size_scaling
    
    print("\nPlotting...")
    plot_batch_size_scaling(
        batch_sizes, test_accs, ic_accs,
        optimal_acc=1.0 - label_flip_p,
        title=f"Scaling Analysis (d={d}, N={N})",
        save_path='exp3_batch_scaling.png'
    )
    print("✓ Saved exp3_batch_scaling.png")
    
    return {
        'batch_sizes': batch_sizes,
        'test_accuracies': test_accs,
        'in_context_accuracies': ic_accs
    }


def main():
    """Run all experiments"""
    print("\n" + "="*70)
    print("ICL Comprehensive Pipeline")
    print("="*70)
    print(f"Device: {device}\n")
    
    # Run experiments
    exp1_results = experiment_1_simple_training()
    exp2_results = experiment_2_gd_vs_ridge()
    exp3_results = experiment_3_scaling()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print_experiment_summary(exp1_results, "Experiment 1: Training")
    print_experiment_summary(exp2_results, "Experiment 2: GD vs Ridge")
    print_experiment_summary(exp3_results['batch_sizes'], "Experiment 3: Scaling")
    
    print("\n✓ All experiments completed successfully!")
    print("Generated files:")
    print("  - exp1_training_curves.png")
    print("  - exp2_gd_vs_ridge.png")
    print("  - exp2_accuracy_comparison.png")
    print("  - exp3_batch_scaling.png")


if __name__ == "__main__":
    main()
