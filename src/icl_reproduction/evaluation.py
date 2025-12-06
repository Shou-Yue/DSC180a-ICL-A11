"""
Evaluation and plotting utilities for ICL experiments

This module contains functions for:
- Evaluating models against baselines (GD, Ridge regression)
- Generating plots and visualizations
- Computing metrics and analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import seaborn as sns

sns.set_style("whitegrid")


@torch.no_grad()
def evaluate_linear_transformer(model, test_dataset, device="cpu"):
    """Evaluate LinearTransformer on test dataset"""
    from torch.utils.data import DataLoader
    
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=None, shuffle=False)
    
    test_correct = 0
    test_total = 0
    in_context_correct_all = []
    
    for batch in test_loader:
        context_x, context_y, target_x, target_y = [t.to(device) for t in batch]
        
        pred = model(context_x, context_y, target_x)
        test_preds = (pred > 0).float()
        test_correct += (test_preds == target_y).sum().item()
        test_total += len(target_y)
        
        in_context_preds = model.compute_in_context_preds(context_x, context_y)
        in_context_correct = (in_context_preds == context_y).float().mean(dim=1)
        in_context_correct_all.append(in_context_correct.cpu().numpy())
    
    test_acc = test_correct / test_total
    in_context_train_acc = np.mean(np.concatenate(in_context_correct_all))
    
    return test_acc, in_context_train_acc


def generate_gaussian_mixture_task(d: int, N: int, R: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a single Gaussian mixture classification task"""
    mu_pos = np.random.randn(d)
    mu_neg = np.random.randn(d)
    
    mu_pos = mu_pos / np.linalg.norm(mu_pos) * R
    mu_neg = mu_neg / np.linalg.norm(mu_neg) * R
    
    labels = (np.random.rand(N + 1) > 0.5).astype(float)
    labels_signal = 2 * labels - 1
    
    noise = np.random.randn(N + 1, d)
    x_all = np.where(labels_signal[..., None] > 0, mu_pos, mu_neg) + noise
    
    context_x = x_all[:N]
    target_x = x_all[N]
    context_y = labels[:N]
    target_y = labels[N]
    
    return context_x, context_y, target_x, target_y


def compare_gd_and_ridge(n_tasks: int = 100, d: int = 20, N: int = 40, 
                         R: float = 2.0) -> Dict[str, List[float]]:
    """Compare GD and ridge regression predictions
    
    Returns:
        Dictionary with predictions and metrics
    """
    from .training import gd_solution, ridge_solution
    
    results = {
        'gd_preds': [],
        'ridge_preds': [],
        'gt': [],
        'gd_acc': 0,
        'ridge_acc': 0
    }
    
    for _ in range(n_tasks):
        context_x, context_y, target_x, target_y = generate_gaussian_mixture_task(d, N, R)
        
        # GD prediction
        w_gd, _ = gd_solution(context_x, context_y, k_steps=3, lr=0.01)
        pred_gd = target_x @ w_gd
        results['gd_preds'].append(pred_gd)
        results['gd_acc'] += int(pred_gd > 0) == int(target_y > 0.5)
        
        # Ridge prediction
        w_ridge = ridge_solution(context_x, context_y, lam=1e-3)
        pred_ridge = target_x @ w_ridge
        results['ridge_preds'].append(pred_ridge)
        results['ridge_acc'] += int(pred_ridge > 0) == int(target_y > 0.5)
        
        results['gt'].append(target_y)
    
    results['gd_acc'] /= n_tasks
    results['ridge_acc'] /= n_tasks
    results['gd_preds'] = np.array(results['gd_preds'])
    results['ridge_preds'] = np.array(results['ridge_preds'])
    results['gt'] = np.array(results['gt'])
    
    return results


def plot_training_curves(train_losses: List[float], train_accs: List[float], 
                        val_accs: List[float], save_path: Optional[str] = None):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(train_losses)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(train_accs, label='Train')
    axes[1].plot(val_accs, label='Validation')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, axes


def plot_batch_size_scaling(batch_sizes: List[int], test_acc: List[float], 
                           in_context_acc: List[float], optimal_acc: float = 0.8,
                           title: str = "Performance vs Batch Size",
                           save_path: Optional[str] = None):
    """Plot batch size scaling experiment"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(batch_sizes, in_context_acc, 'o-', color='red', linewidth=2.5, 
            markersize=8, label='In-context train')
    ax.plot(batch_sizes, test_acc, 's-', color='blue', linewidth=2.5, 
            markersize=8, label='Test')
    ax.axhline(y=optimal_acc, color='green', linestyle='-.', linewidth=2.5, 
               label=f'Optimal ({optimal_acc:.2f})')
    
    ax.set_xscale('log')
    ax.set_xlabel('Tasks (B)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim([0.4, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_gd_vs_ridge_comparison(gd_preds: np.ndarray, ridge_preds: np.ndarray,
                               gt: np.ndarray, save_path: Optional[str] = None):
    """Plot comparison between GD and Ridge predictions"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # GD vs Ground Truth
    axes[0].scatter(gt, gd_preds, alpha=0.6, s=40)
    axes[0].plot([-3, 3], [-3, 3], 'r--', linewidth=2, label='y=x')
    axes[0].set_xlabel('Ground Truth', fontsize=11)
    axes[0].set_ylabel('GD Prediction', fontsize=11)
    axes[0].set_title('GD vs Ground Truth', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Ridge vs Ground Truth
    axes[1].scatter(gt, ridge_preds, alpha=0.6, s=40, color='green')
    axes[1].plot([-3, 3], [-3, 3], 'r--', linewidth=2, label='y=x')
    axes[1].set_xlabel('Ground Truth', fontsize=11)
    axes[1].set_ylabel('Ridge Prediction', fontsize=11)
    axes[1].set_title('Ridge Regression vs Ground Truth', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, axes


def plot_accuracy_comparison(methods: List[str], accuracies: List[float], 
                            colors: Optional[List[str]] = None,
                            save_path: Optional[str] = None):
    """Plot accuracy comparison across methods"""
    if colors is None:
        colors = ['blue', 'green', 'orange']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Method Comparison: Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_multiscale_comparison(dimensions: List[int], results_dict: Dict[str, List[float]],
                              save_path: Optional[str] = None):
    """Plot multi-scale comparison across dimensions"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Plot 1: Batch Size
    if 'batch_sizes' in results_dict and 'batch_test_acc' in results_dict:
        ax = axes[0]
        ax.plot(results_dict['batch_sizes'], results_dict['batch_test_acc'], 'o-', linewidth=2)
        ax.set_xscale('log')
        ax.set_xlabel('Batch Size', fontsize=11)
        ax.set_ylabel('Test Accuracy', fontsize=11)
        ax.set_title('Batch Size Scaling', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Signal Strength
    if 'R_values' in results_dict and 'R_test_acc' in results_dict:
        ax = axes[1]
        ax.plot(results_dict['R_values'], results_dict['R_test_acc'], 's-', color='green', linewidth=2)
        ax.set_xlabel('Signal Strength (R)', fontsize=11)
        ax.set_ylabel('Test Accuracy', fontsize=11)
        ax.set_title('Signal Strength Scaling', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Dimension
    if 'dimensions' in results_dict and 'dim_test_acc' in results_dict:
        ax = axes[2]
        ax.plot(results_dict['dimensions'], results_dict['dim_test_acc'], '^-', 
                color='orange', linewidth=2)
        ax.set_xscale('log')
        ax.set_xlabel('Dimension (d)', fontsize=11)
        ax.set_ylabel('Test Accuracy', fontsize=11)
        ax.set_title('Dimension Scaling', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, axes


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute accuracy and other metrics"""
    binary_preds = (preds > 0).astype(int)
    binary_targets = (targets > 0.5).astype(int)
    
    accuracy = np.mean(binary_preds == binary_targets)
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    
    return {
        'accuracy': accuracy,
        'mse': mse,
        'mae': mae
    }


def print_experiment_summary(results: Dict, experiment_name: str = "Experiment"):
    """Print a summary of experiment results"""
    print("\n" + "="*70)
    print(f"{experiment_name} Summary")
    print("="*70)
    
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
            print(f"  {key}: {np.mean(value):.4f} ± {np.std(value):.4f}")
    
    print("="*70 + "\n")
