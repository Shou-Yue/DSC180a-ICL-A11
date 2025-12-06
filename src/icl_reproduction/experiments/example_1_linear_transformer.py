"""
Example 1: Train LinearTransformer on Binary Classification

This script demonstrates:
1. Creating a binary classification dataset
2. Training a LinearTransformer model
3. Evaluating and plotting results
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from ..models import LinearTransformer
from ..training import BinaryClassificationDataset, train_epoch, evaluate
from ..evaluation import plot_training_curves, plot_accuracy_comparison, print_experiment_summary

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main():
    """Train and evaluate LinearTransformer"""
    
    # Configuration
    config = {
        'd_input': 20,
        'N': 40,
        'num_tasks': 1000,
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 1e-2,
        'R': 20 * 0.5,
        'flip_prob': 0.2
    }

    print("\n" + "="*70)
    print("Training LinearTransformer for Binary Classification")
    print("="*70)
    print(f"Config: {config}\n")

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
        num_tasks=config['num_tasks'] // 5,
        R=config['R'],
        flip_prob=0.0  # No noise in validation
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # Create model
    model = LinearTransformer(d=config['d_input']).to(device)
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

    # Training loop
    train_losses = []
    train_accs = []
    val_accs = []

    print("Starting training...")
    for epoch in range(config['num_epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{config['num_epochs']}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Acc: {val_acc:.4f}\n")

    # Plot results
    print("Generating plots...")
    fig, axes = plot_training_curves(
        train_losses, train_accs, val_accs,
        save_path='training_curves.png'
    )
    print("✓ Saved training_curves.png")

    # Print summary
    results = {
        'final_train_loss': train_losses[-1],
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1],
        'max_val_acc': max(val_accs)
    }
    print_experiment_summary(results, "LinearTransformer Training")


if __name__ == "__main__":
    main()
