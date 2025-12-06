#!/usr/bin/env python3
"""
Multi-layer LSA = Multi-step GD Experiment

Trains and evaluates multi-layer Linear Self-Attention models to approximate
multi-step gradient descent on linear regression tasks.

Based on Oswald et al. (2023): "Transformers Learn In-Context by Gradient Descent"
"""
import os
import sys
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.lsa import MultiLayerLSA
from src.evaluation.gd_baseline import (
    generate_linear_regression_task,
    evaluate_on_task
)


def train_lsa_model(model, d, n_points, num_tasks=10000, batch_size=64, 
                    num_epochs=10, lr=1e-3, sigma=0.0, device='cpu'):
    """
    Train LSA model on linear regression tasks.
    
    Args:
        model: MultiLayerLSA model
        d: input dimension
        n_points: number of training points per task
        num_tasks: number of training tasks
        batch_size: batch size for training
        num_epochs: number of epochs
        lr: learning rate
        sigma: label noise level
        device: torch device
    
    Returns:
        model: trained model
        train_losses: list of training losses
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    num_batches = num_tasks // batch_size
    
    print(f"Training {model.num_layers}-layer LSA model...")
    print(f"  d={d}, n={n_points}, tasks={num_tasks}, batch_size={batch_size}")
    print(f"  epochs={num_epochs}, lr={lr}, sigma={sigma}")
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
        for _ in pbar:
            # Generate batch of tasks
            xs, ys, w_true = generate_linear_regression_task(
                d, n_points, batch_size=batch_size, sigma=sigma, device=device
            )
            
            # Generate query points
            query_x = torch.randn(batch_size, 1, d, device=device)
            query_y = torch.bmm(query_x, w_true.unsqueeze(-1)).squeeze(-1).squeeze(-1)
            
            # Add noise to query labels if needed
            if sigma > 0:
                query_y = query_y + torch.randn_like(query_y) * sigma
            
            # Forward pass
            optimizer.zero_grad()
            y_pred = model(xs, ys, query_x)
            loss = criterion(y_pred, query_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.6f}")
    
    return model, train_losses


def evaluate_lsa_model(model, d, n_points, num_eval_tasks=1000, 
                       batch_size=100, eta=0.1, sigma=0.0, device='cpu'):
    """
    Evaluate trained LSA model against GD baseline.
    
    Args:
        model: trained MultiLayerLSA model
        d: input dimension
        n_points: number of training points per task
        num_eval_tasks: number of evaluation tasks
        batch_size: batch size for evaluation
        eta: learning rate for GD baseline
        sigma: label noise level
        device: torch device
    
    Returns:
        results: dict with aggregated metrics
    """
    model = model.to(device)
    model.eval()
    
    mse_lsa_list = []
    mse_gd_list = []
    cosine_sim_list = []
    
    num_batches = num_eval_tasks // batch_size
    
    print(f"\nEvaluating {model.num_layers}-layer LSA model...")
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Evaluation"):
            # Generate batch of tasks
            xs, ys, w_true = generate_linear_regression_task(
                d, n_points, batch_size=batch_size, sigma=sigma, device=device
            )
            
            # Generate query points
            query_x = torch.randn(batch_size, 1, d, device=device)
            query_y = torch.bmm(query_x, w_true.unsqueeze(-1)).squeeze(-1).squeeze(-1)
            
            # Add noise to query labels if needed
            if sigma > 0:
                query_y = query_y + torch.randn_like(query_y) * sigma
            
            # Evaluate
            results = evaluate_on_task(model, xs, ys, query_x, query_y, eta=eta)
            
            mse_lsa_list.append(results['mse_lsa'])
            mse_gd_list.append(results['mse_gd'])
            cosine_sim_list.append(results['cosine_sim'])
    
    # Aggregate results
    results = {
        'num_layers': model.num_layers,
        'mse_lsa_mean': np.mean(mse_lsa_list),
        'mse_lsa_std': np.std(mse_lsa_list),
        'mse_gd_mean': np.mean(mse_gd_list),
        'mse_gd_std': np.std(mse_gd_list),
        'cosine_sim_mean': np.mean(cosine_sim_list),
        'cosine_sim_std': np.std(cosine_sim_list)
    }
    
    print(f"\nResults for {model.num_layers}-layer LSA:")
    print(f"  MSE LSA: {results['mse_lsa_mean']:.6f} ± {results['mse_lsa_std']:.6f}")
    print(f"  MSE GD:  {results['mse_gd_mean']:.6f} ± {results['mse_gd_std']:.6f}")
    print(f"  Cosine similarity: {results['cosine_sim_mean']:.6f} ± {results['cosine_sim_std']:.6f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Multi-layer LSA = Multi-step GD Experiment')
    parser.add_argument('--d', type=int, default=20, help='Input dimension')
    parser.add_argument('--num_layers_list', type=int, nargs='+', default=[1, 2, 4],
                        help='List of num_layers to experiment with')
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='Hidden dimension for LSA (default: d)')
    parser.add_argument('--num_train_tasks', type=int, default=10000,
                        help='Number of training tasks')
    parser.add_argument('--num_eval_tasks', type=int, default=1000,
                        help='Number of evaluation tasks')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--eta', type=float, default=0.1,
                        help='Learning rate for GD baseline')
    parser.add_argument('--sigma', type=float, default=0.0,
                        help='Label noise level (std dev)')
    parser.add_argument('--output_dir', type=str, default='./results/lsa_multilayer',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Calculate n_points = 2d + 1
    n_points = 2 * args.d + 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_results = []
    
    # Run experiments for each num_layers
    for num_layers in args.num_layers_list:
        print(f"\n{'='*60}")
        print(f"Experiment: num_layers = {num_layers}")
        print(f"{'='*60}")
        
        # Create model
        model = MultiLayerLSA(
            d=args.d,
            num_layers=num_layers,
            hidden_dim=args.hidden_dim
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        model, train_losses = train_lsa_model(
            model=model,
            d=args.d,
            n_points=n_points,
            num_tasks=args.num_train_tasks,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            sigma=args.sigma,
            device=device
        )
        
        # Evaluate model
        results = evaluate_lsa_model(
            model=model,
            d=args.d,
            n_points=n_points,
            num_eval_tasks=args.num_eval_tasks,
            batch_size=100,
            eta=args.eta,
            sigma=args.sigma,
            device=device
        )
        
        results['train_losses'] = train_losses
        all_results.append(results)
        
        # Save model checkpoint
        checkpoint_path = output_dir / f'lsa_{num_layers}layer_checkpoint.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'num_layers': num_layers,
            'd': args.d,
            'hidden_dim': args.hidden_dim,
            'train_losses': train_losses,
            'results': results
        }, checkpoint_path)
        print(f"\nSaved checkpoint to {checkpoint_path}")
    
    # Save all results
    results_file = output_dir / 'all_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"All experiments complete!")
    print(f"Results saved to {results_file}")
    print(f"{'='*60}")
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Layers':<10} {'MSE LSA':<20} {'MSE GD':<20} {'Cosine Sim':<20}")
    print("-"*70)
    for res in all_results:
        print(f"{res['num_layers']:<10} "
              f"{res['mse_lsa_mean']:.6f} ± {res['mse_lsa_std']:.4f}    "
              f"{res['mse_gd_mean']:.6f} ± {res['mse_gd_std']:.4f}    "
              f"{res['cosine_sim_mean']:.6f} ± {res['cosine_sim_std']:.4f}")
    print("="*70)


if __name__ == '__main__':
    main()
