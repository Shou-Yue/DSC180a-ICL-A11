"""
In-Context Learning (ICL) Reproduction Module

This package provides tools for reproducing experiments from:
"Transformers Learn In-Context by Gradient Descent" (ICML 2023)

Main modules:
  - models: Model implementations (LinearTransformer, etc.)
  - training: Training utilities and datasets
  - evaluation: Evaluation functions and plotting
"""

__version__ = "0.1.0"
__author__ = "ICL Research Team"

from .models import LinearTransformer, SingleLayerTransformer, LinearClassifier
from .training import (
    BinaryClassificationDataset,
    GaussianMixtureDataset,
    train_epoch,
    evaluate,
    train_linear_transformer,
    gd_solution,
    ridge_solution,
    data_gen
)
from .evaluation import (
    evaluate_linear_transformer,
    compare_gd_and_ridge,
    plot_training_curves,
    plot_batch_size_scaling,
    plot_gd_vs_ridge_comparison,
    plot_accuracy_comparison,
    compute_metrics,
    print_experiment_summary
)

__all__ = [
    # Models
    'LinearTransformer',
    'SingleLayerTransformer',
    'LinearClassifier',
    
    # Training
    'BinaryClassificationDataset',
    'GaussianMixtureDataset',
    'train_epoch',
    'evaluate',
    'train_linear_transformer',
    'gd_solution',
    'ridge_solution',
    'data_gen',
    
    # Evaluation
    'evaluate_linear_transformer',
    'compare_gd_and_ridge',
    'plot_training_curves',
    'plot_batch_size_scaling',
    'plot_gd_vs_ridge_comparison',
    'plot_accuracy_comparison',
    'compute_metrics',
    'print_experiment_summary',
]
