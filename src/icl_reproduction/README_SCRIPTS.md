"""
In-Context Learning (ICL) Reproduction - Scripts Module

This directory contains modularized Python scripts for reproducing experiments from
"Transformers Learn In-Context by Gradient Descent" (ICML 2023).

Directory Structure:
====================

Core Modules:
  - models.py          : Model implementations (LinearTransformer, SingleLayerTransformer, LinearClassifier)
  - training.py        : Training utilities, datasets, and baselines (GD, Ridge)
  - evaluation.py      : Evaluation functions and plotting utilities

Example Scripts:
  - example_1_linear_transformer.py    : Train and evaluate LinearTransformer
  - example_2_batch_scaling.py         : Batch size scaling experiment
  - example_3_gd_vs_ridge.py           : Compare GD vs Ridge regression

Original Notebooks:
  - Notebooks/icl.ipynb                                  : Core ICL implementation
  - Notebooks/icl_linear_class_single_layer.ipynb        : Single-layer attention transformer
  - Notebooks/gd_icl_comparison.ipynb                    : GD vs LinearTransformer
  - Notebooks/icl_with_llama_linear_class.ipynb          : LLaMA model probing
  - Notebooks/sla_model_vs_task_size.ipynb               : Scaling experiments


Module Details:
===============

1. models.py
-----------
Contains three model classes:

  - LinearTransformer(d)
    * One-layer linear transformer for binary classification
    * Implements: logit = (W @ context_mean) · target_x
    * Key method: compute_in_context_preds() for evaluating memorization
    
  - SingleLayerTransformer(d_input, d_model, dropout)
    * Attention-based transformer with single layer
    * Incorporates labels into value representations
    * Suitable for binary classification tasks
    
  - LinearClassifier(d)
    * Simple linear classifier for in-context learning
    * Learns task-general transformation W
    * Also has compute_in_context_preds() method


2. training.py
--------------
Training utilities and data generation:

  Datasets:
    - BinaryClassificationDataset(d, N, num_tasks, R, flip_prob)
      * Generates binary classification tasks from Gaussian mixtures
      * Supports label flipping for noise injection
      
    - GaussianMixtureDataset(d, N, B, R, is_validation, label_flip_p)
      * Full-batch dataset for scaling experiments
      * Used for batch size scaling analysis
  
  Training Functions:
    - train_epoch(model, dataloader, optimizer, device)
      * Single epoch training loop
      
    - evaluate(model, dataloader, device)
      * Validation/test evaluation
      
    - train_linear_transformer(model, train_dataset, num_epochs, ...)
      * Specialized training for LinearTransformer
      
    - train_model(model, d, N, B, R_train, R_val, ...)
      * Training for LinearClassifier
  
  Baseline Methods:
    - gd_solution(X, y, k_steps, lr)
      * k-step gradient descent
      
    - ridge_solution(X, y, lam)
      * Closed-form ridge regression
      
    - data_gen(d, N, B, R, flip_prob, device, seed)
      * Generate batches of classification tasks


3. evaluation.py
----------------
Evaluation and visualization functions:

  Evaluation:
    - evaluate_linear_transformer(model, test_dataset, device)
      * Returns test accuracy and in-context training accuracy
      
    - compare_gd_and_ridge(n_tasks, d, N, R)
      * Compares GD and Ridge regression predictions
      
    - compute_metrics(preds, targets)
      * Returns accuracy, MSE, MAE
  
  Plotting:
    - plot_training_curves(train_losses, train_accs, val_accs)
      * Training dynamics visualization
      
    - plot_batch_size_scaling(batch_sizes, test_acc, in_context_acc)
      * Scaling experiment results
      
    - plot_gd_vs_ridge_comparison(gd_preds, ridge_preds, gt)
      * Prediction comparison scatter plots
      
    - plot_accuracy_comparison(methods, accuracies)
      * Bar plot of method accuracies
      
    - plot_multiscale_comparison(dimensions, results_dict)
      * Multi-panel comparison across different scales
  
  Utilities:
    - generate_gaussian_mixture_task(d, N, R)
      * Generate single classification task
      
    - print_experiment_summary(results, experiment_name)
      * Pretty-print results


Usage Examples:
===============

1. Train LinearTransformer:
   $ python example_1_linear_transformer.py

2. Run Batch Size Scaling:
   $ python example_2_batch_scaling.py

3. Compare GD vs Ridge:
   $ python example_3_gd_vs_ridge.py


Custom Scripts:
===============

To create a custom script, follow this pattern:

```python
import torch
import numpy as np
from models import LinearTransformer
from training import GaussianMixtureDataset, train_linear_transformer
from evaluation import evaluate_linear_transformer, plot_batch_size_scaling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Create dataset
dataset = GaussianMixtureDataset(d=100, N=20, B=500, R=1.0)

# 2. Create and train model
model = LinearTransformer(d=100).to(device)
train_linear_transformer(model, dataset, num_epochs=50, device=device)

# 3. Evaluate
test_acc, in_context_acc = evaluate_linear_transformer(model, dataset, device)

# 4. Plot
plot_batch_size_scaling([100, 200, 500], [0.6, 0.7, 0.8], [0.5, 0.6, 0.7])
```


Key Parameters:
===============

d (dimension):       Ambient dimension of input space (usually 100-1000)
N (context size):    Number of context examples per task (usually 10-40)
B (batch size):      Number of training tasks (usually 10-1000)
R (signal strength): Signal-to-noise ratio controlling class separation
flip_prob:           Label noise probability (0.0-0.2)
d_model:             Hidden dimension for attention transformer (usually 64-256)


Requirements:
==============
- torch >= 1.9.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0


Notes:
======
- All scripts use CUDA if available, otherwise CPU
- Random seeds are set for reproducibility (SEED=42)
- Plots are saved to PNG files in current directory
- Models use SGD optimizer with learning rate 1e-2 to 1e-3
