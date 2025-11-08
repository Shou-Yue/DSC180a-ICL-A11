# Replicating ICL Binary Classification on a Single-Layer Transformer

This repository contains a notebook and supporting code that reproduce the binary classification experiments used to study in-context learning (ICL) with a single-layer attention transformer. The goal is to match the ICL experimental setup (Gaussian-mixture / signal-plus-noise binary tasks) 
## Goals

- Re-implement a simple, single-layer attention transformer that uses context (x,y) pairs to make a binary prediction for a query point.
- Recreate the synthetic binary classification tasks used in ICL-style experiments (Gaussian mixture / signal-plus-noise generation, label flipping noise, controllable SNR R).
- Run controlled experiments and generate plots that show performance vs: number of training tasks (batch size), input dimension, and context sequence length.
- Provide reproducible code and a simple notebook-based workflow so experiments can be re-run and extended.

## Files of interest

- `Leo/icl_linear_class_single_layer.ipynb` — Main working notebook. Contains:
  - Dataset class `BinaryClassificationDataset` (GMM-like generation with parameter `R` and label flip prob).
  - `SingleLayerTransformer` — the single-layer attention model used for experiments.
  - Training/evaluation helpers (`train_epoch`, `evaluate`).
  - Experiment helper functions: `test_context_sizes`, `test_input_dimensions`, `analyze_task_scaling`, and `analyze_batch_size_performance` (batch-size analysis with error bars).


## Dataset / Task generation

Two equivalent views are used across the codebase:

- Gaussian-mixture style (older notebook code): sample two class means `mu1, mu2` scaled by `R`, then sample points from `N(mu_c, I)` and assign labels 0/1 accordingly. Flip labels with `flip_prob` to simulate noise.
- Signal-plus-noise style (ICL-paper-like alternative): sample random direction `theta` and create `x = noise + R * s * theta` where `s` is a scalar signal; label by sign of projection on `theta`.

The notebook currently uses the GMM-style generator implemented in `BinaryClassificationDataset`, parameterized by `d`, `N`, `R`, and `flip_prob`.

Important parameters:

- d: input dimension
- N: context size (number of (x,y) pairs seen at evaluation time)
- num_tasks / B: number of tasks used during training 
- R: signal-to-noise ratio (controls separability)
- flip_prob: probability of flipping a label (label noise)

## Model

`SingleLayerTransformer` is a minimal attention-based model:

- Projects context and query to d_model, computes attention from query to context keys, forms values using context embeddings modulated by labels, and maps attention-weighted result to a logit.
- Prediction thresholding is done via sigmoid > 0.5.

This intentionally small architecture mirrors the single-layer experiments in the ICL literature and is suitable for controlled experiments.

## Experiments included

1. Training curves: standard train/validation curves recorded across epochs (loss + accuracy).
2. Context-size sweep (`test_context_sizes`): test accuracy as a function of context length N.
3. Input-dimension sweep (`test_input_dimensions`): how performance varies with d.
4. Task-count / Batch-size sweep (`analyze_task_scaling` and `analyze_batch_size_performance`): how model performance (in-context train accuracy and test accuracy) scales with the number of training tasks. The notebook's `analyze_batch_size_performance` reproduces the methodology in `by computing means and standard errors and plotting 95% CIs.

Notes on the `analyze_batch_size_performance` implementation:

- For reproducible error bars we run several independent short trainings per batch-size and report mean ± 1.96 * stderr. This is a lighter-weight approximation of the checkpoint-based evaluation in `eval_and_plot.py`.
- The notebook runs fewer samples by default to keep runtimes reasonable. If you have more compute, increase the repetitions / num_samples for tighter error bars.

## How to run (Windows PowerShell)

1. Create a Python environment (recommended: 3.9+). Example using venv:

   powershell
   python -m venv .venv; .\.venv\Scripts\Activate.ps1

2. Install dependencies (adjust versions to match your CUDA/torch setup if you need GPU):

   pip install --upgrade pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # adjust for CUDA or install cpu wheel
   pip install matplotlib numpy

3. Open and run the notebook in VS Code or Jupyter. In VS Code you can open `Leo/icl_linear_class_single_layer.ipynb` and run the cells sequentially.


## Quick diagnostic tips

- If training accuracy goes down as task count increases, this is often expected: with many diverse tasks the model learns a more general strategy (less overfit to particular tasks), which reduces per-task train accuracy while improving robustness/generalization.
- To verify trade-offs: reduce `flip_prob` to 0.0 and/or increase `R` to make tasks easier; if the drop disappears, it's consistent with the generalization vs memorization trade-off.

## Reproducibility & tips

- Set the random seeds (`SEED` in the notebook) and make sure `torch.manual_seed` and `np.random.seed` are set as shown.
- If using GPU, ensure the torch wheel matches your CUDA version.
- Increase the number of independent training repeats in `analyze_batch_size_performance` to reduce error-bar noise.

## Next steps / suggestions

- Convert the notebook experiments to a small script that saves checkpoints per (d, B, N, R) combination. 
- Add a small harness that parallelizes independent repeats of the same experiment to better estimate error bars.

