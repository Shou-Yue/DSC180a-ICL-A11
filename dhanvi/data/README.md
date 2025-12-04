# Data Generation for Regression Experiments

This project does not rely on an external dataset. Instead, all experiments use **synthetic linear regression tasks** that are generated on the fly within the notebooks. This design choice keeps the experiments controlled, lightweight, and easy to reproduce.

## Task Definition

Each task is a simple linear regression problem of the form:

y = x^T w* + ε


where:
- x ∈ ℝ^d is an input vector,
- w* ∈ ℝ^d is a randomly sampled "teacher" weight vector,
- y ∈ ℝ is a scalar target,
- ε is optional noise (Gaussian in noise experiments).

## Per-Task Sampling Procedure

For each regression task:

1. A ground-truth weight vector `w*` is sampled from a standard normal distribution.
2. A set of **context inputs** `{x_i}_{i=1}^n` is sampled uniformly from a bounded
   range (e.g., [-1, 1]).
3. Context targets are computed as  
   `y_i = x_i^T w*`.
4. A **query input** `x_q` is sampled from the same distribution, and its target  
   `y_q = x_q^T w*` is used for evaluation.
5. In some experiments, Gaussian noise is added to context or query targets in
   order to study robustness.

This setup closely mirrors the synthetic linear regression tasks used in  *Transformers Learn In-Context by Gradient Descent*.


## Implementation Notes

All data-generation logic is implemented directly inside the Jupyter notebooks,
primarily in:

- `01_one_layer_lsa_regression.ipynb`

Because the data is inexpensive to generate and task parameters are explicitly
controlled through code, no static `.csv` or `.npz` files are stored in this
directory. Each notebook generates its own tasks at runtime.

---

## Reproducibility

The regression tasks are fully reproducible:

- All task parameters (input dimension, number of context points, batch size,
  noise level) are explicitly defined near the top of each notebook.

This `data/` directory exists to document the data-generation process and to
provide a clear location for cached datasets if future extensions require
storing generated data on disk.
