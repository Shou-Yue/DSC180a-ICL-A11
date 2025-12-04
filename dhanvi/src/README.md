# Source Code Directory

This directory contains the implementation of all experiments for the regression** branch of the In-Context Learning capstone project.

At present, all experiment logic is implemented directly in Jupyter notebooks
rather than in standalone Python modules. This choice is intentional, as the
project focuses on exploratory and interpretability-driven analysis.

---

## Contents

```
src/
└── notebooks/
|    ├── 01_one_layer_lsa_regression.ipynb
|    ├── 02_scaling_experiments.ipynb
|    └── 03_noise_experiments.ipynb
```

---

## Notes

- The notebooks in `src/notebooks/` are designed to be run top-to-bottom and are
  self-contained.
- Any shared utilities or helper functions may be factored out into Python files
  within `src/` in future extensions of the project.
- For details on data generation, see `data/README.md`.

This structure keeps all experiment code centralized while maintaining a clear
separation between **code**, **data documentation**, and **results**.
