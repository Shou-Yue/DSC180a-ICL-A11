In-Context Learning Reproduction: Linear Classification

This repository contains a reproduction and extension of experiments investigating In-Context Learning (ICL) in linear transformers. Specifically, it focuses on the behavior of transformers trained on random linear classification tasks using Gaussian Mixture Models, as discussed in recent theoretical literature (e.g., Frei et al., 2024).

📄 Project Overview

The goal of this project is to analyze how Linear Transformers learn to implement algorithms (like Gradient Descent) in-context. We investigate:

Batch Size Scaling: How the number of pre-training tasks affects generalization.

Dimensionality Scaling: How model performance changes with input dimension ($d$) and signal strength ($R$).

Algorithmic Alignment: Comparing the Transformer's predictions against theoretical baselines like Gradient Descent (GD) and Ridge Regression.

📂 Repository Structure

DSC180a-ICL-A11/
├── requirements.txt          # Python dependencies
├── results/                  # Generated plots and figures
└── src/
    └── icl_reproduction/     # Main package
        ├── models.py         # LinearTransformer, LinearClassifier definitions
        ├── training.py       # Training loops, Data generation (Gaussian Mixtures)
        ├── evaluation.py     # Plotting and metric calculation
        ├── experiments/      # Runnable experiment scripts
        │   ├── example_2_batch_scaling.py
        │   ├── example_3_gd_vs_ridge.py
        │   └── ...
        └── Notebooks/        # Jupyter notebooks for interactive analysis


🚀 Setup & Installation

Clone the repository:

git clone <repository-url>
cd DSC180a-ICL-A11


Create a virtual environment (Recommended):

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


💻 How to Run Experiments

Critical Note: This project uses relative imports. You must run scripts as modules using the -m flag, and you must execute them from the src directory.

Navigate to the source directory:

cd src


Run Batch Size Scaling Experiment:
Investigates how increasing the number of pre-training tasks improves test accuracy.

python -m icl_reproduction.experiments.example_2_batch_scaling


Output: A plot saved to ../results/batch_size_scaling.png.

Run Dimensionality Scaling Experiment:
Tests the model across dimensions $d=10$ to $d=2000$.

python -m icl_reproduction.experiments.dimension_scaling_custom


Run GD vs Ridge Comparison:
Compares the Transformer's output to analytical baselines.

python -m icl_reproduction.experiments.example_3_gd_vs_ridge


📊 Key Results

Batch Scaling: We observe that generalization capability improves log-linearly with the number of unique tasks seen during pre-training.

GD Alignment: The Linear Transformer's predictions closely match one step of Gradient Descent on the logistic loss, validating theoretical findings.

📚 References

Trained Transformer Classifiers Generalize and Exhibit Benign Overfitting In-Context (Frei & Vardi, 2024).

DSC180a Course Materials, UC San Diego.