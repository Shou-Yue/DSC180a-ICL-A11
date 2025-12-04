## Project Overview

This project aims to reproduce the setting in the paper Trained Transformer Classifiers Generalize and Exhibit Benign Overfitting In-Context (https://arxiv.org/abs/2410.01774). The main goal is to show how a single layer linear transformer learns an optimal solution to binary classification problems in-context. Furthermore, we show the presence of benign overfitting, in which the model showcases an ability to memorize noisy (flipped) data but still generalize accurately to unseen data.

There are 4 major steps to this reproduction (found at icl_repro/icl.ipynb). These are Data Generation, Model Architecture, Model Training, and Evaluation and Plots. The full pipeline is handled by this notebook.


## How to Run the Project

### Environment Setup
Conda Instructions:

conda create -n icl_repro python=3.10
conda activate icl_repro
pip install -r requirements.txt

Core Dependencies: Python 3.10+, PyTorch, NumPy, Matplotlib, tqdm

This code was created with the intention of running on CPU. If you are using a GPU, simply change device = "cpu" to device = "cuda". Note that you may have to move models and tensors to the device with .to(device).
