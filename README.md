# Investigating In-Context Learning of Linear Regression in Transformers

This repository attempts to reproduce and extend the results from _What Can Transformers Learn In Context? A Case Study of Simple Functions?_ (Garg et al. 2023). Specifically, I aim to show that transformers are able to achieve comparable performance to least squares for underparameterized linear regression, min-norm least squares for overparameterized linear regression, and LASSO for sparse linear regression. 

## Structure

The repository is structured as follows:

```
├── src/
│   ├── curriculum.py       # Training curriculum
│   ├── data_sampler.py     # Data & task generation
│   ├── eval.ipynb          # Evaluation versus baselines, visualizations
│   └── losses.py           # Loss functions
│   └── model.py            # Transformer model architecture
│   └── train.py            # Training loop
```

- [todo] go into more detail about what each file does

- [todo] include data generation details
- [todo] training details: batches, epochs, etc.
- [todo] include plots

## Getting Started

1. Clone the repository and switch to the correct branch.

```
git clone https://github.com/Shou-Yue/DSC180a-ICL-A11.
cd DSC180a-ICL-A11
git checkout anish
```

2. Install the required dependencies using `conda`. 

```
conda env create -f environment.yml
conda activate in-context-learning
```

## Training Models

To train, `cd` into the `src` folder. From here, you can train a model for one of three linear regression settings: underparameterized (standard), overparamterized, or sparse. To start one of these training jobs, run the following:

```
python train.py --setting underparameterized
python train.py --setting overparameterized
python train.py --setting sparse
```

- [todo] specify train.py is the entry point
- 

## Evaluation

Once the transformers have been trained, run `eval.ipynb` to compare their performance with the corresponding baselines.

- [todo] what evaluation metrics did we use
- [todo] include graphs
