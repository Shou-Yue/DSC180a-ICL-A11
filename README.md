# Analysis of In-Context Learning on Linear Classification Tasks

## Project Overview

## Structure

├── README.md
├── icl_classification
│   ├── __init__.py
│   ├── checkpoints                                 # Model checkpoints
│   ├── classification_icl.py                       # Training loop for simple transformer model
│   ├── eval_and_plot.py                            # Where models are evaluated and plots are created
│   ├── gpt.py                                      # Training loop for GPT-2 architecture
│   ├── plots                                       # Where result plots are stored
│   ├── test.py                                     # Testing code functionality
│   ├── test_results                                # Results of simple transformer
│   └── test_results_gpt                            # Results of GPT-2 transformer
├── notebooks
│   ├── gpt_test.ipynb                              # Notebook for running with GPT-2 arhitecture
│   ├── test_B_values.ipynb                         # Notebook containing experiments with differing B values (# tasks)
│   └── test_d_values.ipynb                         # Notebook containing experiments with differing D values (model dimension)
└── requirements.txt                                # Packages required to run project

## How to Run