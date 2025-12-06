# Analysis of In-Context Learning on Linear Classification Tasks

## Project Overview

This project investigates how linear transformer models perform in-context learning on synthetic binary classification tasks, with a focus on understanding when and why benign overfitting occurs. By systematically varying dimensionality, signal strength, and label noise, we identify the conditions under which the model successfully generalizes versus when it exhibits harmful overfitting or unstable behavior.

## Structure

```
├── README.md
├── icl_classification
│   ├── __init__.py
│   ├── checkpoints                                 # Model checkpoints
│   ├── classification_icl.py                       # Training loop for simple transformer model
│   ├── eval_and_plot.py                            # Where models are evaluated and plots are created
│   ├── gpt.py                                      # Training loop for GPT-2 architecture
│   ├── plots                                       # Where result plots are stored
│   ├── test.py                                     # Testing code functionality
│   ├── results                                     # Where the results are stored
├── notebooks
│   ├── gpt_test.ipynb                              # Notebook for running with GPT-2 arhitecture
│   ├── test_B_values.ipynb                         # Notebook containing experiments with differing B values (# tasks)
│   └── test_d_values.ipynb                         # Notebook containing experiments with differing D values (dimension)
└── requirements.txt                                # Packages required to run project
```

## How to Run

1. Set up the Conda environment and install the required packages by running the below commands:

```bash
conda create --name icl-classification python=3.12
conda actiavte icl-classification
pip install -r requirments.txt
```

2. Run the code and view the results

    There are two ways to do this:
    1. Using the jupyter notebooks. In the _notebooks_ directory, open the notebook you want to use and run the cells. Results should be printed and corresponding plots will be created.
    2. Using the python scripts. First, cd into the correct directory:
        ```bash
        cd icl_classification
        ```
        Then, run the scripts with the following commands:
        - For the simple transformer, run
            ```py
            python3 classification_icl.py
            ```
        - For the GPT-2 transformer, run
            ```py
            python3 gpt.py
            ```
        At the bottom of the scripts, the experiment configurations and parameters can be modified.
        To evaluate and plot the results, run the following command:
        ```bash
        python3 eval_and_plot.py
        ```
        The resulting plots will be stored in the _plots_ directory.