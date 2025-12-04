# In-Context Learning Replication — Linear Regression

This repository reproduces experiments testing whether large language models (LLMs) such as GPT-5 exhibit gradient-descent-like in-context learning (ICL) on simple linear regression tasks.

Each task provides N = 2d + 1 input–output examples and one test query. The model’s prediction is compared to a closed-form one-step gradient descent baseline (GD-1).

---

# Repository Structure

.
├── src/
│   ├── __init__.py
│   ├── tasks.py         # Regression task generator
│   ├── baselines.py     # GD-1 closed-form baseline
│   ├── prompting.py     # Prompt builder for GPT-5
│   ├── llm.py           # OpenAI API wrapper
│   ├── eval.py          # Evaluate regression suites
│   └── experiments.py   # OOD scaling experiments
│
├── scripts/
│   ├── __init__.py
│   └── run_ood.py       # Run out-of-distribution scaling experiment
│
├── results/             # Created automatically on first run
│   ├── logs/            # CSV metric logs
│   └── figures/         # Saved plots
│
├── .env                 # User-provided API keys (not committed)
├── .env.example         # Template for users
└── README.md

results/ starts empty and is populated when experiments run.

---

# Installation

## Python Version
Python 3.10–3.11

## Create Virtual Environment
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

## Install Dependencies
pip install -r requirements.txt

Manual package list:
- typing-extensions >= 4.12.2
- matplotlib >= 3.8.0
- scikit-learn >= 1.3.0
- openai >= 1.40.0
- python-dotenv >= 1.0.1
- numpy

---

# OpenAI Setup

Create a file named .env at the project root:

OPENAI_API_KEY=sk-...
GPT5_MODEL=gpt-5
LLM_BACKEND=openai

Never commit .env.  
A .env.example template is included for collaborators.

---

# Running the Regression Experiments

## Out-of-Distribution (OOD) Scaling Experiment

Run:

python scripts/run_ood.py

This experiment:
- Samples regression tasks with y = W^T x
- Builds in-context prompts
- Queries GPT-5 for predictions
- Computes MSE and R² vs GD-1
- Saves:
  results/logs/ood_scaling.csv
  results/figures/ood_scaling_r2.png

---

# Parameters

Parameter | Description | Default
--------- | ----------- | --------
d | Input dimension | 10
N | Number of in-context examples (2*d + 1) | 21
alphas | Input scale values | [0.5, 1.0, 1.5, 2.0]
eta | GD-1 learning rate | 1.0
n_tasks | Tasks per α | 5–1000
GPT5_MODEL | OpenAI model name | gpt-5

Example:

d = 10  
N = 2*d + 1  
alphas = [0.5, 1.0, 1.5, 2.0]  
eta = 1.0  
n_tasks = 1000  

---

# What the Code Does

## 1. Task Generation
- Samples W in R^d
- Samples inputs x_i uniformly from [-alpha, alpha]^d
- Computes outputs y_i = W^T x_i
- Samples a test input x_test

## 2. Prompt Construction
(x1 -> y1)
(x2 -> y2)
...
Predict the output for: x_test

GPT-5 returns a numeric prediction.

## 3. GD-1 Baseline
W_1 = (eta / N) * X^T y  
Prediction: y_hat = W_1^T x_test

## 4. Evaluation Metrics
- Mean Squared Error (MSE)
- R² score

## 5. Outputs
Saved under:
results/logs/
results/figures/

---

# How to Replicate Results

1. Install dependencies  
2. Create .env with your OpenAI key  
3. Run:

python scripts/run_ood.py

4. Inspect output:
results/logs/ood_scaling.csv  
results/figures/ood_scaling_r2.png  

---

# References

- Oswald et al. "Transformers Learn In-Context by Gradient Descent" (2023)
- Garg et al. "What Can Transformers Learn In-Context?" (NeurIPS)
- Frei & Vardi (2024)

---

# Maintainer
Shoutai Yue
