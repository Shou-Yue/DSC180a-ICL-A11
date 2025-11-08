# In-Context Learning Replication — Linear Regression & Classification

This repository reproduces experiments testing whether **large language models (LLMs)** such as LLaMA and GPT-5 exhibit **gradient-descent-like in-context learning (ICL)** on simple linear tasks.  
Each prompt contains \( N = 2d + 1 \) labeled examples and a test query. The model’s prediction is compared to an analytical **one-step gradient descent baseline (GD-1)**.

---

## 📁 Folder Structure

.
├── Classification_Model/ # Linear classification notebooks & scripts
├── Regression_Model/ # Linear regression notebooks & scripts

Each subfolder contains a runnable Jupyter notebook.

---

## Requirements

### Python & Hardware
- **Python** 3.10–3.11  
- **GPU (optional)** — tested on NVIDIA GTX 1080 Ti  
  - fp16 supported  
  - no bfloat16 required  

### Python Packages
```bash
typing-extensions >= 4.12.2
transformers >= 4.46.0
tokenizers >= 0.20.1
accelerate >= 0.34.2
safetensors >= 0.4.3
sentencepiece >= 0.2.0
matplotlib >= 3.8.0
scikit-learn >= 1.3.0
openai >= 1.40.0
packaging >= 23.2
```
To manually install packages
```bash
pip install "typing-extensions>=4.12.2" "transformers>=4.46.0" \
            "tokenizers>=0.20.1" "accelerate>=0.34.2" "safetensors>=0.4.3" \
            "sentencepiece>=0.2.0" "matplotlib>=3.8.0" \
            "scikit-learn>=1.3.0" "openai>=1.40.0" "packaging>=23.2"
```

## Model Backends

You can run the experiments using either Hugging Face or OpenAI models.

Option A — Hugging Face (local)
```bash
export LLM_BACKEND=hf
export HF_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

Option B - OpenAI (GPT-5)
```bash
export LLM_BACKEND=openai
export OPENAI_API_KEY=sk-...
export GPT5_MODEL=gpt-5-thinking
```
Note, all results presently in the classification jupyter notebook was obtained through using the openai api

## Running the Experiments

### Classification: 
1. Open `Classification_Model/Classification_Pretrain.ipynb`
2. Verify backend environment variables
3. Run all cells
   the notebook will:
   - Sample $N = 2d+1$ training examples
   - Build prompts and query the LLM
   - Compute accuracy for LLM and accuracy for GD-1
   - Plot accuracy vs input scale $\alpha$

### Regression:
1. Open `icl_with_llama.ipynb`
2. Verify backend environment variables
3. Run all cells
   the notebook will:
   - Sample regression tasks with $y = W^\top x$
   - Compare LLM and GD-1 predictions
   - Plot MSE against $\alpha$

## Key Parameters

| Parameter | Description | Default / Typical Value |
|------------|--------------|--------------------------|
| `d` | Feature dimension of input vectors | 10 |
| `N` | Number of in-context examples per task (usually `2*d + 1`) | 21 |
| `ALPHAS` | Input range scaling factor controlling data magnitude | `[0.5, 1.0, 1.5, 2.0]` |
| `ETA` | Learning rate used in GD-1 baseline | 1.0 |
| `n_tasks` | Number of sampled tasks per α value | 10 (for testing), 1000 (for replication) |
| `BACKEND` | Model interface backend: `"hf"` for Hugging Face or `"openai"` for GPT-5 | `"openai"` |
| `HF_MODEL` | Hugging Face model name if backend is local | `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` |
| `GPT5_MODEL` | OpenAI model name when using GPT-5 backend | `"gpt-5"` |
| `device` | Hardware device used for inference (`cuda` or `cpu`) | `"cuda"` if available |
| `alpha` | Range bound for sampling input features (controls difficulty) | variable |
| `metric` | Evaluation metric: MSE for regression, accuracy/F1 for classification | — |

Example:
```python
D = 10
N = 2*D + 1
ALPHAS = [0.5, 1.0, 1.5, 2.0]
n_tasks = 1000
ETA = 1.0
```

## What the Code Does

1. **Generates synthetic linear tasks**
   - **Regression:** $y = W^\top x$
   - **Classification:** $y = \mathrm{sign}(W^\top x)$
   - A random weight vector $W \in \mathbb{R}^d$ is sampled for each task.
   - Inputs $x_i$ are drawn uniformly from $[-\alpha, \alpha]^d$.
   - $N = 2d + 1$ labeled pairs $(x_i, y_i)$ are created.

2. **Builds in-context prompts**
   - Each prompt contains all $N$ input–output examples and one test query $x_{\text{test}}$.
   - Example (regression):
     ```
     Input: [0.2, -0.4, 0.1] → Output: -0.8  
     Input: [0.9, 0.3, -0.5] → Output: 1.7  
     ...
     Predict the output for: [0.5, -0.2, 0.3]
     ```
   - The LLM must infer the relationship between inputs and outputs using only these examples.

3. **Predicts the test output from the LLM**
   - The model (LLaMA or GPT-5) generates a prediction $\hat{y}_{\text{LLM}}$ for the query.
   - Predictions are interpreted as numeric values (regression) or as class labels $-1$ or $+1$ (classification).

4. **Computes the GD-1 baseline**
   - A single gradient descent update $W_1 = \frac{\eta}{N} X^\top y$
   - Regression prediction $\hat y_{\mathrm{GD\text{-}1}} = W_1^\top x_{\mathrm{test}}$
   - Classification prediction $\hat y_{\mathrm{GD\text{-}1}} = \text{sign}\big(W_1^\top x_{\mathrm{test}}\big)$
5. **Compares results**
   - **Regression:** Mean Squared Error (MSE)
   - **Classification:** Accuracy
   - Metrics are averaged across sampled tasks and plotted against the input scale parameter $\alpha$.

6. **Plots outcomes**
   - Produces performance curves showing how closely LLM predictions match GD-1 results.
   - Illustrates how in-context learning strength changes with task difficulty (controlled by $\alpha$).

## Replicating Results

1. Choose backend (Hugging Face or OpenAI)

2. Set parameters:
example:
```python
n_tasks = 1000
D = 10
N = 2*D + 1
ALPHAS = [0.5, 1.0, 1.5, 2.0]
ETA = 1.0
```
3. Run all cells

4. Export plots and metrics

## References
- 
## Maintainer
Shoutai Yue
