# RQ3 Implementation Summary

**Research Question 3:** "Are the phenomena described theoretically for linear attention transformers preserved when using full transformer architectures?"

## Implementation Overview

This document summarizes the complete implementation of RQ3 testing framework with both commercial LLMs and open-weights model probing.

---

## Phase 1: ✅ Command-Line Runner for Systematic Testing

**File**: [src/icl_reproduction/rq3/run_rq3.py](src/icl_reproduction/rq3/run_rq3.py)

### Components
- `RQ3Config` dataclass for experiment configuration management
- `run_commercial_classification()` - Evaluates Gemini, Claude, GPT on GMM binary classification
- `run_tinyllama_regression()` - Probes TinyLlama-1.1B on linear regression tasks
- `check_api_keys()` - Verifies available API credentials
- Full CLI with argument parsing and dry-run mode

### Usage
```bash
# Test configuration without running
python run_rq3.py --dry-run --d 10,50 --n 5,10 --seeds 0

# Run commercial LLM tests
python run_rq3.py --mode commercial --d 50,100,500 --n 5,10,20 --providers gemini,claude,gpt --seeds 0,1,2

# Run TinyLlama probing
python run_rq3.py --mode regression --d 100,500 --n 5,10,20 --seeds 0,1,2

# Run both
python run_rq3.py --mode both --d 100,500 --providers gemini --seeds 0,1
```

### Results Structure
```
results/rq3/
├── commercial/
│   ├── gemini_d50_N5_R6.45_flip0.00/
│   │   └── seed_0/
│   │       ├── config.json
│   │       ├── summary.json (accuracy, metrics)
│   │       └── results.csv (per-task predictions)
│   ├── claude_...
│   └── gpt_...
└── regression/
    └── d100_N5_R0.00/
        └── seed_0/
            ├── config.json
            ├── summary.json (MSE metrics)
            └── results.csv (task-level results)
```

---

## Phase 2: ✅ Data Generation

**File**: [src/icl_reproduction/rq3/dataset.py](src/icl_reproduction/rq3/dataset.py)

### BinaryClassificationDataset Class
Wraps `data_gen()` function for convenient task generation:

```python
from rq3.dataset import BinaryClassificationDataset

# Generate tasks
dataset = BinaryClassificationDataset(
    d=100,           # Dimension
    N=20,            # Context length (in-context examples)
    num_tasks=10,    # Generate 10 tasks
    R=6.45,          # Signal strength
    flip_prob=0.0,   # Label noise
    seed=42
)

# Access individual task
task = dataset[0]  # Dict: context_x, context_y, query_x, query_y

# Get task with formatted prompt for LLMs
task_dict = dataset.get_task_dict(task_idx=0)
prompt = task_dict['formatted_prompt']
```

### Methods
- `__init__()` - Initialize with parameters, auto-generate tensors
- `__len__()` - Returns number of tasks
- `__getitem__()` - Access task as dictionary
- `format_for_prompt()` - Get human-readable string for LLM prompts
- `get_task_dict()` - Get task dict + formatted prompt

**Test Suite**: [src/icl_reproduction/rq3/test_phase2.py](src/icl_reproduction/rq3/test_phase2.py)

---

## Phase 3: ✅ Open-Weights Model Probing

**File**: [src/icl_reproduction/rq3/tinyllama_probing.py](src/icl_reproduction/rq3/tinyllama_probing.py)

### Linear Regression Evaluation
```python
from rq3.tinyllama_probing import eval_linear_regression

results = eval_linear_regression(
    d=100,
    N=10,
    num_tasks=5,
    seed=42,
    device="cuda"
)
# Returns: List[Dict] with keys:
# - task_idx, mse_model, mse_gd1, mse_gdpp, pred_model, pred_gd1, pred_gdpp, true_output
```

### Baseline Comparisons
- **GD-1**: One-step gradient descent from zero (validates learning)
- **GD++**: Preconditioned gradient descent (Hessian-based)
- **Model**: TinyLlama-1.1B predictions

### OOD Evaluation
```python
from rq3.tinyllama_probing import eval_ood_sine_tasks

ood_results = eval_ood_sine_tasks(
    d=100,
    N=10,
    num_tasks=5,
    frequency=1.0,
    amplitude=1.0,
    seed=42
)
# Test if linear-trained models generalize to non-linear functions
```

**Test Suite**: [src/icl_reproduction/rq3/test_phase3.py](src/icl_reproduction/rq3/test_phase3.py)

---

## Phase 4: ✅ Results Aggregation & Plotting

**File**: [src/icl_reproduction/experiments/plots.py](src/icl_reproduction/experiments/plots.py)

### RQ3 Plotting Functions

#### Provider Comparison Heatmap
```python
from experiments.plots import plot_rq3_provider_comparison_heatmap

plot_rq3_provider_comparison_heatmap(
    results_root="results",
    output_file="results/rq3/_plots/provider_heatmap.png"
)
```
Creates 3-panel heatmap (Gemini | Claude | GPT) showing accuracy across d×N grid.

#### Accuracy vs Signal Strength
```python
from experiments.plots import plot_rq3_accuracy_vs_r

plot_rq3_accuracy_vs_r(
    results_root="results",
    output_file="results/rq3/_plots/accuracy_vs_r.png"
)
```
Line plot with error bands showing how accuracy changes with signal strength R.

#### Regression Baselines
```python
from experiments.plots import plot_rq3_regression_baselines

plot_rq3_regression_baselines(
    results_root="results",
    output_file="results/rq3/_plots/regression_baselines.png"
)
```
Bar plot comparing TinyLlama MSE vs GD-1 vs GD++ baselines.

#### Generate All Plots
```python
from experiments.plots import generate_all_rq3_plots

generate_all_rq3_plots(
    results_root="results",
    plots_root="results/rq3/_plots"
)
```
Master function that generates all RQ3 visualizations.

---

## Phase 5: ✅ Streamlit App Refactoring

**File**: [src/icl_reproduction/rq3/app.py](src/icl_reproduction/rq3/app.py)

### Updated Architecture
- Uses new `BinaryClassificationDataset` from Phase 2
- Consolidated imports and removed unused dependencies
- Cleaner prompt construction with system-level instructions
- Better error handling and API key management

### Three Modes
1. **Single Test**: Manual parameter input, get single prediction
2. **Batch Testing**: Parameter sweeps, parallel execution
3. **Results Dashboard**: Visualize results with charts

### Configuration
Create `.env` file in rq3/ directory:
```
GEMINI_API_KEY=your-key
CLAUDE_API_KEY=your-key
GPT_API_KEY=your-key
```

### Usage
```bash
cd src/icl_reproduction/rq3
streamlit run app.py
```

---

## Complete Experimental Pipeline

### Step 1: Run Commercial LLM Tests
```bash
cd src/icl_reproduction
python rq3/run_rq3.py --mode commercial \
  --d 50,100,500,1000 \
  --n 5,10,20 \
  --r 0.3,6.45 \
  --seeds 0,1,2 \
  --num-tasks 10
```
**Expected output**: `results/rq3/commercial/` directory with provider subdirectories

### Step 2: Run TinyLlama Regression Tests
```bash
python rq3/run_rq3.py --mode regression \
  --d 100,500,1000 \
  --n 5,10,20 \
  --seeds 0,1,2 \
  --num-tasks 10
```
**Expected output**: `results/rq3/regression/` directory with results

### Step 3: Generate All Plots
```bash
python -c "from experiments.plots import generate_all_rq3_plots; generate_all_rq3_plots('results')"
```
**Expected output**: 
- `results/rq3/_plots/provider_comparison_heatmap.png`
- `results/rq3/_plots/accuracy_vs_r.png`
- `results/rq3/_plots/regression_baselines.png`

### Step 4: Interactive Exploration
```bash
streamlit run rq3/app.py
```
Explore predictions, test individual configurations, view dashboard.

---

## Cost Estimation

**Default Configuration** (full sweep):
- 4 dimensions × 3 context lengths × 2 signal strengths × 3 seeds × 10 tasks = 2,160 API calls
- **Per provider**: ~$0.10-$0.30
- **Total (3 providers)**: ~$0.30-$0.90

**Recommended starting configuration** (cost-effective):
```bash
python rq3/run_rq3.py --mode commercial \
  --d 100,500 \
  --n 5,10 \
  --r 6.45 \
  --seeds 0 \
  --num-tasks 5 \
  --providers gemini
```
- Single provider, 2 dimensions, 2 context lengths, 1 seed, 5 tasks = 20 calls
- **Cost**: Free (Gemini free tier) to $0.001

---

## File Checklist

- ✅ [run_rq3.py](src/icl_reproduction/rq3/run_rq3.py) - CLI runner (580 lines)
- ✅ [dataset.py](src/icl_reproduction/rq3/dataset.py) - Data generation (180 lines)
- ✅ [tinyllama_probing.py](src/icl_reproduction/rq3/tinyllama_probing.py) - TinyLlama evaluation (398 lines)
- ✅ [llm_providers.py](src/icl_reproduction/rq3/llm_providers.py) - API integration (existing)
- ✅ [app.py](src/icl_reproduction/rq3/app.py) - Streamlit interface (refactored)
- ✅ [plots.py](src/icl_reproduction/experiments/plots.py) - RQ3 plotting (extended)
- ✅ [.env](src/icl_reproduction/rq3/.env) - API key configuration
- ✅ [test_phase2.py](src/icl_reproduction/rq3/test_phase2.py) - Dataset tests
- ✅ [test_phase3.py](src/icl_reproduction/rq3/test_phase3.py) - Probing tests

---

## Key Design Decisions

1. **Lazy imports**: Torch loaded only when needed to avoid setup errors in dry-run
2. **Consistent structure**: Mirrors RQ1 patterns (configs, CSV outputs, aggregation)
3. **Provider abstraction**: Single interface for Gemini, Claude, GPT
4. **Result persistence**: JSON configs + CSV metrics for reproducibility
5. **Flexible sweeps**: Independent control over d, N, R, seeds via CLI
6. **Error handling**: Graceful API key handling, partial result saving

---

## Next Steps (Optional Enhancements)

1. **Add W&B logging**: Track experiments with Weights & Biases
2. **Extend OOD testing**: Polynomial, sinusoidal, chaotic tasks
3. **Cross-model comparison**: Include Llama 2, Mistral alongside TinyLlama
4. **Statistical testing**: Significance tests for provider differences
5. **Visualization dashboard**: Interactive Plotly/Dash interface

---

**Implementation Date**: March 6, 2026  
**Team**: RQ3 Research Implementation  
**Status**: ✅ COMPLETE & TESTED
