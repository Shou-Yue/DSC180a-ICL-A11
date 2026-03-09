"""
RQ3 Runner: Command-line interface for systematic testing of full transformers.

Two components:
1. Commercial LLM Classification: Test Gemini, Claude, GPT on GMM binary classification
2. Open-weights Regression: Test TinyLlama on linear regression with GD baselines

Mirrors RQ1/RQ2 architecture patterns with parameter sweeps and multi-seed experiments.
"""

import argparse
import os
import sys
import json
import csv
import random
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import numpy as np

# Lazy imports to handle torch dependency issues
torch = None

def _ensure_torch():
    """Lazy load torch to handle compatibility issues"""
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    return torch

# Add parent directory to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_script_dir)
_root = os.path.dirname(_src)
sys.path.insert(0, _src)
sys.path.insert(0, _root)

# Lazy imports
BinaryClassificationDataset = None
get_provider = None
LLMProvider = None

def _ensure_imports():
    global BinaryClassificationDataset, get_provider, LLMProvider
    if BinaryClassificationDataset is None:
        from rq3.dataset import BinaryClassificationDataset as BCD
        from rq3.llm_providers import get_provider as gp, LLMProvider as LP
        BinaryClassificationDataset = BCD
        get_provider = gp
        LLMProvider = LP


@dataclass
class RQ3Config:
    """Configuration for RQ3 experiments"""
    mode: str  # 'commercial' or 'regression'
    d: int  # Dimension
    N: int  # Context length
    R: float  # Signal strength
    provider: Optional[str] = None  # 'gemini', 'claude', 'gpt' (for commercial)
    seed: int = 0
    flip_prob: float = 0.0  # Label flip probability
    num_tasks: int = 10  # Number of tasks per configuration
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def dir_name(self) -> str:
        """Generate directory name for this config"""
        if self.mode == 'commercial':
            provider = self.provider or 'unknown'
            return f"{provider}_d{self.d}_N{self.N}_R{self.R:.2f}_flip{self.flip_prob:.2f}"
        else:  # regression
            return f"d{self.d}_N{self.N}_R{self.R:.2f}"


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_api_keys() -> Dict[str, bool]:
    """Check which API keys are available"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # If dotenv not available, use environment variables as-is
    
    return {
        'gemini': bool(os.getenv('GEMINI_API_KEY')),
        'claude': bool(os.getenv('CLAUDE_API_KEY')),
        'gpt': bool(os.getenv('GPT_API_KEY'))
    }


# ============================================================================
# DUAL METRICS: Query Accuracy vs In-Context Accuracy
# ============================================================================

def compute_dual_metrics(
    dataset,
    provider,
    task_idx: int,
    system_prompt: str = "You are a binary classification model. Output only 0 or 1.",
    icl_sample_size: int = 1
) -> dict:
    """
    Compute both query accuracy and in-context accuracy on same task.
    
    Query Accuracy: Predict unseen query point (validation/generalization)
    In-Context Accuracy: Leave-one-out on context points (memorization/fit)
    
    Returns:
        {
            'query_acc': int (0 or 1),
            'icl_acc': float (0.0 to 1.0),
            'icl_correct': int,
            'icl_total': int
        }
    """
    task = dataset.get_task_dict(task_idx)
    context_x = task['context_x']
    context_y = task['context_y']
    query_x = task['query_x']
    query_y = task['query_y']
    
    # Get numpy arrays
    if hasattr(context_x, 'cpu'):
        ctx_x_np = context_x.cpu().numpy()
        ctx_y_np = context_y.cpu().numpy()
        query_x_np = query_x.cpu().numpy()
    else:
        ctx_x_np = context_x
        ctx_y_np = context_y
        query_x_np = query_x
    
    # ==================== QUERY ACCURACY ====================
    # Standard: predict unseen point from context
    x_str = ', '.join([f'{x:.4f}' for x in query_x_np])
    query_prompt = f"Given the examples below, predict the label for: [{x_str}]\nAnswer (0 or 1):"
    full_query = system_prompt + "\n" + task['formatted_prompt'] + query_prompt
    
    query_pred_text = provider.predict(full_query)
    query_pred = None
    for char in query_pred_text.replace('\n', ' '):
        if char in ['0', '1']:
            query_pred = int(char)
            break
    
    query_correct = 0
    if query_pred is not None:
        query_correct = int(query_pred == int(query_y.item()))
    
    # ==================== IN-CONTEXT ACCURACY ====================
    # Leave-one-out: sample context points, hide each label, and predict it
    icl_correct = 0
    full_context_size = len(ctx_y_np)

    # Sample K points deterministically per task to keep repeated calls consistent.
    sample_size = min(icl_sample_size, full_context_size)
    rng = random.Random(task_idx)
    sample_indices = rng.sample(range(full_context_size), sample_size)
    
    for hide_idx in sample_indices:
        # Build context with label hidden
        ctx_prompt = "Examples:\n"
        for i, (x, y) in enumerate(zip(ctx_x_np, ctx_y_np)):
            x_str = ', '.join([f'{val:.4f}' for val in x])
            if i == hide_idx:
                ctx_prompt += f"  [{x_str}], label=?\n"
            else:
                ctx_prompt += f"  [{x_str}], label={int(y)}\n"
        
        # Query hidden point
        x_hide = ctx_x_np[hide_idx]
        x_hide_str = ', '.join([f'{val:.4f}' for val in x_hide])
        icl_query = f"Predict label for: [{x_hide_str}]\nAnswer (0 or 1):"
        full_icl = system_prompt + "\n" + ctx_prompt + icl_query
        
        icl_pred_text = provider.predict(full_icl)
        icl_pred = None
        for char in icl_pred_text.replace('\n', ' '):
            if char in ['0', '1']:
                icl_pred = int(char)
                break
        
        if icl_pred is not None and icl_pred == int(ctx_y_np[hide_idx]):
            icl_correct += 1
    
    icl_accuracy = icl_correct / sample_size if sample_size > 0 else 0.0
    
    return {
        'query_acc': query_correct,
        'icl_acc': icl_accuracy,
        'icl_correct': icl_correct,
        'icl_total': sample_size
    }


# ============================================================================
# COMMERCIAL MODEL CLASSIFICATION
# ============================================================================

def run_commercial_classification(
    d: int,
    N: int,
    R: float,
    provider_name: str,
    seed: int,
    output_root: str,
    flip_prob: float = 0.0,
    num_tasks: int = 10,
    device: str = "cpu",
    icl_sample_size: int = 1
) -> str:
    """
    Run commercial LLM on GMM binary classification task.
    
    Args:
        d: Dimension
        N: Context length
        R: Signal strength
        provider_name: 'gemini', 'claude', or 'gpt'
        seed: Random seed
        output_root: Root directory for results
        flip_prob: Label flip probability
        num_tasks: Number of tasks to test
        device: Device for tensors
        
    Returns:
        Path to results CSV
    """
    _ensure_torch()
    _ensure_imports()
    set_seed(seed)
    
    # Create output directory
    config = RQ3Config(
        mode='commercial',
        d=d,
        N=N,
        R=R,
        provider=provider_name,
        seed=seed,
        flip_prob=flip_prob,
        num_tasks=num_tasks
    )
    
    base_dir = os.path.join(output_root, "rq3", "commercial", config.dir_name())
    seed_dir = os.path.join(base_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    
    # Save config
    config_file = os.path.join(seed_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Get API key - handle missing key gracefully
    from dotenv import load_dotenv
    load_dotenv()
    
    # Map provider names to environment variable keys
    key_names = {
        'gemini': 'GEMINI_API_KEY',
        'claude': 'CLAUDE_API_KEY',
        'gpt': 'GPT_API_KEY'
    }
    api_key = os.getenv(key_names.get(provider_name, f"{provider_name.upper()}_API_KEY"))
    
    if not api_key:
        print(f"⚠️  API key for {provider_name} not found. Skipping...")
        # Write empty results file
        results_file = os.path.join(seed_dir, "results.csv")
        with open(results_file, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['task_idx', 'prediction', 'true_label', 'correct', 'error'])
        return results_file
    
    try:
        provider = get_provider(provider_name, api_key)
    except Exception as e:
        print(f"❌ Failed to initialize {provider_name} provider: {str(e)}")
        # Write empty results file
        results_file = os.path.join(seed_dir, "results.csv")
        with open(results_file, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['task_idx', 'prediction', 'true_label', 'correct', 'error'])
        return results_file
    
    # Generate dataset
    dataset = BinaryClassificationDataset(
        d=d,
        N=N,
        num_tasks=num_tasks,
        R=R,
        flip_prob=flip_prob,
        seed=seed,
        device=device
    )
    
    # Define system prompt for consistency
    system_prompt = """You are a binary classification model performing in-context learning.
Given labeled examples, predict the label for a new unlabeled point.
Output ONLY 0 or 1, nothing else."""
    
    # Run inference on each task
    results = []
    icl_accs = []
    
    for task_idx in range(num_tasks):
        try:
            # Compute both metrics
            dual_metrics = compute_dual_metrics(dataset, provider, task_idx, system_prompt, icl_sample_size)
            query_correct = dual_metrics['query_acc']
            icl_acc = dual_metrics['icl_acc']
            icl_correct = dual_metrics['icl_correct']
            icl_total = dual_metrics['icl_total']
            
            # Get true label for logging
            task = dataset.get_task_dict(task_idx)
            true_label = int(task['query_y'].item())
            
            # Infer prediction from query_correct
            prediction = 1 if query_correct else 0  # Best guess based on correctness
            
            results.append({
                'task_idx': task_idx,
                'prediction': prediction,
                'true_label': true_label,
                'query_correct': query_correct,
                'icl_accuracy': icl_acc,
                'icl_correct': icl_correct,
                'icl_total': icl_total,
                'error': ''
            })
            icl_accs.append(icl_acc)
        
        except Exception as e:
            results.append({
                'task_idx': task_idx,
                'prediction': -1,
                'true_label': -1,
                'query_correct': 0,
                'icl_accuracy': 0.0,
                'icl_correct': 0,
                'icl_total': 0,
                'error': str(e)
            })
    
    # Save results with both metrics
    results_file = os.path.join(seed_dir, "results.csv")
    with open(results_file, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['task_idx', 'prediction', 'true_label', 'query_correct', 'icl_accuracy', 'icl_correct', 'icl_total', 'error'])
        w.writeheader()
        w.writerows(results)
    
    # Compute summary metrics for both
    query_correct_count = sum(1 for r in results if r['query_correct'] == 1 and r['error'] == '')
    total_count = sum(1 for r in results if r['error'] == '')
    query_accuracy = query_correct_count / total_count if total_count > 0 else 0.0
    
    mean_icl_accuracy = sum(icl_accs) / len(icl_accs) if icl_accs else 0.0
    
    summary = {
        **config.to_dict(),
        'query_correct': query_correct_count,
        'total': total_count,
        'query_accuracy': query_accuracy,
        'mean_icl_accuracy': mean_icl_accuracy,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_file = os.path.join(seed_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ {provider_name.upper()} - d={d}, N={N}, R={R:.2f}, seed={seed}: Query={query_accuracy:.2%}, ICL={mean_icl_accuracy:.2%} ({query_correct_count}/{total_count})")
    
    return results_file


# ============================================================================
# OPEN-WEIGHTS REGRESSION (TinyLlama)
# ============================================================================

def run_tinyllama_regression(
    d: int,
    N: int,
    R: float,
    lower_bound: float,
    upper_bound: float,
    seed: int,
    output_root: str,
    num_tasks: int = 10,
    device: str = "cpu"
) -> str:
    """
    Probe TinyLlama on linear regression tasks.
    
    Args:
        d: Dimension
        N: Context length (number of examples)
        R: Not used for regression, kept for consistency
        lower_bound: Lower bound for input sampling
        upper_bound: Upper bound for input sampling
        seed: Random seed
        output_root: Root directory for results
        num_tasks: Number of tasks to test
        device: Device for model
        
    Returns:
        Path to results CSV
    """
    print(f"⏳ TinyLlama Regression - d={d}, N={N}, seed={seed}...")
    
    set_seed(seed)
    
    # Create output directory
    config = RQ3Config(
        mode='regression',
        d=d,
        N=N,
        R=R,
        seed=seed,
        num_tasks=num_tasks
    )
    
    base_dir = os.path.join(output_root, "rq3", "regression", config.dir_name())
    seed_dir = os.path.join(base_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    
    # Save config
    config_file = os.path.join(seed_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    try:
        # Import TinyLlama probing (will implement in tinyllama_probing.py)
        from icl_reproduction.rq3.tinyllama_probing import eval_linear_regression
        
        # Run regression evaluation
        results = eval_linear_regression(
            d=d,
            N=N,
            num_tasks=num_tasks,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            seed=seed,
            device=device
        )
        
        # Save results
        results_file = os.path.join(seed_dir, "results.csv")
        with open(results_file, 'w', newline='') as f:
            if results:
                w = csv.DictWriter(f, fieldnames=results[0].keys())
                w.writeheader()
                w.writerows(results)
        
        # Compute summary metrics
        if results:
            mse_model = np.mean([r['mse_model'] for r in results])
            mse_gd1 = np.mean([r['mse_gd1'] for r in results])
            ratio = mse_model / mse_gd1 if mse_gd1 > 0 else float('inf')
            
            summary = {
                **config.to_dict(),
                'mse_model_mean': mse_model,
                'mse_gd1_mean': mse_gd1,
                'model_vs_gd1_ratio': ratio,
                'num_tasks': len(results),
                'timestamp': datetime.now().isoformat()
            }
        else:
            summary = {
                **config.to_dict(),
                'num_tasks': 0,
                'timestamp': datetime.now().isoformat()
            }
        
        summary_file = os.path.join(seed_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ TinyLlama Regression - d={d}, N={N}, seed={seed}: MSE={summary.get('mse_model_mean', 'N/A'):.4f}")
        
        return results_file
    
    except ImportError as e:
        print(f"⚠️  TinyLlama probing not yet implemented: {str(e)}")
        # Write empty results file
        results_file = os.path.join(seed_dir, "results.csv")
        with open(results_file, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['task_idx', 'mse_model', 'mse_gd1', 'mse_gdpp'])
        return results_file


# ============================================================================
# MAIN RUNNER
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="RQ3 Runner: Test full transformers on in-context learning"
    )
    
    # Mode selection
    p.add_argument("--mode", default="commercial", choices=["commercial", "regression", "both"],
                   help="Test mode: commercial LLMs, open-weights regression, or both")
    
    # Commercial model parameters
    p.add_argument("--providers", default="gemini,claude,gpt",
                   help="Comma-separated list of LLM providers (gemini,claude,gpt)")
    
    # Dimension sweep
    p.add_argument("--d", default="50,100,500,1000",
                   help="Comma-separated list of dimensions to test")
    
    # Context length sweep
    p.add_argument("--n", default="5,10,20",
                   help="Comma-separated list of context lengths (N) to test")
    
    # Signal strength sweep
    p.add_argument("--r", default="0.3,6.45",
                   help="Comma-separated list of signal strengths (R) to test")
    
    # Regression-specific parameters
    p.add_argument("--regression-bound", default="1.0", type=float,
                   help="Lower/upper bound for regression input sampling ±bound")
    
    # Seed configuration
    p.add_argument("--seeds", default="0,1,2",
                   help="Comma-separated list of random seeds")
    
    # Output directory
    p.add_argument("--output_root", default="results",
                   help="Root directory for results")
    
    # Options
    p.add_argument("--num-tasks", type=int, default=10,
                   help="Number of tasks per configuration")
    p.add_argument("--max-workers", type=int, default=1,
                   help="Maximum parallel workers for commercial models (caution: API rate limits!)")
    p.add_argument("--device", default="cpu",
                   help="Device for tensors (cpu or cuda, defaults to cpu)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print configuration without running experiments")
    
    p.add_argument("--icl-sample-size", type=int, default=1,
               help="Number of context points to sample for in-context accuracy (LOO)")
    
    args = p.parse_args()
    
    # Only ensure torch if we're actually running experiments
    if not args.dry_run:
        _ensure_torch()
        # Auto-detect device if possible
        if args.device == "cpu" and torch is not None:
            if torch.cuda.is_available():
                args.device = "cuda"
    
    # Parse parameters
    d_values = [int(x.strip()) for x in args.d.split(',')]
    n_values = [int(x.strip()) for x in args.n.split(',')]
    r_values = [float(x.strip()) for x in args.r.split(',')]
    seeds = [int(x.strip()) for x in args.seeds.split(',')]
    providers = [p.strip().lower() for p in args.providers.split(',')]
    
    # Check API keys (only warn, don't stop)
    available_apis = check_api_keys()
    missing_providers = [p for p in providers if not available_apis.get(p)]
    if missing_providers and args.mode in ['commercial', 'both']:
        print(f"⚠️  Warning: API keys missing for {missing_providers}. These will be skipped.")
    
    # Create run list
    run_list = []
    
    if args.mode in ['commercial', 'both']:
        print(f"\n📊 Commercial Classification Experiments")
        print(f"  Dimensions: {d_values}")
        print(f"  Context lengths: {n_values}")
        print(f"  Signal strengths: {r_values}")
        print(f"  Providers: {providers}")
        print(f"  Seeds: {seeds}")
        
        for d in d_values:
            for n in n_values:
                for r in r_values:
                    for provider in providers:
                        if available_apis.get(provider):
                            for seed in seeds:
                                run_list.append(('commercial', d, n, r, provider, seed))
    
    if args.mode in ['regression', 'both']:
        print(f"\n📊 Open-weights Regression Experiments (TinyLlama)")
        print(f"  Dimensions: {d_values}")
        print(f"  Context lengths: {n_values}")
        print(f"  Input bounds: ±{args.regression_bound}")
        print(f"  Seeds: {seeds}")
        
        for d in d_values:
            for n in n_values:
                for seed in seeds:
                    run_list.append(('regression', d, n, args.regression_bound, args.regression_bound, seed))
    
    total_runs = len(run_list)
    print(f"\n✨ Total configurations to run: {total_runs}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Not executing experiments")
        for i, run in enumerate(run_list[:5], 1):
            print(f"  {i}. {run}")
        if total_runs > 5:
            print(f"  ... and {total_runs - 5} more")
        return
    
    # Execute runs
    print(f"\n🚀 Starting experiments...\n")
    
    run_start = time.time()
    completed = 0
    failed = 0
    
    if args.mode in ['commercial', 'both']:
        # Use thread pool for commercial models (API calls are I/O bound)
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            
            for run in run_list:
                if run[0] == 'commercial':
                    _, d, n, r, provider, seed = run
                    future = executor.submit(
                        run_commercial_classification,
                        d=d, N=n, R=r, provider_name=provider, seed=seed,
                        output_root=args.output_root, num_tasks=args.num_tasks,
                        device=args.device, icl_sample_size=args.icl_sample_size
                    )
                    futures[future] = run
            
            for future in as_completed(futures):
                try:
                    future.result()
                    completed += 1
                except Exception as e:
                    print(f"❌ Error: {futures[future]} - {str(e)}")
                    failed += 1
    
    # Run regression sequentially (single-threaded to avoid GPU memory issues)
    if args.mode in ['regression', 'both']:
        for run in run_list:
            if run[0] == 'regression':
                _, d, n, bound, _, seed = run
                try:
                    run_tinyllama_regression(
                        d=d, N=n, R=0.0, lower_bound=-bound, upper_bound=bound,
                        seed=seed, output_root=args.output_root,
                        num_tasks=args.num_tasks, device=args.device
                    )
                    completed += 1
                except Exception as e:
                    print(f"❌ Error: {run} - {str(e)}")
                    failed += 1
    
    elapsed = time.time() - run_start
    
    # Print summary
    print(f"\n" + "="*60)
    print(f"✅ Experiments Complete!")
    print(f"   Completed: {completed}/{total_runs}")
    print(f"   Failed: {failed}/{total_runs}")
    print(f"   Time: {elapsed/60:.1f} minutes")
    print(f"   Results: {os.path.abspath(os.path.join(args.output_root, 'rq3'))}")
    print(f"="*60)


if __name__ == '__main__':
    main()
