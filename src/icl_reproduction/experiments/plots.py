import json
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _scan_run_dirs(results_root: str, exp_name: str, subdir: Optional[str] = None) -> List[str]:
    base = os.path.join(results_root, "rq1", exp_name)
    if subdir:
        base = os.path.join(base, subdir)
    if not os.path.isdir(base):
        return []
    dirs = []
    for name in os.listdir(base):
        if name.startswith("_"):
            continue
        path = os.path.join(base, name)
        if not os.path.isdir(path):
            continue
        for item in os.listdir(path):
            seed_path = os.path.join(path, item)
            if os.path.isdir(seed_path) and item.startswith("seed_"):
                if os.path.isfile(os.path.join(seed_path, "summary.json")):
                    dirs.append(seed_path)
    return dirs


def _load_summary(path: str) -> Optional[dict]:
    p = os.path.join(path, "summary.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def _runs_df(run_dirs: List[str]) -> pd.DataFrame:
    rows = []
    for d in run_dirs:
        s = _load_summary(d)
        if s is not None:
            s["_dir"] = d
            rows.append(s)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{k: v for k, v in x.items() if k != "_dir"} for x in rows])
    df["_dir"] = [x["_dir"] for x in rows]
    return df


def _agg_df(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return pd.DataFrame()
    cols = ["d", "n", "b", "r_mode", "r_value", "r_resolved"]
    for c in cols:
        if c not in runs_df.columns:
            return pd.DataFrame()
    df = runs_df[cols + ["best_val_acc", "final_val_acc", "steps_to_95pct_best_val_acc"]].copy()
    agg = df.groupby(cols, dropna=False).agg(
        best_val_acc_mean=("best_val_acc", "mean"),
        best_val_acc_std=("best_val_acc", "std"),
        final_val_acc_mean=("final_val_acc", "mean"),
        final_val_acc_std=("final_val_acc", "std"),
        steps_to_95_mean=("steps_to_95pct_best_val_acc", "mean"),
        steps_to_95_std=("steps_to_95pct_best_val_acc", "std"),
    ).reset_index()
    return agg


def _save_png(fig, path: str) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sweep_d(agg_df: pd.DataFrame, n0: int, b0: int, out_path: str) -> None:
    sub = agg_df[(agg_df["n"] == n0) & (agg_df["b"] == b0)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for r_mode in ["const", "snr"]:
        rdf = sub[sub["r_mode"] == r_mode].sort_values("d")
        if rdf.empty:
            continue
        label = "R const" if r_mode == "const" else "R = c*sqrt(d)"
        ax.errorbar(rdf["d"], rdf["best_val_acc_mean"], yerr=rdf["best_val_acc_std"].fillna(0), label=label, capsize=3)
    ax.set_xlabel("d")
    ax.set_ylabel("Best val acc")
    ax.legend()
    ax.set_title("Best val acc vs dimension (N,B fixed)")
    fig.tight_layout()
    _save_png(fig, out_path)


def plot_sweep_n(agg_df: pd.DataFrame, d0: int, b0: int, out_path: str) -> None:
    sub = agg_df[(agg_df["d"] == d0) & (agg_df["b"] == b0)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for r_mode in ["const", "snr"]:
        rdf = sub[sub["r_mode"] == r_mode].sort_values("n")
        if rdf.empty:
            continue
        ax.errorbar(rdf["n"], rdf["best_val_acc_mean"], yerr=rdf["best_val_acc_std"].fillna(0), label=f"R {r_mode}", capsize=3)
    ax.set_xlabel("N")
    ax.set_ylabel("Best val acc")
    ax.legend()
    ax.set_title("Best val acc vs N (d,B fixed)")
    fig.tight_layout()
    _save_png(fig, out_path)


def plot_sweep_b(agg_df: pd.DataFrame, d0: int, n0: int, out_path: str) -> None:
    sub = agg_df[(agg_df["d"] == d0) & (agg_df["n"] == n0)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for r_mode in ["const", "snr"]:
        rdf = sub[sub["r_mode"] == r_mode].sort_values("b")
        if rdf.empty:
            continue
        ax.errorbar(rdf["b"], rdf["best_val_acc_mean"], yerr=rdf["best_val_acc_std"].fillna(0), label=f"R {r_mode}", capsize=3)
    ax.set_xlabel("B")
    ax.set_ylabel("Best val acc")
    ax.legend()
    ax.set_title("Best val acc vs B (d,N fixed)")
    fig.tight_layout()
    _save_png(fig, out_path)


def plot_heatmap_d_n(runs_df: pd.DataFrame, r_mode: str, b0: int, out_path: str) -> None:
    sub = runs_df[(runs_df["r_mode"] == r_mode) & (runs_df["b"] == b0)]
    if sub.empty:
        return
    grid = sub.pivot_table(index="d", columns="n", values="best_val_acc", aggfunc="mean")
    if grid.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(grid.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel("N")
    ax.set_ylabel("d")
    ax.set_title(f"Best val acc (d×N), R {r_mode}, B={b0}")
    plt.colorbar(im, ax=ax, label="Best val acc")
    fig.tight_layout()
    _save_png(fig, out_path)


def plot_heatmap_b_n(runs_df: pd.DataFrame, r_mode: str, d0: int, out_path: str) -> None:
    sub = runs_df[(runs_df["r_mode"] == r_mode) & (runs_df["d"] == d0)]
    if sub.empty:
        return
    grid = sub.pivot_table(index="b", columns="n", values="best_val_acc", aggfunc="mean")
    if grid.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(grid.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel("N")
    ax.set_ylabel("B")
    ax.set_title(f"Best val acc (B×N), R {r_mode}, d={d0}")
    plt.colorbar(im, ax=ax, label="Best val acc")
    fig.tight_layout()
    _save_png(fig, out_path)


def plot_learning_curves(runs_df: pd.DataFrame, d0: int, n0: int, b0: int, d_vals: List[int], n_vals: List[int], b_vals: List[int], plots_dir: str) -> None:
    rep = [
        (d0, n0, b0),
        (min(d_vals), n0, b0), (max(d_vals), n0, b0),
        (d0, min(n_vals), b0), (d0, max(n_vals), b0),
        (d0, n0, min(b_vals)), (d0, n0, max(b_vals)),
    ]
    for (d, n, b) in rep:
        for r_mode in ["const", "snr"]:
            sub = runs_df[(runs_df["d"] == d) & (runs_df["n"] == n) & (runs_df["b"] == b) & (runs_df["r_mode"] == r_mode)]
            if sub.empty:
                continue
            dir_path = sub.iloc[0]["_dir"]
            csv_path = os.path.join(dir_path, "metrics.csv")
            if not os.path.isfile(csv_path):
                continue
            df = pd.read_csv(csv_path)
            if "step" not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df["step"], df["val_acc"], label="Val acc")
            ax.plot(df["step"], df["in_context_acc"], label="In-context acc")
            ax.set_xlabel("Step")
            ax.set_ylabel("Accuracy")
            ax.legend()
            ax.set_title(f"Learning curves d={d} n={n} b={b} R={r_mode}")
            fig.tight_layout()
            _save_png(fig, os.path.join(plots_dir, f"curves_d{d}_n{n}_b{b}_{r_mode}.png"))


def generate_all_plots(
    results_root: str,
    exp_name: str,
    d0: int,
    n0: int,
    b0: int,
    d_vals: List[int],
    n_vals: List[int],
    b_vals: List[int],
    plots_dir: Optional[str] = None,
    subdir: Optional[str] = None,
) -> None:
    base = os.path.join(results_root, "rq1", exp_name)
    if subdir:
        base = os.path.join(base, subdir)
    if plots_dir is None:
        plots_dir = os.path.join(base, "_plots")
    os.makedirs(plots_dir, exist_ok=True)
    run_dirs = _scan_run_dirs(results_root, exp_name, subdir)
    runs_df = _runs_df(run_dirs)
    if runs_df.empty:
        return
    agg_df = _agg_df(runs_df)
    if not agg_df.empty:
        plot_sweep_d(agg_df, n0, b0, os.path.join(plots_dir, "sweep_d.png"))
        plot_sweep_n(agg_df, d0, b0, os.path.join(plots_dir, "sweep_n.png"))
        plot_sweep_b(agg_df, d0, n0, os.path.join(plots_dir, "sweep_b.png"))
    for r_mode in ["const", "snr"]:
        plot_heatmap_d_n(runs_df, r_mode, b0, os.path.join(plots_dir, f"heatmap_d_n_{r_mode}.png"))
        plot_heatmap_b_n(runs_df, r_mode, d0, os.path.join(plots_dir, f"heatmap_b_n_{r_mode}.png"))
    plot_learning_curves(runs_df, d0, n0, b0, d_vals, n_vals, b_vals, plots_dir)


# ============================================================================
# RQ3: Full Transformer Evaluation - Plotting Functions
# ============================================================================

def _scan_rq3_commercial_dirs(results_root: str) -> List[str]:
    """Scan for RQ3 commercial LLM experiment directories"""
    base = os.path.join(results_root, "rq3", "commercial")
    if not os.path.isdir(base):
        return []
    dirs = []
    for provider_config in os.listdir(base):
        path = os.path.join(base, provider_config)
        if not os.path.isdir(path):
            continue
        for item in os.listdir(path):
            seed_path = os.path.join(path, item)
            if os.path.isdir(seed_path) and item.startswith("seed_"):
                if os.path.isfile(os.path.join(seed_path, "summary.json")):
                    dirs.append(seed_path)
    return dirs


def _scan_rq3_regression_dirs(results_root: str) -> List[str]:
    """Scan for RQ3 regression experiment directories"""
    base = os.path.join(results_root, "rq3", "regression")
    if not os.path.isdir(base):
        return []
    dirs = []
    for config in os.listdir(base):
        path = os.path.join(base, config)
        if not os.path.isdir(path):
            continue
        for item in os.listdir(path):
            seed_path = os.path.join(path, item)
            if os.path.isdir(seed_path) and item.startswith("seed_"):
                if os.path.isfile(os.path.join(seed_path, "summary.json")):
                    dirs.append(seed_path)
    return dirs


def plot_rq3_provider_comparison_heatmap(
    results_root: str,
    output_file: str = "rq3_provider_heatmap.png"
) -> None:
    """
    Plot accuracy heatmap (d × N) for each commercial LLM provider.
    
    Creates a 3-panel figure with Gemini, Claude, GPT accuracy heatmaps side-by-side.
    """
    run_dirs = _scan_rq3_commercial_dirs(results_root)
    if not run_dirs:
        print("⚠️  No RQ3 commercial results found")
        return
    
    # Load all results
    rows = []
    for d in run_dirs:
        summary_file = os.path.join(d, "summary.json")
        if os.path.isfile(summary_file):
            with open(summary_file) as f:
                summary = json.load(f)
                summary["_dir"] = d
                rows.append(summary)
    
    if not rows:
        print("⚠️  No summaries found")
        return
    
    df = pd.DataFrame(rows)
    
    # Create heatmaps for each provider
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    providers = ['gemini', 'claude', 'gpt']
    
    for idx, provider_name in enumerate(providers):
        provider_df = df[df['provider'] == provider_name]
        
        if provider_df.empty:
            axes[idx].text(0.5, 0.5, f'No {provider_name.upper()} results', 
                          ha='center', va='center', fontsize=12)
            axes[idx].set_title(f"{provider_name.upper()}")
            continue
        
        # Pivot to create heatmap
        pivot = provider_df.pivot_table(
            values='accuracy',
            index='d',
            columns='N',
            aggfunc='mean'
        )
        
        im = axes[idx].imshow(pivot, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        axes[idx].set_title(f"{provider_name.upper()}")
        axes[idx].set_xlabel("Context Length (N)")
        axes[idx].set_ylabel("Dimension (d)")
        axes[idx].set_xticks(range(len(pivot.columns)))
        axes[idx].set_xticklabels(pivot.columns)
        axes[idx].set_yticks(range(len(pivot.index)))
        axes[idx].set_yticklabels(pivot.index)
        
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    axes[idx].text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=axes[-1], label="Accuracy")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved provider heatmap to {output_file}")


def plot_rq3_accuracy_vs_r(
    results_root: str,
    output_file: str = "rq3_accuracy_vs_r.png"
) -> None:
    """
    Plot accuracy vs signal strength (R) for each provider with error bands.
    """
    run_dirs = _scan_rq3_commercial_dirs(results_root)
    if not run_dirs:
        print("⚠️  No RQ3 commercial results found")
        return
    
    rows = []
    for d in run_dirs:
        summary_file = os.path.join(d, "summary.json")
        if os.path.isfile(summary_file):
            with open(summary_file) as f:
                summary = json.load(f)
                rows.append(summary)
    
    df = pd.DataFrame(rows)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    providers = ['gemini', 'claude', 'gpt']
    colors = {'gemini': '#4285F4', 'claude': '#000000', 'gpt': '#10A37F'}
    
    for provider_name in providers:
        provider_df = df[df['provider'] == provider_name].copy()
        
        if provider_df.empty:
            continue
        
        # Group by R and compute mean/std accuracy
        grouped = provider_df.groupby('R')['accuracy'].agg(['mean', 'std', 'count']).reset_index()
        
        if grouped.empty:
            continue
        
        # Plot line with error band
        ax.plot(grouped['R'], grouped['mean'], marker='o', label=provider_name.upper(), 
               color=colors[provider_name], linewidth=2, markersize=8)
        ax.fill_between(grouped['R'], 
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        alpha=0.2, color=colors[provider_name])
    
    ax.set_xlabel("Signal Strength (R)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("RQ3: Commercial LLM Accuracy vs Signal Strength", fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved accuracy vs R plot to {output_file}")


def plot_rq3_regression_baselines(
    results_root: str,
    output_file: str = "rq3_regression_baselines.png"
) -> None:
    """
    Plot MSE comparison between TinyLlama model and gradient descent baselines.
    """
    run_dirs = _scan_rq3_regression_dirs(results_root)
    if not run_dirs:
        print("⚠️  No RQ3 regression results found")
        return
    
    rows = []
    for d in run_dirs:
        summary_file = os.path.join(d, "summary.json")
        if os.path.isfile(summary_file):
            with open(summary_file) as f:
                summary = json.load(f)
                rows.append(summary)
    
    if not rows:
        print("⚠️  No summaries found")
        return
    
    df = pd.DataFrame(rows)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by N (context length) and compute mean MSE
    grouped = df.groupby('N')[['mse_model_mean', 'mse_gd1_mean', 'mse_gdpp_mean']].mean().reset_index()
    
    x = np.arange(len(grouped['N']))
    width = 0.25
    
    ax.bar(x - width, grouped['mse_model_mean'], width, label='TinyLlama', color='#FFA500', alpha=0.8)
    ax.bar(x, grouped['mse_gd1_mean'], width, label='GD-1 (One-step)', color='#4285F4', alpha=0.8)
    ax.bar(x + width, grouped['mse_gdpp_mean'], width, label='GD++ (Preconditioned)', color='#34A853', alpha=0.8)
    
    ax.set_xlabel("Context Length (N)", fontsize=12)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=12)
    ax.set_title("RQ3: Linear Regression - Model vs Gradient Descent Baselines", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped['N'])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved regression baselines plot to {output_file}")


def generate_all_rq3_plots(results_root: str, plots_root: str = None) -> None:
    """Generate all RQ3 visualization plots"""
    if plots_root is None:
        plots_root = os.path.join(results_root, "rq3", "_plots")
    
    os.makedirs(plots_root, exist_ok=True)
    
    print(f"\n📊 Generating RQ3 plots to {plots_root}...")
    
    # Commercial LLM plots
    plot_rq3_provider_comparison_heatmap(
        results_root,
        os.path.join(plots_root, "provider_comparison_heatmap.png")
    )
    
    plot_rq3_accuracy_vs_r(
        results_root,
        os.path.join(plots_root, "accuracy_vs_r.png")
    )
    
    # Regression plots
    plot_rq3_regression_baselines(
        results_root,
        os.path.join(plots_root, "regression_baselines.png")
    )
    
    print(f"✅ All RQ3 plots generated in {plots_root}")

