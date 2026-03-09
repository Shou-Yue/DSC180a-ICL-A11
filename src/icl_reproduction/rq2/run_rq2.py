"""
RQ2: Benign overfitting — when does the model memorize noisy in-context labels
while still achieving high test accuracy? Context-only label noise.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_script_dir)
sys.path.insert(0, _src)

from icl_reproduction.experiments.runner import run_one_rq2
from icl_reproduction.experiments.plots import _scan_run_dirs, _load_summary

# RQ2 constants (Section 2.3.4)
EPSILON_VALS = [0, 0.05, 0.1, 0.2, 0.3, 0.4]
D_NOISE, N_NOISE, B_NOISE = 500, 20, 1000
R_CONST = 6.45
R_SNR_C = 0.3
EPSILON_FIXED = 0.2
D_SWEEP = [100, 200, 500, 1000]
N_SWEEP = [5, 10, 20, 40, 80]
# Signal-to-noise: low / medium / high
R_CONST_VALS = [4.0, 6.45, 10.0]
R_SNR_C_VALS = [0.2, 0.3, 0.5]

MAX_STEPS = 1000
LR = 1e-2
EVAL_EVERY = 10
LOG_EVERY = 10
EXP_NAME = "rq2_benign"


def _runs_noise_sweep(seeds):
    for r_mode, r_val in [("const", R_CONST), ("snr", R_SNR_C)]:
        for eps in EPSILON_VALS:
            for seed in seeds:
                yield (D_NOISE, N_NOISE, B_NOISE, r_mode, r_val, eps, seed)


def _runs_snr(seeds):
    for r_mode in ["const", "snr"]:
        vals = R_CONST_VALS if r_mode == "const" else R_SNR_C_VALS
        for r_val in vals:
            for seed in seeds:
                yield (D_NOISE, N_NOISE, B_NOISE, r_mode, r_val, EPSILON_FIXED, seed)


def _runs_dim(seeds):
    for d in D_SWEEP:
        for r_mode, r_val in [("const", R_CONST), ("snr", R_SNR_C)]:
            for seed in seeds:
                yield (d, N_NOISE, B_NOISE, r_mode, r_val, EPSILON_FIXED, seed)


def _runs_context(seeds):
    for n in N_SWEEP:
        for r_mode, r_val in [("const", R_CONST), ("snr", R_SNR_C)]:
            for seed in seeds:
                yield (D_NOISE, n, B_NOISE, r_mode, r_val, EPSILON_FIXED, seed)


def _runs_df(run_dirs):
    rows = []
    for path in run_dirs:
        s = _load_summary(path)
        if s is not None:
            s["_dir"] = path
            rows.append(s)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{k: v for k, v in x.items() if k != "_dir"} for x in rows])
    df["_dir"] = [x["_dir"] for x in rows]
    return df


def _agg_rq2(df, group_cols):
    if df.empty or "best_val_acc" not in df.columns:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in df.columns]
    return df.groupby(cols, dropna=False).agg(
        best_val_acc_mean=("best_val_acc", "mean"),
        best_val_acc_std=("best_val_acc", "std"),
        best_in_context_acc_mean=("best_in_context_acc", "mean"),
        best_in_context_acc_std=("best_in_context_acc", "std"),
    ).reset_index()


def _save_png(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_sweep_epsilon(agg, plots_dir):
    if agg.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for r_mode in ["const", "snr"]:
        sub = agg[agg["r_mode"] == r_mode].sort_values("epsilon")
        if sub.empty:
            continue
        label = "R const" if r_mode == "const" else "R = c*sqrt(d)"
        ax.errorbar(
            sub["epsilon"],
            sub["best_val_acc_mean"],
            yerr=sub["best_val_acc_std"].fillna(0),
            label=label,
            capsize=3,
        )
    ax.set_xlabel("ε (context label noise)")
    ax.set_ylabel("Best val acc")
    ax.legend()
    ax.set_title("RQ2: Val acc vs context noise (d=500, N=20, B=1000)")
    fig.tight_layout()
    _save_png(fig, os.path.join(plots_dir, "sweep_epsilon.png"))


def _plot_sweep_d(agg, plots_dir):
    if agg.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for r_mode in ["const", "snr"]:
        sub = agg[agg["r_mode"] == r_mode].sort_values("d")
        if sub.empty:
            continue
        label = "R const" if r_mode == "const" else "R = c*sqrt(d)"
        ax.errorbar(
            sub["d"],
            sub["best_val_acc_mean"],
            yerr=sub["best_val_acc_std"].fillna(0),
            label=label,
            capsize=3,
        )
    ax.set_xlabel("d")
    ax.set_ylabel("Best val acc")
    ax.legend()
    ax.set_title("RQ2: Val acc vs dimension (N=20, B=1000, ε=0.2)")
    fig.tight_layout()
    _save_png(fig, os.path.join(plots_dir, "sweep_d.png"))


def _plot_sweep_n(agg, plots_dir):
    if agg.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for r_mode in ["const", "snr"]:
        sub = agg[agg["r_mode"] == r_mode].sort_values("n")
        if sub.empty:
            continue
        label = "R const" if r_mode == "const" else "R = c*sqrt(d)"
        ax.errorbar(
            sub["n"],
            sub["best_val_acc_mean"],
            yerr=sub["best_val_acc_std"].fillna(0),
            label=label,
            capsize=3,
        )
    ax.set_xlabel("N")
    ax.set_ylabel("Best val acc")
    ax.legend()
    ax.set_title("RQ2: Val acc vs context size (d=500, B=1000, ε=0.2)")
    fig.tight_layout()
    _save_png(fig, os.path.join(plots_dir, "sweep_n.png"))


def _plot_sweep_r(agg, plots_dir):
    if agg.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for r_mode in ["const", "snr"]:
        sub = agg[agg["r_mode"] == r_mode].sort_values("r_resolved")
        if sub.empty:
            continue
        label = "R const" if r_mode == "const" else "R = c*sqrt(d)"
        ax.errorbar(
            sub["r_resolved"],
            sub["best_val_acc_mean"],
            yerr=sub["best_val_acc_std"].fillna(0),
            label=label,
            capsize=3,
        )
    ax.set_xlabel("R (resolved)")
    ax.set_ylabel("Best val acc")
    ax.legend()
    ax.set_title("RQ2: Val acc vs signal strength (d=500, N=20, B=1000, ε=0.2)")
    fig.tight_layout()
    _save_png(fig, os.path.join(plots_dir, "sweep_r.png"))


def _plot_learning_curves_rq2(runs_df, plots_dir, d0, n0, b0):
    """RQ1-style learning curves: curves_d{d}_n{n}_b{b}_{r_mode}_eps{eps}.png"""
    rep_eps = [0.0, 0.2, 0.4]
    for (d, n, b) in [(d0, n0, b0), (min(D_SWEEP), n0, b0), (max(D_SWEEP), n0, b0),
                      (d0, min(N_SWEEP), b0), (d0, max(N_SWEEP), b0)]:
        for r_mode in ["const", "snr"]:
            for eps in rep_eps:
                sub = runs_df[(runs_df["d"] == d) & (runs_df["n"] == n) & (runs_df["b"] == b) &
                             (runs_df["r_mode"] == r_mode) & np.isclose(runs_df["epsilon"], eps)]
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
                if "train_acc" in df.columns:
                    ax.plot(df["step"], df["train_acc"], label="Train acc")
                ax.plot(df["step"], df["val_acc"], label="Val acc")
                ax.plot(df["step"], df["in_context_acc"], label="In-context acc")
                ax.set_xlabel("Step")
                ax.set_ylabel("Accuracy")
                ax.legend()
                ax.set_title(f"RQ2 curves d={d} n={n} b={b} R={r_mode} ε={eps}")
                fig.tight_layout()
                eps_str = f"{eps:.2f}".replace(".", "_")  # e.g. 0_20
                _save_png(fig, os.path.join(plots_dir, f"curves_d{d}_n{n}_b{b}_{r_mode}_eps{eps_str}.png"))


def generate_rq2_plots(results_root: str, exp_name: str) -> None:
    """Single _plots folder under results/rq2/<exp_name>/ (RQ1-style)."""
    plots_dir = os.path.join(results_root, "rq2", exp_name, "_plots")
    os.makedirs(plots_dir, exist_ok=True)
    for subdir in ["noise_sweep", "snr", "dim", "context"]:
        run_dirs = _scan_run_dirs(results_root, exp_name, subdir=subdir, rq_subdir="rq2")
        if not run_dirs:
            continue
        df = _runs_df(run_dirs)
        if df.empty:
            continue
        if subdir == "noise_sweep":
            agg = _agg_rq2(df, ["d", "n", "b", "r_mode", "r_value", "epsilon"])
            _plot_sweep_epsilon(agg, plots_dir)
        elif subdir == "snr":
            agg = _agg_rq2(df, ["d", "n", "b", "r_mode", "r_value", "r_resolved", "epsilon"])
            _plot_sweep_r(agg, plots_dir)
        elif subdir == "dim":
            agg = _agg_rq2(df, ["d", "n", "b", "r_mode", "r_value", "epsilon"])
            _plot_sweep_d(agg, plots_dir)
        elif subdir == "context":
            agg = _agg_rq2(df, ["d", "n", "b", "r_mode", "r_value", "epsilon"])
            _plot_sweep_n(agg, plots_dir)
    # Learning curves (RQ1-style naming): use all run_dirs from all subdirs
    all_dirs = []
    for subdir in ["noise_sweep", "snr", "dim", "context"]:
        all_dirs.extend(_scan_run_dirs(results_root, exp_name, subdir=subdir, rq_subdir="rq2"))
    if all_dirs:
        runs_df = _runs_df(all_dirs)
        if not runs_df.empty:
            _plot_learning_curves_rq2(runs_df, plots_dir, D_NOISE, N_NOISE, B_NOISE)


def main():
    p = argparse.ArgumentParser(description="RQ2: Benign overfitting (context-only noise)")
    p.add_argument("--exp_name", default=EXP_NAME)
    p.add_argument("--output_root", default="results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--steps", type=int, default=MAX_STEPS)
    p.add_argument("--only_plots", action="store_true")
    args = p.parse_args()
    steps = min(args.steps, MAX_STEPS)
    seeds = [int(x.strip()) for x in args.seeds.split(",")]
    os.makedirs(os.path.join(args.output_root, "rq2"), exist_ok=True)

    if not args.only_plots:
        for d, n, b, r_mode, r_val, eps, seed in _runs_noise_sweep(seeds):
            run_one_rq2(d, n, b, r_mode, r_val, eps, seed, args.output_root, args.exp_name, "noise_sweep", steps, LR, EVAL_EVERY, LOG_EVERY, args.device)
        for d, n, b, r_mode, r_val, eps, seed in _runs_snr(seeds):
            run_one_rq2(d, n, b, r_mode, r_val, eps, seed, args.output_root, args.exp_name, "snr", steps, LR, EVAL_EVERY, LOG_EVERY, args.device)
        for d, n, b, r_mode, r_val, eps, seed in _runs_dim(seeds):
            run_one_rq2(d, n, b, r_mode, r_val, eps, seed, args.output_root, args.exp_name, "dim", steps, LR, EVAL_EVERY, LOG_EVERY, args.device)
        for d, n, b, r_mode, r_val, eps, seed in _runs_context(seeds):
            run_one_rq2(d, n, b, r_mode, r_val, eps, seed, args.output_root, args.exp_name, "context", steps, LR, EVAL_EVERY, LOG_EVERY, args.device)

    generate_rq2_plots(args.output_root, args.exp_name)

    base = os.path.join(args.output_root, "rq2", args.exp_name)
    print(f"Results: {os.path.abspath(base)}")
    print(f"Plots:   {os.path.abspath(base)}/_plots/")
    if args.only_plots:
        print("(Plots-only run; use without --only_plots to train.)")


if __name__ == "__main__":
    main()
