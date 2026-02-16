import json
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


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
