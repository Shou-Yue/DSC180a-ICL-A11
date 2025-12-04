import os
from typing import Iterable, List, Dict, Any
import csv

import matplotlib.pyplot as plt

from .eval import eval_regr_suite


def run_ood_scaling(
    alphas: Iterable[float],
    d: int,
    N: int,
    eta: float = 1.0,
    n_tasks: int = 10,
    results_dir: str = "results",
) -> List[Dict[str, Any]]:
    """
    Reproduce the OOD experiment from the notebook, but:
    - log metrics to `results/logs/ood_scaling.csv`
    - save a plot to `results/figures/ood_scaling_r2.png`
    """
    alphas = list(alphas)

    logs_dir = os.path.join(results_dir, "logs")
    figs_dir = os.path.join(results_dir, "figures")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for a in alphas:
        r = eval_regr_suite(n_tasks=n_tasks, d=d, N=N, alpha=a, eta=eta)
        r_with_alpha = {"alpha": a, **r}
        results.append(r_with_alpha)
        print(
            f"alpha={a:.2f} | "
            f"LLM MSE={r['mse_llm']:.4f}, GD-1 MSE={r['mse_gd1']:.4f}, "
            f"LLM R2={r['r2_llm']:.4f}, GD-1 R2={r['r2_gd1']:.4f}"
        )

    # Save CSV log
    csv_path = os.path.join(logs_dir, "ood_scaling.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    # Plot R2 vs alpha
    if results:
        plt.figure()
        plt.plot(
            alphas,
            [r["r2_llm"] for r in results],
            marker="o",
            linestyle="--",
            label="LLM (R2)",
        )
        plt.plot(
            alphas,
            [r["r2_gd1"] for r in results],
            marker="o",
            label="GD-1 (R2)",
        )
        plt.xlabel("alpha (input range)")
        plt.ylabel("R^2")
        plt.title(f"OOD scaling (d={d}, N={N})")
        plt.legend()
        fig_path = os.path.join(figs_dir, "ood_scaling_r2.png")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()

    return results
