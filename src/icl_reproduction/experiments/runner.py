import csv
import json
import os
import random
import time

import numpy as np
import torch

from icl_reproduction.model import LinearClassifier
from icl_reproduction.train_and_eval import evaluate, train_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_dir_name(d: int, n: int, b: int, r_mode: str) -> str:
    return f"d{d}_n{n}_b{b}_{r_mode}"


def run_one(
    d: int,
    n: int,
    b: int,
    r_mode: str,
    r_value: float,
    seed: int,
    output_root: str,
    exp_name: str,
    steps: int,
    lr: float,
    eval_every: int,
    log_every: int,
    device: str,
    flip_train: float = 0.0,
    flip_val: float = 0.0,
    subdir: str = "",
) -> str:
    set_seed(seed)
    R = r_value if r_mode == "const" else r_value * (d ** 0.5)
    base = os.path.join(output_root, "rq1", exp_name)
    if subdir:
        base = os.path.join(base, subdir)
    out_dir = os.path.join(base, run_dir_name(d, n, b, r_mode), f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    config = {
        "d": d, "n": n, "b": b, "r_mode": r_mode, "r_value": r_value, "r_resolved": R,
        "flip_train": flip_train, "flip_val": flip_val,
        "steps": steps, "lr": lr, "eval_every": eval_every, "log_every": log_every,
        "device": device, "seed": seed, "exp_name": exp_name,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    model = LinearClassifier(d=d)
    t0 = time.perf_counter()
    metrics = train_model(
        model, d=d, N=n, B=b, R_train=R, R_val=R,
        flip_train=flip_train, flip_val=flip_val,
        steps=steps, lr=lr, device=device, return_metrics=True,
        eval_every=eval_every, log_every=log_every,
        train_seed=seed, eval_seed=seed + 10000,
        early_stop=True,
    )
    elapsed = time.perf_counter() - t0
    val_accs = metrics["val_acc"]
    ic_accs = metrics["ic_acc"]
    best_val_acc = max(val_accs)
    step_best = int(np.argmax(val_accs))
    best_ic = max(ic_accs)
    target_95 = 0.95 * best_val_acc
    steps_to_95 = None
    for s, acc in enumerate(val_accs):
        if acc >= target_95:
            steps_to_95 = s
            break
    summary = {
        "best_val_acc": best_val_acc, "final_val_acc": val_accs[-1],
        "best_in_context_acc": best_ic, "final_in_context_acc": ic_accs[-1],
        "step_best_val_acc": step_best, "steps_to_95pct_best_val_acc": steps_to_95,
        "elapsed_sec": round(elapsed, 2), **config,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "val_loss", "train_acc", "val_acc", "in_context_acc"])
        for step in range(len(metrics["train_loss"])):
            w.writerow([step, metrics["train_loss"][step], metrics["val_loss"][step],
                        metrics["train_acc"][step], metrics["val_acc"][step], metrics["ic_acc"][step]])
    return out_dir
