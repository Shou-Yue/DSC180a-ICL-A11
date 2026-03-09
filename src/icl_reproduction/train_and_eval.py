import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from data import data_gen
from model import LinearClassifier, MiniTransformer


def evaluate(
    model, d, N, B_val, R_val, flip_val=0.0,
    flip_context_val=None, flip_query_val=None,
    device="cpu", seed=None,
):
    """flip_context_val/flip_query_val override flip_val when set (for RQ2 context-only noise)."""
    model.eval()
    use_separate = flip_context_val is not None or flip_query_val is not None
    kwargs = dict(d=d, N=N, B=B_val, R=R_val, device=device, seed=seed)
    if use_separate:
        kwargs["flip_context_prob"] = flip_context_val if flip_context_val is not None else 0.0
        kwargs["flip_query_prob"] = flip_query_val if flip_query_val is not None else 0.0
    else:
        kwargs["flip_prob"] = flip_val
    with torch.no_grad():
        x_ctx, y_ctx, x_tgt, y_tgt = data_gen(**kwargs)
        logits = model(x_ctx, y_ctx, x_tgt)
        val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_tgt.float()
        )
        preds = (logits > 0).float()
        val_acc = (preds == y_tgt).float().mean().item()
        ctx_preds = model.compute_in_context_preds(x_ctx, y_ctx)
        ctx_acc = (ctx_preds == y_ctx).float().mean().item()
        return val_loss.item(), val_acc, ctx_acc


def train_model(
    model,
    d: int,
    N: int,
    B: int,
    R_train: float,
    R_val: float,
    flip_train: float = 0.0,
    flip_val: float = 0.0,
    flip_context_train: float | None = None,
    flip_query_train: float | None = None,
    flip_context_val: float | None = None,
    flip_query_val: float | None = None,
    steps: int = 300,
    lr: float = 1e-2,
    device: str = "cpu",
    return_metrics: bool = True,
    eval_every: int = 1,
    log_every: int = 10,
    train_seed=None,
    eval_seed=None,
    early_stop: bool = False,
    early_stop_plateau_window: int = 50,
    early_stop_plateau_min_steps: int = 500,
    early_stop_plateau_tol: float = 0.01,
):
    """For RQ2 context-only noise: set flip_context_train=eps, flip_query_train=0, flip_context_val=0, flip_query_val=0.
    Early stop: max steps 1000; after early_stop_plateau_min_steps, stop if train/val/ic acc change < tol over last window steps."""
    model = model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    metrics = {
        "train_acc": [],
        "val_acc": [],
        "ic_acc": [],
        "train_loss": [],
        "val_loss": [],
    }
    use_separate_train = flip_context_train is not None or flip_query_train is not None
    use_separate_val = flip_context_val is not None or flip_query_val is not None
    last_val_loss, last_val_acc, last_ctx_acc = None, None, None
    for step in range(steps):
        train_kw = dict(d=d, N=N, B=B, R=R_train, device=device, seed=train_seed + step if train_seed is not None else None)
        if use_separate_train:
            train_kw["flip_context_prob"] = flip_context_train if flip_context_train is not None else 0.0
            train_kw["flip_query_prob"] = flip_query_train if flip_query_train is not None else 0.0
        else:
            train_kw["flip_prob"] = flip_train
        x_ctx, y_ctx, x_tgt, y_tgt = data_gen(**train_kw)
        logits = model(x_ctx, y_ctx, x_tgt)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_tgt.float()
        )
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_acc = ((logits > 0).float() == y_tgt).float().mean().item()
        eval_kw = dict(model=model, d=d, N=N, B_val=B, R_val=R_val, device=device,
                       seed=eval_seed + step if eval_seed is not None else None)
        if use_separate_val:
            eval_kw["flip_context_val"] = flip_context_val if flip_context_val is not None else 0.0
            eval_kw["flip_query_val"] = flip_query_val if flip_query_val is not None else 0.0
        else:
            eval_kw["flip_val"] = flip_val
        if step % eval_every == 0:
            last_val_loss, last_val_acc, last_ctx_acc = evaluate(**eval_kw)
        if last_val_loss is None:
            eval_kw["seed"] = eval_seed if eval_seed is not None else None
            last_val_loss, last_val_acc, last_ctx_acc = evaluate(**eval_kw)
        metrics["train_loss"].append(loss.item())
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(last_val_loss)
        metrics["val_acc"].append(last_val_acc)
        metrics["ic_acc"].append(last_ctx_acc)
        if step % log_every == 0:
            print(
                f"Step {step:03d} | "
                f"Train Loss: {loss.item():.2f} | Train Acc: {train_acc:.2f} | "
                f"Val Loss: {last_val_loss:.2f} | Val Acc: {last_val_acc:.2f} | "
                f"In-Context Acc: {last_ctx_acc:.2f}"
            )
        if early_stop and step >= 50 and train_acc >= 0.999 and last_val_acc >= 0.999:
            break
        # Plateau early stop: after min_steps (e.g. 500), stop if no significant change in accuracies over last window
        if early_stop and step >= early_stop_plateau_min_steps:
            k = early_stop_plateau_window
            if len(metrics["train_acc"]) >= k:
                ta = metrics["train_acc"][-k:]
                va = metrics["val_acc"][-k:]
                ia = metrics["ic_acc"][-k:]
                if (max(ta) - min(ta) <= early_stop_plateau_tol and
                    max(va) - min(va) <= early_stop_plateau_tol and
                    max(ia) - min(ia) <= early_stop_plateau_tol):
                    break
    if return_metrics:
        return metrics
    return model


#run experiments and plot 

#training loop will run with all values in index 0, then 1, etc.
d_vals       = [500, 1000, 1500]
n_vals       = [20,   20,   20]
b_vals       = [1000, 1000, 1000]             
r_vals = [d**0.3 for d in d_vals]
#Comment below line if r will stay the same relative to d for all experiments
#r_vals = [1500**0.3, 20**0.1, 50**0.5]
flips_train = [0.3, 0.3, 0.3]
flips_val = [0.3, 0.3, 0.3]

steps = 300
device = "cpu"

#will run a single experiment with predetermined vars
def run_experiment(
    d, N, B, R_train, R_val, flip_train, flip_val, steps=300, device="cpu"
):
    print(f"Running experiment with following hyperparams: d={d}, n={N}, b={B}, R={R_train}, flip={flip_train}")

    # build model
    model = LinearClassifier(d=d)

    # run training 
    metrics = train_model(
        model,
        d=d,
        N=N,
        B=B,
        R_train=R_train,
        R_val=R_val,
        flip_train=flip_train,
        flip_val=flip_val,
        steps=steps,
        device=device,
        return_metrics=True,   
    )

    #plot train, val, IC accuracy vs training step
    steps_axis = range(len(metrics["train_acc"]))

    plt.figure(figsize=(7,5))
    plt.plot(steps_axis, metrics["train_acc"], label="Train Acc")
    plt.plot(steps_axis, metrics["val_acc"], label="Val Acc")
    plt.plot(steps_axis, metrics["ic_acc"], label="In-Context Acc")
    plt.legend()
    plt.xlabel("Training Step")
    plt.ylabel("Accuracy")
    plt.title(f"Train, Validation, and In-Context Accuracies by Training Step for d={d}, N={N}, B={B}, R={R}, noise={flip_val}")
    
    fname = f"plot_d{d}_N{N}_B{B}_R{R:.2f}_noise{flip_val:.2f}.png"
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()



    return metrics


if __name__ == "__main__":
    for i in range(len(d_vals)):
        d = d_vals[i]
        N = n_vals[i]
        B = b_vals[i]
        R = r_vals[i]
        flip_train_val = flips_train[i]
        flip_val_val = flips_val[i]
        run_experiment(
            d=d,
            N=N,
            B=B,
            R_train=R,
            R_val=R,
            flip_train=flip_train_val,
            flip_val=flip_val_val,
            steps=steps,
            device=device,
        )