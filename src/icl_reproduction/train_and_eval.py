import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
from data import data_gen
from model import LinearClassifier
import os


#gets in context predictions + accuracy on a validation batch
def evaluate(model, d, N, B_val, R_val, flip_val=0.0, device="cpu"):
    model.eval()
    with torch.no_grad():

        #sample batch
        x_ctx, y_ctx, x_tgt, y_tgt = data_gen(d, N, B_val, R_val, flip_prob = flip_val, device=device)

        #forward pass
        logits = model(x_ctx, y_ctx, x_tgt)
        val_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tgt.float())

        #compute target acc
        preds = (logits > 0).float()
        val_acc = (preds == y_tgt).float().mean().item()

        #compute icl acc
        ctx_preds = model.compute_in_context_preds(x_ctx, y_ctx)
        ctx_acc = (ctx_preds == y_ctx).float().mean().item()

        return val_loss.item(), val_acc, ctx_acc
    
#full training loop
def train_model(
    model,
    d: int,
    N: int,
    B: int,
    R_train: float,
    R_val: float,
    flip_train: float = 0.0,
    flip_val: float = 0.0,
    steps: int = 300,
    lr: float = 1e-2,
    device: str = "cpu",
    return_metrics=True
):
    model = model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    
    metrics = {
        "train_acc": [],
        "val_acc": [],
        "ic_acc": [],
        "train_loss": [],
        "val_loss": []
    }

    for step in range(steps):
        x_ctx, y_ctx, x_tgt, y_tgt = data_gen(
            d, N, B, R_train, flip_prob=flip_train, device=device
        )

        logits = model(x_ctx, y_ctx, x_tgt)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_tgt.float()
        )

        # update
        optim.zero_grad()
        loss.backward()
        optim.step()

        #get train acc
        train_acc = ((logits > 0).float() == y_tgt).float().mean().item()

        #get val acc
        val_loss, val_acc, ctx_acc = evaluate(
            model, d, N, B, R_val, flip_val=flip_val, device=device
        )
        
        metrics["train_loss"].append(loss.item())
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)
        metrics["ic_acc"].append(ctx_acc)

        #print updates every 10 steps
        if step % 10 == 0:
            print(
                f"Step {step:03d} | "
                f"Train Loss: {loss.item():.2f} | Train Acc: {train_acc:.2f} | "
                f"Val Loss: {val_loss:.2f} | Val Acc: {val_acc:.2f} | "
                f"In-Context Acc: {ctx_acc:.2f}"
            )
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
    
    #save plot
    fname = f"plot_d{d}_N{N}_B{B}_R{R:.2f}_noise{flip_val:.2f}.png"
    save_dir = "../../results"
    save_path = os.path.join(save_dir, fname)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()



    return metrics


#run all experiments
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