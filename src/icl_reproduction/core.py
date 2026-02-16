import torch
from data import data_gen
from model import LinearClassifier


def evaluate(model, d, N, B_val, R_val, flip_val=0.0, device="cpu", seed=None):
    model.eval()
    with torch.no_grad():
        x_ctx, y_ctx, x_tgt, y_tgt = data_gen(
            d, N, B_val, R_val, flip_prob=flip_val, device=device, seed=seed
        )
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
    steps: int = 300,
    lr: float = 1e-2,
    device: str = "cpu",
    return_metrics: bool = True,
    eval_every: int = 1,
    log_every: int = 10,
    train_seed: int = None,
    eval_seed: int = None,
):
    model = model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    metrics = {
        "train_acc": [],
        "val_acc": [],
        "ic_acc": [],
        "train_loss": [],
        "val_loss": [],
    }
    last_val_loss, last_val_acc, last_ctx_acc = None, None, None
    for step in range(steps):
        x_ctx, y_ctx, x_tgt, y_tgt = data_gen(
            d, N, B, R_train,
            flip_prob=flip_train,
            device=device,
            seed=train_seed + step if train_seed is not None else None,
        )
        logits = model(x_ctx, y_ctx, x_tgt)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_tgt.float()
        )
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_acc = ((logits > 0).float() == y_tgt).float().mean().item()
        if step % eval_every == 0:
            last_val_loss, last_val_acc, last_ctx_acc = evaluate(
                model, d, N, B, R_val,
                flip_val=flip_val,
                device=device,
                seed=eval_seed + step if eval_seed is not None else None,
            )
        if last_val_loss is None:
            last_val_loss, last_val_acc, last_ctx_acc = evaluate(
                model, d, N, B, R_val,
                flip_val=flip_val,
                device=device,
                seed=eval_seed if eval_seed is not None else None,
            )
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
    if return_metrics:
        return metrics
    return model
