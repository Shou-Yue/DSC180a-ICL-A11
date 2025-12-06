"""
Training utilities for ICL models

This module contains training loops and utilities for:
- Binary classification with Gaussian mixtures
- Linear regression tasks
- Baseline comparisons (GD, Ridge regression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Optional
import numpy as np


class BinaryClassificationDataset(Dataset):
    """Dataset for binary classification with Gaussian mixtures"""
    
    def __init__(self, 
                 d: int = 20,
                 N: int = 40,
                 num_tasks: int = 100,
                 R: float = 1.0,
                 flip_prob: float = 0.2):
        self.d = d
        self.N = N
        self.num_tasks = num_tasks
        self.R = R
        self.flip_prob = flip_prob
        
        self.context_x = []
        self.context_y = []
        self.query_x = []
        self.query_y = []
        
        self._generate_tasks()
    
    def _generate_tasks(self):
        """Generate all tasks"""
        for _ in range(self.num_tasks):
            mu1 = torch.randn(self.d)
            mu2 = torch.randn(self.d)
            
            mu1 = mu1 * torch.sqrt(torch.tensor(self.R))
            mu2 = mu2 * torch.sqrt(torch.tensor(self.R))
            
            n_class1 = self.N // 2
            n_class2 = self.N - n_class1
            
            x1 = torch.randn(n_class1, self.d) + mu1
            y1 = torch.zeros(n_class1)
            
            x2 = torch.randn(n_class2, self.d) + mu2
            y2 = torch.ones(n_class2)
            
            x = torch.cat([x1, x2], dim=0)
            y = torch.cat([y1, y2], dim=0)
            perm = torch.randperm(self.N)
            x = x[perm]
            y = y[perm]
            
            flip_mask = torch.rand(self.N) < self.flip_prob
            y[flip_mask] = 1 - y[flip_mask]
            
            if torch.rand(1) < 0.5:
                x_q = torch.randn(self.d) + mu1
                y_q = torch.tensor(0.0)
            else:
                x_q = torch.randn(self.d) + mu2
                y_q = torch.tensor(1.0)
            
            if torch.rand(1) < self.flip_prob:
                y_q = 1 - y_q
            
            self.context_x.append(x)
            self.context_y.append(y)
            self.query_x.append(x_q)
            self.query_y.append(y_q)
    
    def __len__(self):
        return self.num_tasks
    
    def __getitem__(self, idx):
        return {
            'context_x': self.context_x[idx],
            'context_y': self.context_y[idx],
            'query_x': self.query_x[idx],
            'query_y': self.query_y[idx]
        }


class GaussianMixtureDataset(Dataset):
    """Dataset for Gaussian mixture classification tasks (full batch)"""
    
    def __init__(self, d: int, N: int, B: int, R: float, 
                 is_validation: bool = False, label_flip_p: float = 0.0):
        self.d = d
        self.N = N 
        self.B = B
        self.R = R
        self.is_validation = is_validation
        self.label_flip_p = label_flip_p
        
        if is_validation:
            torch.manual_seed(42)
        
        self.context_x, self.context_y, self.target_x, self.target_y = self._generate_data()
    
    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate all data at once"""
        mus = torch.randn(self.B, self.d)
        mus = mus / torch.norm(mus, dim=1, keepdim=True) * self.R
        
        y_all = (torch.rand(self.B, self.N + 1) > 0.5).float()
        y_signal = 2 * y_all - 1
        
        z = torch.randn(self.B, self.N + 1, self.d)
        
        x = y_signal[..., None] * mus[:, None, :] + z

        if self.label_flip_p:
            flip_mask = (torch.rand(self.B, self.N + 1) < self.label_flip_p)
            y_all = torch.where(flip_mask, 1 - y_all, y_all)
        
        context_x = x[:, :self.N, :]
        target_x = x[:, -1, :]
        context_y = y_all[:, :self.N]
        target_y = y_all[:, -1]
        
        return context_x, context_y, target_x, target_y
    
    def __len__(self) -> int:
        return 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert idx == 0, "Only one batch supported"
        return self.context_x, self.context_y, self.target_x, self.target_y


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        context_x = batch['context_x'].to(device)
        context_y = batch['context_y'].to(device)
        query_x = batch['query_x'].to(device)
        query_y = batch['query_y'].to(device)
        
        logits = model(context_x, context_y, query_x)
        loss = F.binary_cross_entropy_with_logits(logits, query_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == query_y).sum().item()
        total += len(query_y)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model on validation set"""
    model.eval()
    correct = 0
    total = 0
    
    for batch in dataloader:
        context_x = batch['context_x'].to(device)
        context_y = batch['context_y'].to(device)
        query_x = batch['query_x'].to(device)
        query_y = batch['query_y'].to(device)
        
        logits = model(context_x, context_y, query_x)
        preds = (logits > 0).float()
        correct += (preds == query_y).sum().item()
        total += len(query_y)
    
    accuracy = correct / total
    return accuracy


def train_linear_transformer(model, train_dataset, num_epochs=50, 
                            learning_rate=0.01, device="cpu", verbose=False):
    """Train LinearTransformer on dataset"""
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)
    
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            context_x, context_y, target_x, target_y = [t.to(device) for t in batch]
            
            pred = model(context_x, context_y, target_x)
            loss = F.binary_cross_entropy_with_logits(pred, target_y.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        losses.append(epoch_loss)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss = {epoch_loss:.6f}")
    
    return losses


def train_model(model, d: int, N: int, B: int, R_train: float, R_val: float,
                flip_train: float = 0.0, flip_val: float = 0.0, steps: int = 300,
                lr: float = 1e-2, device: str = "cpu", return_metrics=True):
    """Train linear classifier"""
    model = model.to(device)
    optim_obj = torch.optim.SGD(model.parameters(), lr=lr)
    
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
        loss = F.binary_cross_entropy_with_logits(logits, y_tgt.float())

        optim_obj.zero_grad()
        loss.backward()
        optim_obj.step()

        train_acc = ((logits > 0).float() == y_tgt).float().mean().item()

        val_loss, val_acc, ctx_acc = evaluate_classifier(
            model, d, N, B, R_val, flip_val=flip_val, device=device
        )
        
        metrics["train_loss"].append(loss.item())
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)
        metrics["ic_acc"].append(ctx_acc)

        if step % 50 == 0:
            print(f"Step {step:03d} | Train Loss: {loss.item():.2f} | "
                  f"Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f}")
    
    return metrics if return_metrics else model


def data_gen(d: int, N: int, B: int, R: float, flip_prob: float = 0.0,
            device: str = "cpu", seed: Optional[int] = None):
    """Generate synthetic binary classification tasks"""
    
    if seed is not None:
        g_base = torch.Generator(device="cpu").manual_seed(seed)
        g_flip = torch.Generator(device="cpu").manual_seed(seed + 1)
    else:
        g_base = None
        g_flip = None

    mu = torch.randn(B, d, generator=g_base)
    mu = mu / mu.norm(dim=1, keepdim=True)  
    mu = R * mu

    labels = (torch.rand(B, N + 1, generator=g_base) > 0.5).float()
    y_signal = 2 * labels - 1

    noise = torch.randn(B, N + 1, d, generator=g_base)

    x = (y_signal.unsqueeze(-1) * mu.unsqueeze(1) + noise)

    if flip_prob > 0.0:
        flip_mask = torch.rand(B, N + 1, generator=g_flip) < flip_prob
        labels = torch.where(flip_mask, 1.0 - labels, labels)

    x_context = (x[:, :N, :]).to(device)           
    x_target = (x[:, -1, :]).to(device)           
    y_context = (labels[:, :N]).to(device)         
    y_target = (labels[:, -1]).to(device)        

    return (x_context, y_context, x_target, y_target)


def evaluate_classifier(model, d, N, B_val, R_val, flip_val=0.0, device="cpu"):
    """Evaluate linear classifier"""
    model.eval()
    with torch.no_grad():
        x_ctx, y_ctx, x_tgt, y_tgt = data_gen(d, N, B_val, R_val, flip_prob=flip_val, device=device)

        logits = model(x_ctx, y_ctx, x_tgt)
        val_loss = F.binary_cross_entropy_with_logits(logits, y_tgt.float())

        preds = (logits > 0).float()
        val_acc = (preds == y_tgt).float().mean().item()

        ctx_preds = model.compute_in_context_preds(x_ctx, y_ctx)
        ctx_acc = (ctx_preds == y_ctx).float().mean().item()

        return val_loss.item(), val_acc, ctx_acc


# Baseline methods
def gd_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, lr: float = 0.01) -> np.ndarray:
    """Single gradient descent step"""
    residual = X @ w - y
    grad = X.T @ residual / len(y)
    w_new = w - lr * grad
    return w_new


def gd_solution(X: np.ndarray, y: np.ndarray, k_steps: int = 10, 
                lr: float = 0.01, init_w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Run k steps of gradient descent"""
    d = X.shape[1]
    w = np.zeros(d) if init_w is None else init_w.copy()
    weights_history = [w.copy()]
    
    for _ in range(k_steps):
        w = gd_step(X, y, w, lr)
        weights_history.append(w.copy())
    
    return w, weights_history


def ridge_solution(X: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Closed-form ridge regression solution"""
    N, d = X.shape
    A = X.T @ X + lam * np.eye(d)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    return w
