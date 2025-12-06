"""
Models for In-Context Learning

This module contains model implementations for ICL tasks:
- LinearTransformer: One-layer linear transformer for binary classification
- SingleLayerTransformer: Attention-based transformer with single layer
- LinearClassifier: Simple linear classifier for in-context learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LinearTransformer(nn.Module):
    """One-layer linear transformer model for binary classification.
    
    Implements: logit = (W @ context_mean) · target_x
    where context_mean = (1/N) Σ y_i * x_i
    """
    
    def __init__(self, d: int):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(d, d))

    def _predict_single(self, context_x: torch.Tensor, context_y: torch.Tensor, 
                        target_x: torch.Tensor) -> torch.Tensor:
        """Compute prediction: logit = (W @ context_mean) · target_x"""
        N = context_x.shape[1]
        
        # Convert labels 0/1 to -1/+1 for computation
        context_y_signal = 2 * context_y - 1  # Shape: (B, N)
        
        # Compute context mean: (1/N) Σ y_i * x_i
        context_term = (1/N) * torch.sum(context_y_signal[..., None] * context_x, dim=1)
        
        # Transform: W @ context_mean
        transformed = context_term @ self.W  # Shape: (B, d)
        
        # Take inner product with target
        logits = (transformed * target_x).sum(dim=1)  # Shape: (B,)
        return logits
        
    def forward(self, context_x: torch.Tensor, context_y: torch.Tensor, 
                target_x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass"""
        return self._predict_single(context_x, context_y, target_x)
    
    def compute_in_context_preds(self, context_x: torch.Tensor, 
                                context_y: torch.Tensor) -> torch.Tensor:
        """Compute predictions for each position in the context.
        
        Returns:
            predictions: Shape (batch_size, N) - predicted labels for each position
        """
        B, N, d = context_x.shape

        context_x_norm = context_x / torch.norm(context_x, dim=2, keepdim=True)
        context_y_signal = 2 * context_y - 1
        
        hat_mu = (1/N) * torch.sum(context_y_signal[..., None] * context_x_norm, dim=1)
        transformed = hat_mu @ self.W
        logits = transformed[:, None, :] @ context_x_norm.transpose(-1, -2)
        predictions = (logits[:, 0, :] > 0).float()
        return predictions


class SingleLayerTransformer(nn.Module):
    """Single-layer attention transformer for binary classification."""
    
    def __init__(self, 
                 d_input: int = 20,
                 d_model: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)
        
        # Attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output layers
        self.output_proj = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with small values"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
    
    def forward(self, context_x, context_y, query_x):
        """Forward pass for attention mechanism"""
        context_h = self.input_proj(context_x)  # [B, N, D]
        query_h = self.input_proj(query_x)      # [B, D]
        
        # Compute Q, K, V
        q = self.q_proj(query_h.unsqueeze(1))   # [B, 1, D]
        k = self.k_proj(context_h)              # [B, N, D]
        
        # Include labels in value computation
        context_y_expanded = context_y.unsqueeze(-1)  # [B, N, 1]
        v = self.v_proj(context_h) * context_y_expanded  # [B, N, D]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))  # [B, 1, N]
        scores = scores / math.sqrt(k.size(-1))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Compute weighted sum of values
        out = torch.matmul(attn, v)  # [B, 1, D]
        out = self.norm1(out)
        
        # Project to output
        logits = self.output_proj(out).squeeze(-1).squeeze(-1)  # [B]
        return logits
    
    def predict(self, context_x, context_y, query_x):
        """Make binary predictions"""
        with torch.no_grad():
            logits = self(context_x, context_y, query_x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            return preds


class LinearClassifier(nn.Module):
    """Linear classifier for in-context learning."""
    
    def __init__(self, d: int):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.W.weight)

    def forward(self, x_ctx, y_ctx, x_tgt):
        """Forward pass"""
        B, N, d = x_ctx.shape

        y_signal = 2 * y_ctx - 1 
        
        weighted = y_signal.unsqueeze(-1) * x_ctx   
        mu_hat = weighted.mean(dim=1)            

        v = self.W(mu_hat)     

        return (v * x_tgt).sum(dim=1)
    
    def compute_in_context_preds(self, x_ctx, y_ctx):
        """Compute in-context predictions"""
        B, N, d = x_ctx.shape
        y_signal = 2 * y_ctx - 1

        mu_hat = (y_signal.unsqueeze(-1) * x_ctx).mean(dim=1)  
        v = self.W(mu_hat)             
        logits = (v.unsqueeze(1) * x_ctx).sum(dim=2)           
        
        return (logits > 0).float()
