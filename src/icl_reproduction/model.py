import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt

class LinearClassifier(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        
        #Linear layer
        self.W = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.W.weight)

    #only forward pass
    def forward(self, x_ctx, y_ctx, x_tgt):
        B, N, d = x_ctx.shape

        y_signal = 2*y_ctx - 1 
        
        weighted = y_signal.unsqueeze(-1) * x_ctx   
        mu_hat = weighted.mean(dim=1)            

        v = self.W(mu_hat)     

        return (v * x_tgt).sum(dim=1)
    
    #use same model as target prediction to predict in context labels
    def compute_in_context_preds(self, x_ctx, y_ctx):

        B, N, d = x_ctx.shape
        y_signal = 2*y_ctx - 1

        mu_hat = (y_signal.unsqueeze(-1) * x_ctx).mean(dim=1)  
        v = self.W(mu_hat)             
        logits = (v.unsqueeze(1) * x_ctx).sum(dim=2)           
        
        return (logits > 0).float()

#Tests

#test output shape after forward pass
def test_forward_shapes():
    d = 100
    N = 5
    B = 10

    x_ctx = torch.randn(B, N, d)
    y_ctx = (torch.rand(B, N) > 0.5).float()
    x_tgt = torch.randn(B, d)

    model = LinearClassifier(d)

    logits = model(x_ctx, y_ctx, x_tgt)

    print("logits shape:", logits.shape)
    assert logits.shape == (B,)

test_forward_shapes()

#forward pass is working and gradients exist 

def test_pass_and_grad():
    d = 50
    N = 3
    B = 4

    x_ctx = torch.randn(B, N, d, requires_grad=True)
    y_ctx = (torch.rand(B, N) > 0.5).float()
    x_tgt = torch.randn(B, d, requires_grad=True)

    model = LinearClassifier(d)

    logits = model(x_ctx, y_ctx, x_tgt).sum()
    logits.backward()

    assert model.W.weight.grad is not None
    print("Gradient shape:", model.W.weight.grad.shape)
    
test_pass_and_grad()

print("All tests passed!")