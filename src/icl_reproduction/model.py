import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Model, GPT2Config

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

#decoder only mini transformer
class MiniTransformer(nn.Module):
    def __init__(self, d: int, n_heads: int = 4, n_layers: int = 2, d_model: int = None):
        super().__init__()
        self.d = d
        
        config = GPT2Config(
            vocab_size=1,  
            n_positions=1024, 
            n_embd=self.d_model,
            n_layer=n_layers,
            n_head=n_heads,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        

        self.transformer = GPT2Model(config)
        self.input_embed = nn.Linear(d, self.d_model, bias=False)
        self.label_embed = nn.Linear(1, self.d_model, bias=False)
        self.output_proj = nn.Linear(self.d_model, d, bias=False)
        nn.init.zeros_(self.output_proj.weight)

    def forward(self, x_ctx, y_ctx, x_tgt):
        B, N, d = x_ctx.shape
        
        x_emb = self.input_embed(x_ctx) 
        y_emb = self.label_embed(y_ctx.unsqueeze(-1))  
        #complete embeddings
        ctx_emb = x_emb + y_emb  
        
        transformer_outputs = self.transformer(inputs_embeds=ctx_emb)
        hidden_states = transformer_outputs.last_hidden_state 
        ctx_repr = hidden_states.mean(dim=1)  
        
        v = self.output_proj(ctx_repr)  
        logits = (v * x_tgt).sum(dim=1)  # (B,)
        
        return logits
    
    def compute_in_context_preds(self, x_ctx, y_ctx):
        B, N, d = x_ctx.shape
        x_emb = self.input_embed(x_ctx) 
        y_emb = self.label_embed(y_ctx.unsqueeze(-1))
        ctx_emb = x_emb + y_emb 
        

        transformer_outputs = self.transformer(inputs_embeds=ctx_emb)
        hidden_states = transformer_outputs.last_hidden_state 
        

        hidden_proj = self.output_proj(hidden_states)  
        logits = (hidden_proj * x_ctx).sum(dim=2)
        
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

