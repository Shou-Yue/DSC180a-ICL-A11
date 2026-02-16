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
        # Call parent class constructor to initialize nn.Module
        super().__init__()
        
        # Store input dimension d (feature dimension of input vectors)
        self.d = d
        
        # Set d_model: if not provided, use d; ensure it's divisible by n_heads (required by GPT2)
        # Round up to nearest multiple of n_heads to satisfy GPT2's requirement
        if d_model is None:
            self.d_model = ((d + n_heads - 1) // n_heads) * n_heads
        else:
            self.d_model = ((d_model + n_heads - 1) // n_heads) * n_heads
        
        # Create GPT2Config object to configure the decoder-only transformer architecture
        config = GPT2Config(
            vocab_size=1,  # Set to 1 since we're not using tokenization, we embed directly
            n_positions=1024,  # Maximum sequence length the model can handle
            n_embd=self.d_model,  # Embedding dimension (must match our internal representation size)
            n_layer=n_layers,  # Number of transformer decoder layers to stack
            n_head=n_heads,  # Number of attention heads in each layer
            resid_pdrop=0.0,  # Dropout probability for residual connections (disabled)
            embd_pdrop=0.0,  # Dropout probability for embeddings (disabled)
            attn_pdrop=0.0,  # Dropout probability for attention weights (disabled)
            use_cache=False,  # Disable caching for training (saves memory, not needed here)
        )
        

        # Initialize the GPT2Model (decoder-only transformer) with our config
        self.transformer = GPT2Model(config)
        
        # Linear layer to embed input features x_ctx from d dimensions to d_model dimensions
        # bias=False means no bias term (similar to LinearClassifier style)
        self.input_embed = nn.Linear(d, self.d_model, bias=False)
        
        # Linear layer to embed binary labels y_ctx (1 dim) to d_model dimensions
        # This allows the label information to be combined with input features
        self.label_embed = nn.Linear(1, self.d_model, bias=False)
        
        # Linear layer to project from d_model back to d dimensions for final prediction
        # This projects the transformer's output representation back to original feature space
        self.output_proj = nn.Linear(self.d_model, d, bias=False)
        
        # Initialize output projection weights to zeros (similar to LinearClassifier initialization)
        # This ensures the model starts from a similar baseline
        nn.init.zeros_(self.output_proj.weight)

    def forward(self, x_ctx, y_ctx, x_tgt):
        # Unpack dimensions: B=batch size, N=number of context examples, d=feature dimension
        B, N, d = x_ctx.shape
        
        # Embed input context features: map each (B, N, d) input to (B, N, d_model)
        # This projects input features into the transformer's embedding space
        x_emb = self.input_embed(x_ctx) 
        
        # Embed labels: first add dimension (B, N) -> (B, N, 1), then embed to (B, N, d_model)
        # This converts binary labels into learnable embeddings
        y_emb = self.label_embed(y_ctx.unsqueeze(-1))  
        
        # Combine embeddings: element-wise addition merges feature and label information
        # This creates a unified representation where each context example has both x and y info
        ctx_emb = x_emb + y_emb  
        
        # Pass combined embeddings through the transformer (decoder-only architecture)
        # inputs_embeds bypasses tokenization and directly uses our embeddings
        # Returns an object containing last_hidden_state and other outputs
        transformer_outputs = self.transformer(inputs_embeds=ctx_emb)
        
        # Extract the last hidden states from all transformer layers
        # Shape: (B, N, d_model) - one d_model-dim vector per context example
        hidden_states = transformer_outputs.last_hidden_state 
        
        # Aggregate context information: mean pooling over sequence dimension
        # This combines information from all N context examples into single (B, d_model) representation
        ctx_repr = hidden_states.mean(dim=1)  
        
        # Project aggregated representation back to original feature space d
        # Shape: (B, d_model) -> (B, d)
        v = self.output_proj(ctx_repr)  
        
        # Compute logits: element-wise multiplication then sum over feature dimension
        # This computes dot product similarity between learned vector v and target x_tgt
        # Shape: (B, d) * (B, d) -> (B, d) -> sum -> (B,)
        logits = (v * x_tgt).sum(dim=1) 
        
        # Return raw logits (not probabilities) for binary classification
        return logits
    
    def compute_in_context_preds(self, x_ctx, y_ctx):
        # Unpack dimensions: B=batch size, N=number of context examples, d=feature dimension
        B, N, d = x_ctx.shape
        
        # Embed input context features: map (B, N, d) to (B, N, d_model)
        x_emb = self.input_embed(x_ctx) 
        
        # Embed labels: add dimension and embed (B, N) -> (B, N, 1) -> (B, N, d_model)
        y_emb = self.label_embed(y_ctx.unsqueeze(-1))
        
        # Combine feature and label embeddings via element-wise addition
        ctx_emb = x_emb + y_emb 
        

        # Process through transformer: get contextualized representations
        # Each position now has information from all other positions via self-attention
        transformer_outputs = self.transformer(inputs_embeds=ctx_emb)
        
        # Extract hidden states: (B, N, d_model) - one vector per context example position
        hidden_states = transformer_outputs.last_hidden_state 
        

        # Project hidden states back to original feature dimension d
        # Shape: (B, N, d_model) -> (B, N, d)
        hidden_proj = self.output_proj(hidden_states)  
        
        # Compute logits for each context position: element-wise multiply with original x_ctx
        # then sum over feature dimension to get prediction score for each context example
        # Shape: (B, N, d) * (B, N, d) -> (B, N, d) -> sum -> (B, N)
        logits = (hidden_proj * x_ctx).sum(dim=2)
        
        # Convert logits to binary predictions: > 0 -> 1.0, <= 0 -> 0.0
        # Returns (B, N) tensor of predicted labels
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
