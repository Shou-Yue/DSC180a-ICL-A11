import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt

def data_gen(
    d: int,
    N: int,
    B: int,
    R: float,
    flip_prob: float = 0.0,
    flip_context_prob: float | None = None,
    flip_query_prob: float | None = None,
    device: str = "cpu",
    seed = None,
):
    """Generate Gaussian mixture data. By default flip_prob applies to all labels.
    For context-only noise (e.g. RQ2), set flip_context_prob=eps and flip_query_prob=0."""
    if seed is not None:
        #keep base & flip RNG different. the original paper did not do this which led to issues when running their code.
        g_base = torch.Generator(device="cpu").manual_seed(seed)
        g_flip = torch.Generator(device="cpu").manual_seed(seed + 1)
    else:
        g_base = None
        g_flip = None

    mu = torch.randn(B, d, generator=g_base)#(B, d)
    mu = mu / mu.norm(dim=1, keepdim=True)
    mu = R * mu

    labels = (torch.rand(B, N + 1, generator=g_base) > 0.5).float()#(B, N+1)
    y_signal = 2 * labels - 1

    noise = torch.randn(B, N + 1, d, generator=g_base)#(B, N+1, d)

    x = (y_signal.unsqueeze(-1) * mu.unsqueeze(1) + noise)#(B, N+1, d)

    # introduce noise to labels
    use_separate = flip_context_prob is not None or flip_query_prob is not None
    if use_separate:
        p_ctx = flip_context_prob if flip_context_prob is not None else 0.0
        p_q = flip_query_prob if flip_query_prob is not None else 0.0
        if p_ctx > 0:
            flip_ctx = torch.rand(B, N, generator=g_flip) < p_ctx
            labels[:, :N] = torch.where(flip_ctx, 1.0 - labels[:, :N], labels[:, :N])
        if p_q > 0:
            flip_tgt = torch.rand(B, generator=g_flip) < p_q
            labels[:, -1] = torch.where(flip_tgt, 1.0 - labels[:, -1], labels[:, -1])
    elif flip_prob > 0.0:
        flip_mask = torch.rand(B, N + 1, generator=g_flip) < flip_prob
        labels = torch.where(flip_mask, 1.0 - labels, labels)

    #output
    x_context = (x[:, :N, :]).to(device)
    x_target = (x[:, -1, :]).to(device)
    y_context = (labels[:, :N]).to(device)
    y_target = (labels[:, -1]).to(device)

    return (x_context, y_context, x_target, y_target)


#Data Testing

#simple testing
d, N, B, R = 1000, 20, 500, 5.0

x_ctx, y_ctx, x_tgt, y_tgt = data_gen(d, N, B, R, flip_prob=0.0, seed=100)

print("x_ctx shape:", x_ctx.shape)#(B, N, d)
print("y_ctx shape:", y_ctx.shape)#(B, N)
print("x_tgt shape:", x_tgt.shape)#(B, d)
print("y_tgt shape:", y_tgt.shape)#(B,)
#check printed shapes match comments

print("Context label mean:", y_ctx.mean().item())
print("Target label mean:", y_tgt.float().mean().item())
#around 0.5 assuming N is high enough

#noise tests
p = 0.3
seed = 100

#clean set (no flips)
x_ctx_clean, y_ctx_clean, x_tgt_clean, y_tgt_clean = data_gen(
    d, N, B, R, flip_prob=0.0, seed=seed
)

#p=0.3 flips
x_ctx_noisy, y_ctx_noisy, x_tgt_noisy, y_tgt_noisy = data_gen(
    d, N, B, R, flip_prob=p, seed=seed
)

#true for both
print("Context X equal:", torch.allclose(x_ctx_clean, x_ctx_noisy))
print("Target X equal:", torch.allclose(x_tgt_clean, x_tgt_noisy))

#about 0.3 for both
ctx_flip_rate = (y_ctx_clean != y_ctx_noisy).float().mean().item()
tgt_flip_rate = (y_tgt_clean != y_tgt_noisy).float().mean().item()

print(f"Context flip rate ~ {ctx_flip_rate:.3f} (target {p})")
print(f"Target  flip rate ~ {tgt_flip_rate:.3f} (target {p})")


#asserts to verify above tests
assert x_ctx.shape == (B, N, d), f"x_ctx shape mismatch: actual: {x_ctx.shape}, expected: ({[B, N, d]})"
assert y_ctx.shape == (B, N), f"y_ctx shape mismatch: actual: {y_ctx.shape}, expected: ({[B, N]})"
assert x_tgt.shape == (B, d), f"x_tgt shape mismatch: actual: {x_tgt.shape}, expected: ({[B, d]})"
assert y_tgt.shape == (B,), f"y_tgt shape mismatch: actual: {y_tgt.shape}, expected: ({B})"

#should be around 0.5 
ctx_mean = y_ctx.mean().item()
tgt_mean = y_tgt.float().mean().item()

assert 0.4 < ctx_mean < 0.6, f"Context label mean {ctx_mean:.2f} not near 0.5"
assert 0.4 < tgt_mean < 0.6, f"Target label mean {tgt_mean:.2f} not near 0.5"

#clean and noise context and target points should match
assert torch.allclose(x_ctx_clean, x_ctx_noisy), "Context X mismatch in points"
assert torch.allclose(x_tgt_clean, x_tgt_noisy), "Target X mismatch in points"

#flip rates should be roughly p
ctx_flip_rate = (y_ctx_clean != y_ctx_noisy).float().mean().item()
tgt_flip_rate = (y_tgt_clean != y_tgt_noisy).float().mean().item()

assert abs(ctx_flip_rate - p) < 0.05, (
    f"Context flip rate {ctx_flip_rate:.3f} not close to expected {p}"
)

assert abs(tgt_flip_rate - p) < 0.05, (
    f"Target flip rate {tgt_flip_rate:.3f} not close to expected {p}"
)

print("All tests passed!")
