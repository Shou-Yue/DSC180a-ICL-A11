import torch

def sample_gaussian_xs(batch_size, n_points, n_dims, n_dims_trunc, device):
    """
    Sample Gaussian inputs and optionally truncate to a lower-dimensional
    subspace by zeroing out the last coordinates.
    """
    xs = torch.randn(batch_size, n_points, n_dims, device = device)
    
    if n_dims_trunc is not None and n_dims_trunc < n_dims:
        xs[:, :, n_dims_trunc:] = 0.0
        
    return xs

def sample_linear_regression_weights(batch_size, n_dims, device, sparsity = None):
    """
    Sample one weight vector per batch element, w ~ N(0, I).
    """
    w = torch.randn(batch_size, n_dims, 1, device = device)

    if sparsity is None or sparsity >= n_dims:
        return w

    for b in range(batch_size):
        idx = torch.randperm(n_dims, device = device)[:sparsity]
        mask = torch.zeros(n_dims, 1, device = device, dtype = torch.bool)
        mask[idx] = True
        w[b, ~mask] = 0.0

    return w

def evaluate_linear_regression(xs, w, scale = 1.0):
    """
    xs: [B, T, D]
    w: [B, D, 1]

    Returns ys: [B, T]
    """
    ys = scale * (xs @ w)   # [B, T, 1]
    return ys[:, :, 0]

def sample_linear_regression_task(batch_size, n_points, n_dims, device, n_dims_trunc = None, sparsity = None, scale = 1.0):
    xs = sample_gaussian_xs(
        batch_size = batch_size,
        n_points = n_points,
        n_dims = n_dims,
        n_dims_trunc = n_dims_trunc,
        device = device,
    )

    w = sample_linear_regression_weights(
        batch_size = batch_size,
        n_dims = n_dims,
        device = device,
        sparsity = sparsity,
    )

    ys = evaluate_linear_regression(xs, w, scale = scale)

    return xs, ys, w