from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class RegrTask:
    W: np.ndarray
    X: np.ndarray
    y: np.ndarray
    x_test: np.ndarray
    y_test: float


def sample_regr_task(
    d: int = 10,
    N: Optional[int] = None,
    alpha: float = 1.0,
    noise: float = 0.0,
) -> RegrTask:
    """
    Generate a simple linear regression task:
        y = X @ W + ε,   ε ~ N(0, noise^2)
    """
    if N is None:
        N = 2 * d + 1

    W = np.random.randn(d)
    X = np.random.uniform(-alpha, alpha, size=(N, d))
    y = X @ W + noise * np.random.randn(N)

    x_t = np.random.uniform(-alpha, alpha, size=d)
    y_t = x_t @ W + noise * np.random.randn()

    return RegrTask(W, X, y, x_t, y_t)
