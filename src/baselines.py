import numpy as np


def gd1_regr_predict(
    X: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    eta: float = 1.0,
) -> float:
    """
    Perform one gradient-descent step on squared-loss linear regression
    starting from W=0, then return prediction on x_test.
    """
    N = X.shape[0]
    # One GD step from W0 = 0
    # W1 = W0 - eta * (1/N) * X^T(XW0 - y) = (eta/N) * X^T y
    W1 = (eta / N) * (X.T @ y)

    return float(W1 @ x_test)
