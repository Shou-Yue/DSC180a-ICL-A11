import numpy as np

def mean_squared_error(pred, target):
    return ((pred - target) ** 2).mean()