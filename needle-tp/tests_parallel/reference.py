"""Dense numpy references for the sharded forward pass."""
import numpy as np


def first_linear(x, W1):
    return x @ W1


def dense_mlp(x, W1, W2):
    return np.maximum(x @ W1, 0) @ W2
