"""Split a weight matrix of shape (in, out) into equal shards."""
import numpy as np


def split_column(W, k):
    """Split W into k shards along its output dimension."""
    out = W.shape[1]
    if out % k != 0:
        raise ValueError(f"output width {out} is not divisible by k={k}")
    return np.split(W, k, axis=1)


def split_row(W, k):
    """Split W into k shards along its input dimension."""
    in_ = W.shape[0]
    if in_ % k != 0:
        raise ValueError(f"input width {in_} is not divisible by k={k}")
    return np.split(W, k, axis=0)
