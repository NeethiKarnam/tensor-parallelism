"""Sharded forward pass for an MLP: column-parallel W1 -> ReLU -> row-parallel W2.

All k shards run in this one process, so the shard index stands in for a rank.
"""
import needle as ndl
from needle import nn

from .sharding import split_column, split_row


def sharded_mlp_steps(X, W1, W2, k, device):
    """Run the sharded forward pass and return every intermediate.

    X is a needle Tensor (B, d). W1 is numpy (d, h). W2 is numpy (h, d).

    Returns a dict with "col" (k pre-activation outputs), "relu" (k activated
    outputs), "partial" (k full-shape partial outputs), and "out" (their sum).
    """
    relu = nn.ReLU()
    col, activated, partial = [], [], []

    for A_np, B_np in zip(split_column(W1, k), split_row(W2, k)):
        A = ndl.Tensor(A_np, device=device)
        B = ndl.Tensor(B_np, device=device)

        col_i = X @ A
        relu_i = relu(col_i)

        col.append(col_i)
        activated.append(relu_i)
        partial.append(relu_i @ B)

    # All-reduce placeholder: a real all-reduce would sum the partials across ranks.
    out = partial[0]
    for p in partial[1:]:
        out = out + p

    return {"col": col, "relu": activated, "partial": partial, "out": out}


def sharded_mlp_forward(X, W1, W2, k, device):
    """Return the output of the sharded forward pass."""
    return sharded_mlp_steps(X, W1, W2, k, device)["out"]
