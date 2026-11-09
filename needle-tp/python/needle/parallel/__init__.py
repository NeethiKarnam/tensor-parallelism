"""Tensor parallelism for Needle: forward-pass proof of concept (single process)."""
from .sharding import split_column, split_row
from .mlp import sharded_mlp_steps, sharded_mlp_forward

__all__ = ["split_column", "split_row", "sharded_mlp_steps", "sharded_mlp_forward"]
