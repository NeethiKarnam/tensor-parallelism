import math
from typing import Any

from .init_basic import *


def kaiming_uniform(fan_in: int, fan_out: int, shape: tuple | None = None, nonlinearity: str = "relu", **kwargs: Any):
    """Uniform on [-bound, bound] with bound = sqrt(2) * sqrt(3 / fan_in).

    `shape` defaults to (fan_in, fan_out). Pass it to draw a tensor whose shape
    differs from the fan values, e.g. a row-parallel shard of shape (in/N, out)
    that must still use the full fan_in.
    """
    assert nonlinearity == "relu", "Only relu supported currently"
    bound = math.sqrt(2) * math.sqrt(3 / fan_in)
    if shape is None:
        shape = (fan_in, fan_out)
    return rand(*shape, low=-bound, high=bound, **kwargs)
