import math
from .init_basic import *
from typing import Any


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    if shape is None:
        shape = (fan_in, fan_out)
    return rand(*shape, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    if shape is None:
        shape = (fan_in, fan_out)
    return randn(*shape, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3 / fan_in)
    if shape is None:
        shape = (fan_in, fan_out)
    return rand(*shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3 / fan_in)
    if shape is None:
        shape = (fan_in, fan_out)
    return rand(*shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION

def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan_in)
    if shape is None:
        shape = (fan_in, fan_out)
    return randn(*shape, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION