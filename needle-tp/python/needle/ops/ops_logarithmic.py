from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *


from ..backend_selection import array_api, BACKEND

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(axis=1, keepdims=True).broadcast_to(Z.shape)
        shifted = Z - max_z
        log_sum = shifted.exp().sum(axis=1, keepdims=True).log().broadcast_to(Z.shape)
        return shifted - log_sum
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        softmax_z = exp(node)

        grad_sum = summation(out_grad, (1,))
        grad_sum = reshape(grad_sum, (grad_sum.shape[0], 1))
        grad_sum = broadcast_to(grad_sum, node.shape)

        return (out_grad - softmax_z * grad_sum)
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_red = Z.max(axis=self.axes)
        if self.axes is None:
            keep = (1,) * Z.ndim
        else:
            axes = (self.axes,) if isinstance(self.axes, int) else tuple(self.axes)
            axes = tuple(a % Z.ndim for a in axes)
            keep = tuple(1 if i in axes else s for i, s in enumerate(Z.shape))
        shifted = Z - max_red.reshape(keep).broadcast_to(Z.shape)
        return shifted.exp().sum(axis=self.axes).log() + max_red
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        input_shape = Z.shape

        if self.axes is None:
            axes = tuple(range(len(input_shape)))
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = self.axes

        axes = tuple(
            axis if axis >= 0 else axis + len(input_shape)
            for axis in axes
        )

        reduced_shape = list(input_shape)
        for axis in axes:
            reduced_shape[axis] = 1
        reduced_shape = tuple(reduced_shape)

      
        logsumexp_broadcast = broadcast_to(reshape(node, reduced_shape), input_shape)
        softmax = exp(Z - logsumexp_broadcast)

        out_grad_broadcast = broadcast_to(reshape(out_grad, reduced_shape), input_shape)

        return (out_grad_broadcast * softmax)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)