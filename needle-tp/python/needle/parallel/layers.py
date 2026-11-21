import needle as ndl
import needle.nn as nn
import numpy as np
from needle.autograd import Tensor
from needle.parallel.ops import f, g
import needle.distributed as dist


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_flag = bias

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # out_features is split along column dimension (axis 1)
        assert out_features % self.world_size == 0, "out_features must be divisible by world_size"
        self.split_out_features = out_features // self.world_size

        # Needle layout for weights is (in_features, out_features) for X @ W
        self.weight = nn.Parameter(
            ndl.init.randn(in_features, self.split_out_features, device=device, dtype=dtype)
        )

        if self.bias_flag:
            # 2D bias (1, split_out), matching nn.Linear's (1, out) layout
            self.bias = nn.Parameter(
                ndl.init.zeros(1, self.split_out_features, device=device, dtype=dtype)
            )
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        # Input is duplicated across all devices (via f op)
        input_parallel = f(X)
        output = input_parallel @ self.weight
        if self.bias is not None:
            output = output + self.bias.broadcast_to(output.shape)
        return output

    @classmethod
    def from_dense(cls, dense_linear, device=None, dtype="float32"):
        in_features = dense_linear.in_features
        out_features = dense_linear.out_features
        bias = dense_linear.bias is not None

        instance = cls(in_features, out_features, bias=bias, device=device, dtype=dtype)

        rank = instance.rank
        start_col = rank * instance.split_out_features
        end_col = (rank + 1) * instance.split_out_features

        full_weight_numpy = dense_linear.weight.numpy()
        instance.weight.data = Tensor(full_weight_numpy[:, start_col:end_col], device=device, dtype=dtype)

        if bias:
            # Dense bias is (1, out_features): slice along axis 1
            full_bias_numpy = dense_linear.bias.numpy().reshape(1, -1)
            instance.bias.data = Tensor(full_bias_numpy[:, start_col:end_col], device=device, dtype=dtype)

        return instance


class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_flag = bias

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # in_features is split along row dimension (axis 0)
        assert in_features % self.world_size == 0, "in_features must be divisible by world_size"
        self.split_in_features = in_features // self.world_size

        self.weight = nn.Parameter(
            ndl.init.randn(self.split_in_features, out_features, device=device, dtype=dtype)
        )

        # Full (1, out_features) bias on every rank; added after the g (all-reduce) op
        if self.bias_flag:
            self.bias = nn.Parameter(
                ndl.init.zeros(1, out_features, device=device, dtype=dtype)
            )
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        # X is already sharded at (B, split_in_features)
        output_sharded = X @ self.weight
        # Reduce output across all devices
        output = g(output_sharded)
        # Row bias is added after g
        if self.bias is not None:
            output = output + self.bias.broadcast_to(output.shape)
        return output

    @classmethod
    def from_dense(cls, dense_linear, device=None, dtype="float32"):
        in_features = dense_linear.in_features
        out_features = dense_linear.out_features
        bias = dense_linear.bias is not None

        instance = cls(in_features, out_features, bias=bias, device=device, dtype=dtype)

        rank = instance.rank
        start_row = rank * instance.split_in_features
        end_row = (rank + 1) * instance.split_in_features

        full_weight_numpy = dense_linear.weight.numpy()
        instance.weight.data = Tensor(full_weight_numpy[start_row:end_row, :], device=device, dtype=dtype)

        if bias:
            # Full bias is used after the reduction, so no slicing
            instance.bias.data = Tensor(dense_linear.bias.numpy().reshape(1, -1), device=device, dtype=dtype)

        return instance


class TPMLP(nn.Module):
    def __init__(self, d, h, bias=True, device=None, dtype="float32"):
        super().__init__()
        # f -> column -> ReLU -> row -> g
        self.col_linear = ColumnParallelLinear(d, h, bias=bias, device=device, dtype=dtype)
        self.row_linear = RowParallelLinear(h, d, bias=bias, device=device, dtype=dtype)
        self.relu = nn.ReLU()

    def forward(self, X: Tensor) -> Tensor:
        x = self.col_linear(X)
        x = self.relu(x)
        x = self.row_linear(x)
        return x

    @classmethod
    def from_dense(cls, dense_lin1, dense_lin2, device=None, dtype="float32"):
        d = dense_lin1.in_features
        h = dense_lin1.out_features
        bias = dense_lin1.bias is not None

        instance = cls(d, h, bias=bias, device=device, dtype=dtype)
        instance.col_linear = ColumnParallelLinear.from_dense(dense_lin1, device=device, dtype=dtype)
        instance.row_linear = RowParallelLinear.from_dense(dense_lin2, device=device, dtype=dtype)
        return instance