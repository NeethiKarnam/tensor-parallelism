import sys
import os
import numpy as np

# Ensure needle-tp is in the python path
# sys.path.append('/content/drive/MyDrive/tensor-parallelism/needle-tp')

# Create a mock distributed module dynamically so the layers can import and execute it
import types
mock_dist = types.ModuleType('needle.distributed')
mock_dist.is_initialized = lambda: True
mock_dist.get_world_size = lambda: 1
mock_dist.get_rank = lambda: 0
mock_dist.all_reduce = lambda x: x  # Identity in a single-process/mock environment

# Inject mock module into sys.modules before importing the parallel layers
sys.modules['needle.distributed'] = mock_dist

import needle as ndl
import needle.nn as nn
from needle.autograd import Tensor, TensorOp

from needle.parallel.layers import ColumnParallelLinear, RowParallelLinear, TPMLP

def test_mock_parallel_correctness():
    print("--- Running TP MLP Correctness Tests with Mock Distributed Backend (Tolerance: 1e-5) ---")
    np.random.seed(42)
    
    # Dimensions for testing
    B = 4
    d = 8
    h = 16
    
    # Create a dense MLP using two linear layers
    dense_lin1 = nn.Linear(d, h, bias=True)
    dense_lin2 = nn.Linear(h, d, bias=True)
    
    X_np = np.random.randn(B, d).astype(np.float32)
    X = Tensor(X_np)
    
    # 1. Test TPMLP from_dense and Forward correctness under mock
    try:
        tp_mlp = TPMLP.from_dense(dense_lin1, dense_lin2)
        
        # Dense Forward
        dense_out = dense_lin2(nn.ReLU()(dense_lin1(X)))
        
        # Parallel Forward (Should behave identically to dense forward with world_size = 1)
        parallel_out = tp_mlp(X)
        
        diff_forward = np.max(np.abs(dense_out.numpy() - parallel_out.numpy()))
        print(f"Max Forward Difference: {diff_forward:.2e}")
        assert diff_forward < 1e-5, f"Forward difference too high: {diff_forward}"
        print("✅ Mock Forward evaluation check PASSED.")
        
    except Exception as e:
        print(f"❌ Mock Forward evaluation failed with exception: {e}")
        return

    # 2. Test Gradient Correctness (Backward) under mock
    try:
        target = Tensor(np.random.randn(B, d).astype(np.float32))
        
        # Dense Backward
        dense_out = dense_lin2(nn.ReLU()(dense_lin1(X)))
        dense_loss = ((dense_out - target) ** 2).sum()
        dense_loss.backward()
        
        # Parallel Backward
        parallel_out = tp_mlp(X)
        parallel_loss = ((parallel_out - target) ** 2).sum()
        parallel_loss.backward()
        
        # Compare Loss
        diff_loss = float(np.abs(dense_loss.numpy() - parallel_loss.numpy()).max())
        print(f"Loss Difference: {diff_loss:.2e}")
        assert diff_loss < 1e-5, f"Loss difference too high: {diff_loss}"
        
        # Compare column parallel weights gradients (should match full dense layer since world_size = 1)
        dense_gW1 = dense_lin1.weight.grad.numpy()
        parallel_gW1 = tp_mlp.col_linear.weight.grad.numpy()
        
        diff_gW1 = np.max(np.abs(dense_gW1 - parallel_gW1))
        print(f"ColParallel Weight Gradient Difference: {diff_gW1:.2e}")
        assert diff_gW1 < 1e-5, f"ColumnParallel gradient difference too high: {diff_gW1}"
        
        print("✅ Mock Backward evaluation check PASSED.")
        
    except Exception as e:
        print(f"❌ Mock Backward evaluation failed with exception: {e}")

if __name__ == '__main__':
    test_mock_parallel_correctness()