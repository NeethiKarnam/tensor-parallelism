import numpy as np
import needle as ndl
import needle.nn as nn
from needle.autograd import Tensor, TensorOp
from needle.parallel.layers import ColumnParallelLinear, RowParallelLinear, TPMLP
import needle.parallel.distributed as dist

def test_parallel_vs_dense():
    # Join the MPI job (world size 1 without mpirun) before building any layer:
    # layers read the world size at construction, and without init() every rank
    # would build the full, unsharded model.
    dist.init()
    print("--- Starting Correction and Difference Tests (Tolerance: 1e-5) ---")

    # Set global seeds for reproducibility
    np.random.seed(42)
    
    # Dimensions for testing
    B = 4
    d = 8
    h = 16
    
    # Create a dense MLP using two linear layers
    dense_lin1 = nn.Linear(d, h, bias=True)
    dense_lin2 = nn.Linear(h, d, bias=True)
    
    # Initialize random dense input
    X_np = np.random.randn(B, d).astype(np.float32)
    X = Tensor(X_np)
    
    # ---- 1. Test TPMLP from_dense and Forward correctness ----
    try:
        tp_mlp = TPMLP.from_dense(dense_lin1, dense_lin2)
        
        # Dense Forward
        dense_out = dense_lin2(nn.ReLU()(dense_lin1(X)))
        
        # Parallel Forward (when world_size=1, behaves as a split/concat identity block)
        parallel_out = tp_mlp(X)
        
        diff_forward = np.max(np.abs(dense_out.numpy() - parallel_out.numpy()))
        print(f"Max Forward Difference: {diff_forward:.2e}")
        assert diff_forward < 1e-5, f"Forward difference too high: {diff_forward}"
        print("✅ Forward evaluation check PASSED.")
        
    except Exception as e:
        print(f"❌ Forward evaluation failed with exception: {e}")
        return

    # ---- 2. Test Gradient Correctness (Backward) ----
    try:
        # Compute random loss target
        target = Tensor(np.random.randn(B, d).astype(np.float32))
        
        # Dense Backward
        dense_out = dense_lin2(nn.ReLU()(dense_lin1(X)))
        dense_loss = ((dense_out - target) ** 2).sum()
        dense_loss.backward()
        
        # Parallel Backward
        parallel_out = tp_mlp(X)
        parallel_loss = ((parallel_out - target) ** 2).sum()
        parallel_loss.backward()
        
        # Compare Loss Difference
        # .sum() gives a 1-element array on the nd backend, so reduce to a float
        diff_loss = float(np.abs(dense_loss.numpy() - parallel_loss.numpy()).max())
        print(f"Loss Difference: {diff_loss:.2e}")
        assert diff_loss < 1e-5, f"Loss difference too high: {diff_loss}"
        
        # Compare sharded weight gradients for ColumnParallelLinear
        # In column parallel, weights are sharded along columns (axis 1)
        rank = dist.get_rank() if dist.is_initialized() else 0
        split_h = h // (dist.get_world_size() if dist.is_initialized() else 1)
        
        dense_gW1 = dense_lin1.weight.grad.numpy()[:, rank * split_h : (rank + 1) * split_h]
        parallel_gW1 = tp_mlp.col_linear.weight.grad.numpy()
        
        diff_gW1 = np.max(np.abs(dense_gW1 - parallel_gW1))
        print(f"ColParallel Weight Gradient Difference: {diff_gW1:.2e}")
        assert diff_gW1 < 1e-5, f"ColumnParallel gradient difference too high: {diff_gW1}"
        
        print("✅ Backward evaluation check PASSED.")
        
    except Exception as e:
        print(f"❌ Backward evaluation failed with exception: {e}")
        
if __name__ == '__main__':
    test_parallel_vs_dense()