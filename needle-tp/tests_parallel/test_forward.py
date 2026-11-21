"""Forward-pass checks for the sharded MLP.

Run from needle-tp/:  NEEDLE_BACKEND=np python tests_parallel/test_forward.py
"""
import os
import sys
import traceback

if os.environ.get("NEEDLE_BACKEND") != "np":
    print("These tests require NEEDLE_BACKEND=np.")
    sys.exit(1)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))  # the needle package lives here

import numpy as np
import needle as ndl
from needle.parallel import sharded_mlp_steps, split_column, split_row
from tests_parallel.reference import dense_mlp, first_linear

device = ndl.cpu()
results = []


def f32(values):
    return np.array(values, dtype=np.float32)


def concat(tensors):
    return np.concatenate([t.numpy() for t in tensors], axis=1)


def show(label, array):
    """Print a matrix with its shape, abbreviating large ones."""
    text = np.array2string(
        array, precision=4, suppress_small=True, threshold=30, edgeitems=2
    )
    print(f"{label}, shape {array.shape}:")
    print("  " + text.replace("\n", "\n  "))


def show_steps(steps, W1, W2, k):
    """Print each rank's shards and outputs, in the order they are computed."""
    A, B = split_column(W1, k), split_row(W2, k)
    for i in range(k):
        print(f"--- rank {i} of {k} ---")
        show(f"A[{i}] (column shard of W1)", A[i])
        show(f"col[{i}] = X @ A[{i}]", steps["col"][i].numpy())
        show(f"relu[{i}] = ReLU(col[{i}])", steps["relu"][i].numpy())
        show(f"B[{i}] (row shard of W2)", B[i])
        show(f"partial[{i}] = relu[{i}] @ B[{i}]", steps["partial"][i].numpy())
    print("--- all ranks ---")
    show("out = sum of partials", steps["out"].numpy())


def check(name, fn):
    """Run one check, print PASS or FAIL, and record the result."""
    try:
        fn()
    except AssertionError as e:
        print(f"FAIL  {name}\n{e}")
        results.append(False)
    else:
        print(f"PASS  {name}")
        results.append(True)


def test_hand_checked():
    X_np = f32([[1, 2], [-1, 1]])
    W1_np = f32([[1, 0, -1, 2], [0, 1, 1, -1]])
    W2_np = f32([[3, 0], [0, 1], [1, 2], [2, -1]])

    expected_col = f32([[1, 2, 1, 0], [-1, 1, 2, -3]])
    expected_relu = f32([[1, 2, 1, 0], [0, 1, 2, 0]])
    expected_partial = [f32([[3, 2], [0, 1]]), f32([[1, 2], [2, 4]])]
    expected_out = f32([[4, 4], [2, 5]])

    steps = sharded_mlp_steps(ndl.Tensor(X_np, device=device), W1_np, W2_np, 2, device)

    print("=== Test A: hand-checked example, k=2 ===")
    show("X", X_np)
    show("W1", W1_np)
    show("W2", W2_np)
    show_steps(steps, W1_np, W2_np, 2)
    show("expected out (literal)", expected_out)

    def exact(got, want, err_msg):
        np.testing.assert_allclose(got, want, rtol=0, atol=0, err_msg=err_msg)

    def column_layer():
        col = concat(steps["col"])
        exact(col, expected_col, "column outputs vs literal")
        exact(col, first_linear(X_np, W1_np), "column outputs vs numpy")

    def all_reduce():
        exact(concat(steps["relu"]), expected_relu, "ReLU outputs vs literal")
        for i, want in enumerate(expected_partial):
            exact(steps["partial"][i].numpy(), want, f"rank {i} partial vs literal")
        out = steps["out"].numpy()
        exact(out, expected_out, "final output vs literal")
        exact(out, dense_mlp(X_np, W1_np, W2_np), "final output vs numpy")

    check("A check 1 (column layer)", column_layer)
    check("A check 2 (all-reduce)", all_reduce)


def test_random():
    np.random.seed(0)
    X_np = np.random.randn(8, 16).astype("float32")
    W1_np = np.random.randn(16, 64).astype("float32")
    W2_np = np.random.randn(64, 16).astype("float32")
    X = ndl.Tensor(X_np, device=device)

    def close(got, want, err_msg):
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4, err_msg=err_msg)

    want_col = first_linear(X_np, W1_np)
    want_out = dense_mlp(X_np, W1_np, W2_np)

    print("=== Test B: random inputs ===")
    show("X", X_np)
    show("W1", W1_np)
    show("W2", W2_np)

    for k in (1, 2, 4):
        steps = sharded_mlp_steps(X, W1_np, W2_np, k, device)
        col = concat(steps["col"])
        out = steps["out"].numpy()
        print(f"=== Test B: k={k} ===")
        show_steps(steps, W1_np, W2_np, k)
        show("numpy reference for out", want_out)
        print(
            f"k={k}: column max|err| = {np.max(np.abs(col - want_col)):.2e}, "
            f"all-reduce max|err| = {np.max(np.abs(out - want_out)):.2e}"
        )
        check(f"B k={k} check 1 (column layer)", lambda: close(col, want_col, "column layer"))
        check(f"B k={k} check 2 (all-reduce)", lambda: close(out, want_out, "all-reduce"))


def main():
    for test in (test_hand_checked, test_random):
        try:
            test()
        except Exception:
            print(f"FAIL  {test.__name__} raised an exception")
            traceback.print_exc(file=sys.stdout)
            results.append(False)

    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
