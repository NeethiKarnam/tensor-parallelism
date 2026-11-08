"""Run from the repo root:  NEEDLE_BACKEND=nd python tests_base/smoke_test.py (or NEEDLE_BACKEND=np)
"""
import os, sys, traceback
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import needle as ndl
from needle import nn

def reference(x, W, b):
    z = x @ W + b
    y = np.maximum(z, 0)
    mask = (z > 0).astype("float32")
    g = mask                                   # d sum(y) / dz
    return y, x.T @ g, g.sum(0), g @ W.T       # y, dW, db, dx

def run(device):
    np.random.seed(0)
    x = np.random.randn(8, 16).astype("float32")
    lin = nn.Linear(16, 32, device=device)
    X = ndl.Tensor(x, device=device)
    Y = nn.ReLU()(lin(X))
    Y.sum().backward()
    W, b = lin.weight.numpy(), lin.bias.numpy()
    y, dW, db, dx = reference(x, W, b)
    for name, got, want in [("forward", Y.numpy(), y), ("dW", lin.weight.grad.numpy(), dW),
                            ("db", lin.bias.grad.numpy(), db), ("dx", X.grad.numpy(), dx)]:
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5, err_msg=name)

devices = {"np": [("numpy", lambda: ndl.cpu())],
           "nd": [("cpu_numpy", lambda: ndl.cpu_numpy()), ("cpu (C++)", lambda: ndl.cpu())]}
for name, make in devices[os.environ.get("NEEDLE_BACKEND", "nd")]:
    try:
        run(make()); print(f"PASS  {name}")
    except Exception as e:
        last = traceback.extract_tb(e.__traceback__)[-1]
        print(f"FAIL  {name}: {type(e).__name__}: {str(e).splitlines()[0][:110]}  [{os.path.basename(last.filename)}:{last.lineno}]")