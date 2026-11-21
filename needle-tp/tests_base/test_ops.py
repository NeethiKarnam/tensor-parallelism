"""Forward + gradient checks of the ops tensor parallelism uses, on needle's nd backend.

Run from needle-tp/:  NEEDLE_BACKEND=nd python tests_base/test_ops.py
Checks every device that is available (cpu, plus cuda when the CUDA backend is built).
"""
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "python"))
import numpy as np
import needle as ndl
from needle import ops

rng = np.random.default_rng(0)
devices = [("cpu", ndl.cpu())]
if ndl.cuda().enabled():
    devices.append(("cuda", ndl.cuda()))


def numeric_grad(f, arrays, i, eps=1e-3):
    """Central-difference gradient of scalar f(*arrays) w.r.t. arrays[i] (float64)."""
    x = arrays[i]
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps; fp = f(*arrays)
        x[idx] = old - eps; fm = f(*arrays)
        x[idx] = old
        g[idx] = (fp - fm) / (2 * eps)
    return g


def check(name, op, np_fn, shapes, device, positive=False):
    arrays = [rng.standard_normal(s) for s in shapes]
    if positive:
        arrays = [np.abs(a) + 0.5 for a in arrays]
    out_shape = np.asarray(np_fn(*arrays)).shape
    w = rng.standard_normal(out_shape)  # random projection -> scalar loss

    # forward
    ts = [ndl.Tensor(a.astype("float32"), device=device) for a in arrays]
    out = op(*ts)
    fwd_err = np.abs(out.numpy().reshape(out_shape) - np_fn(*arrays)).max()

    # backward: loss = sum(out * w)
    W = ndl.Tensor(w.astype("float32"), device=device, requires_grad=False)
    loss = ops.summation(ops.multiply(ops.reshape(out, out_shape) if out.shape != out_shape else out, W))
    loss.backward()
    grad_err = 0.0
    for i, t in enumerate(ts):
        want = numeric_grad(lambda *a: float((np_fn(*a) * w).sum()), arrays, i)
        grad_err = max(grad_err, np.abs(t.grad.numpy() - want).max())
    ok = fwd_err < 1e-4 and grad_err < 2e-2
    print(f"{'PASS' if ok else 'FAIL'}  {device!r:8} {name:28} fwd {fwd_err:.1e}  grad {grad_err:.1e}")
    return ok


def softmax_xent(Z, y_onehot):
    m = Z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(Z - m).sum(axis=1)) + m[:, 0]
    return (lse - (Z * y_onehot).sum(axis=1)).mean()


cases = [
    ("add", ops.add, lambda a, b: a + b, [(4, 5), (4, 5)], False),
    ("multiply", ops.multiply, lambda a, b: a * b, [(4, 5), (4, 5)], False),
    ("divide", ops.divide, lambda a, b: a / b, [(4, 5), (4, 5)], True),
    ("add_scalar", lambda a: ops.add_scalar(a, 2.0), lambda a: a + 2.0, [(4, 5)], False),
    ("mul_scalar", lambda a: ops.mul_scalar(a, 3.0), lambda a: a * 3.0, [(4, 5)], False),
    ("divide_scalar", lambda a: ops.divide_scalar(a, 4.0), lambda a: a / 4.0, [(4, 5)], False),
    ("power_scalar", lambda a: ops.power_scalar(a, 2), lambda a: a ** 2, [(4, 5)], False),
    ("negate", ops.negate, lambda a: -a, [(4, 5)], False),
    ("exp", ops.exp, np.exp, [(4, 5)], False),
    ("log", ops.log, np.log, [(4, 5)], True),
    ("relu", ops.relu, lambda a: np.maximum(a, 0), [(4, 5)], False),
    ("matmul", ops.matmul, lambda a, b: a @ b, [(6, 4), (4, 3)], False),
    ("transpose", ops.transpose, lambda a: a.swapaxes(-1, -2), [(3, 4)], False),
    ("transpose(0,2)", lambda a: ops.transpose(a, (0, 2)), lambda a: a.swapaxes(0, 2), [(2, 3, 4)], False),
    ("reshape", lambda a: ops.reshape(a, (6, 2)), lambda a: a.reshape(6, 2), [(3, 4)], False),
    ("broadcast_to row", lambda a: ops.broadcast_to(a, (5, 4)), lambda a: np.broadcast_to(a, (5, 4)), [(1, 4)], False),
    ("broadcast_to bias", lambda a: ops.broadcast_to(ops.reshape(a, (1, 4)), (5, 4)), lambda a: np.broadcast_to(a.reshape(1, 4), (5, 4)), [(4,)], False),
    ("summation all", ops.summation, lambda a: np.array(a.sum()).reshape(1), [(3, 4)], False),
    ("summation axis 0", lambda a: ops.summation(a, (0,)), lambda a: a.sum(0), [(3, 4)], False),
    ("summation axes (0,2)", lambda a: ops.summation(a, (0, 2)), lambda a: a.sum((0, 2)), [(2, 3, 4)], False),
    ("logsumexp axis 1", lambda a: ops.logsumexp(a, (1,)), lambda a: np.log(np.exp(a).sum(1)), [(4, 5)], False),
    # The project's actual computation: ReLU(X W1) W2 + b, then softmax cross-entropy.
    ("mlp block", lambda x, w1, w2, b: ops.add(ops.matmul(ops.relu(ops.matmul(x, w1)), w2),
                                               ops.broadcast_to(ops.reshape(b, (1, 3)), (6, 3))),
     lambda x, w1, w2, b: np.maximum(x @ w1, 0) @ w2 + b, [(6, 4), (4, 8), (8, 3), (3,)], False),
]

all_ok = True
for dname, dev in devices:
    for name, op, np_fn, shapes, pos in cases:
        try:
            all_ok &= check(name, op, np_fn, shapes, dev, pos)
        except Exception as e:
            all_ok = False
            print(f"FAIL  {dev!r:8} {name:28} {type(e).__name__}: {str(e).splitlines()[0][:90]}")

    # SoftmaxLoss as nn uses it
    try:
        from needle import nn
        Z = rng.standard_normal((6, 5)); y = rng.integers(0, 5, 6)
        Zt = ndl.Tensor(Z.astype("float32"), device=dev)
        loss = nn.SoftmaxLoss()(Zt, ndl.Tensor(y.astype("float32"), device=dev, requires_grad=False))
        loss.backward()
        onehot = np.eye(5)[y]
        want = numeric_grad(lambda z: softmax_xent(z, onehot), [Z], 0)
        fe = abs(float(loss.numpy().reshape(-1)[0]) - softmax_xent(Z, onehot)); ge = np.abs(Zt.grad.numpy() - want).max()
        ok = fe < 1e-4 and ge < 2e-2; all_ok &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {dev!r:8} {'nn.SoftmaxLoss':28} fwd {fe:.1e}  grad {ge:.1e}")
    except Exception as e:
        all_ok = False
        print(f"FAIL  {dev!r:8} {'nn.SoftmaxLoss':28} {type(e).__name__}: {str(e).splitlines()[0][:90]}")

print("ALL PASS" if all_ok else "SOME FAILED")
