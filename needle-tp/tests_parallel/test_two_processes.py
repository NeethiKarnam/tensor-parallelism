"""
Two-process tensor-parallel correctness test for needle-tp.

Spawns WORLD_SIZE real processes. Each one gets its own mock `needle.parallel.distributed`
whose all_reduce really communicates (via multiprocessing queues) and sums the
tensors across ranks. Every rank then checks that its TPMLP shard matches the
corresponding slice of a dense MLP built from identical weights.

Run from the repo root:
    NEEDLE_BACKEND=nd python tests_parallel/test_two_process.py
"""
import sys
import types
import queue
import traceback
import multiprocessing as mp

import numpy as np

# Keep needle imports out of module scope: with the "spawn" start method each child
# re-imports this file, and the mock must be installed before needle.parallel loads.

WORLD_SIZE = 2
B, D, H = 4, 8, 16
RTOL, ATOL = 1e-4, 1e-4
COMM_TIMEOUT = 60


def build_mock_dist(rank, world_size, queues):
    """Mock needle.parallel.distributed with an all_reduce(sum) that talks to the other ranks."""
    mock = types.ModuleType("needle.parallel.distributed")
    mock.is_initialized = lambda: True
    mock.get_world_size = lambda: world_size
    mock.get_rank = lambda: rank

    def all_reduce(x):
        # Imported lazily so the mock can be installed before needle is imported.
        from needle.autograd import Tensor
        import needle.backend_ndarray as nd

        # Same contract as the real all_reduce (ROADMAP.md, Interfaces §1): raw NDArray in,
        # new NDArray out. Rejecting Tensors catches f's gradient passing one by mistake.
        if isinstance(x, Tensor):
            raise TypeError("all_reduce takes a raw NDArray, got Tensor")
        local = np.asarray(x.numpy(), dtype=np.float32)

        parts = [None] * world_size
        parts[rank] = local
        for dst in range(world_size):
            if dst != rank:
                queues[rank][dst].put(local)
        for src in range(world_size):
            if src != rank:
                parts[src] = queues[src][rank].get(timeout=COMM_TIMEOUT)

        # Sum in rank order so every rank gets bit-identical results
        total = parts[0].copy()
        for p in parts[1:]:
            total = total + p

        return nd.array(total, device=x.device)

    mock.all_reduce = all_reduce
    return mock


def worker(rank, world_size, queues, result_queue):
    log = []
    state = {"ok": True}

    def check(name, actual, expected):
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        if actual.shape != expected.shape:
            state["ok"] = False
            log.append(f"  ❌ {name}: shape {actual.shape} != expected {expected.shape}")
            return
        diff = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
        if np.allclose(actual, expected, rtol=RTOL, atol=ATOL):
            log.append(f"  ✅ {name} (max diff {diff:.2e})")
        else:
            state["ok"] = False
            log.append(f"  ❌ {name}: max diff {diff:.2e}")

    try:
        # Must happen before anything imports needle.parallel
        sys.modules["needle.parallel.distributed"] = build_mock_dist(rank, world_size, queues)

        import needle as ndl  # noqa: F401
        import needle.nn as nn
        from needle.autograd import Tensor
        from needle.parallel.layers import TPMLP

        dist = sys.modules["needle.parallel.distributed"]

        # ---- 0. all_reduce really sums across processes ----
        local = Tensor(np.full((2, 3), rank + 1, dtype=np.float32)).realize_cached_data()
        reduced = dist.all_reduce(local).numpy()
        expected_sum = world_size * (world_size + 1) / 2
        check("all_reduce sums across ranks", reduced, np.full((2, 3), expected_sum))

        # ---- Identical deterministic data on every rank ----
        rs = np.random.RandomState(0)
        W1 = (rs.randn(D, H) * 0.1).astype(np.float32)
        b1 = (rs.randn(H) * 0.1).astype(np.float32)
        W2 = (rs.randn(H, D) * 0.1).astype(np.float32)
        b2 = (rs.randn(D) * 0.1).astype(np.float32)
        X_np = rs.randn(B, D).astype(np.float32)
        T_np = rs.randn(B, D).astype(np.float32)

        def set_param(p, arr):
            # reshape handles both (out,) and (1, out) bias layouts
            p.data = Tensor(arr.reshape(p.shape).astype(np.float32))

        lin1 = nn.Linear(D, H, bias=True)
        lin2 = nn.Linear(H, D, bias=True)
        set_param(lin1.weight, W1)
        set_param(lin1.bias, b1)
        set_param(lin2.weight, W2)
        set_param(lin2.bias, b2)

        tp = TPMLP.from_dense(lin1, lin2)
        shard = H // world_size
        sl = slice(rank * shard, (rank + 1) * shard)

        # ---- 1. layers are actually sharded ----
        check("col weight shape is sharded", np.zeros(tp.col_linear.weight.shape), np.zeros((D, shard)))
        check("row weight shape is sharded", np.zeros(tp.row_linear.weight.shape), np.zeros((shard, D)))

        # ---- 2. forward ----
        X_dense = Tensor(X_np)
        X_tp = Tensor(X_np)
        target = Tensor(T_np)

        dense_out = lin2(nn.ReLU()(lin1(X_dense)))
        tp_out = tp(X_tp)
        check("forward output", tp_out.numpy(), dense_out.numpy())

        # ---- 3. backward ----
        dense_loss = ((dense_out - target) ** 2).sum()
        tp_loss = ((tp_out - target) ** 2).sum()
        dense_loss.backward()
        tp_loss.backward()

        check("loss", np.reshape(tp_loss.numpy(), -1), np.reshape(dense_loss.numpy(), -1))
        check("col weight grad (shard)", tp.col_linear.weight.grad.numpy(), lin1.weight.grad.numpy()[:, sl])
        check("col bias grad (shard)", tp.col_linear.bias.grad.numpy().reshape(-1), lin1.bias.grad.numpy().reshape(-1)[sl])
        check("row weight grad (shard)", tp.row_linear.weight.grad.numpy(), lin2.weight.grad.numpy()[sl, :])
        check("row bias grad (full)", tp.row_linear.bias.grad.numpy().reshape(-1), lin2.bias.grad.numpy().reshape(-1))
        # X.grad only comes out right if f's backward all-reduces the partial gradients
        check("input grad (f backward all-reduce)", X_tp.grad.numpy(), X_dense.grad.numpy())

    except Exception:
        state["ok"] = False
        log.append("  ❌ exception:\n" + traceback.format_exc())

    result_queue.put((rank, state["ok"], log))


def main():
    ctx = mp.get_context("spawn")
    # queues[src][dst]: one FIFO per ordered pair of ranks
    queues = [[ctx.Queue() if s != d else None for d in range(WORLD_SIZE)] for s in range(WORLD_SIZE)]
    result_queue = ctx.Queue()

    procs = [
        ctx.Process(target=worker, args=(r, WORLD_SIZE, queues, result_queue))
        for r in range(WORLD_SIZE)
    ]
    for p in procs:
        p.start()

    print(f"--- TP MLP correctness, {WORLD_SIZE} processes, all_reduce = sum (rtol={RTOL}, atol={ATOL}) ---")
    results = {}
    try:
        for _ in range(WORLD_SIZE):
            rank, ok, log = result_queue.get(timeout=COMM_TIMEOUT * 3)
            results[rank] = (ok, log)
    except queue.Empty:
        print("❌ Timed out waiting for workers (likely a deadlock or a crashed rank).")
        for p in procs:
            p.terminate()
        sys.exit(1)
    finally:
        for p in procs:
            p.join(timeout=10)

    all_ok = True
    for rank in sorted(results):
        ok, log = results[rank]
        all_ok &= ok
        print(f"[rank {rank}] {'PASSED' if ok else 'FAILED'}")
        print("\n".join(log))

    print("✅ All ranks passed." if all_ok and len(results) == WORLD_SIZE else "❌ Some ranks failed.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()