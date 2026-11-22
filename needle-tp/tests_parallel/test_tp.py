"""
Multi-rank tensor-parallel correctness tests.

Spawns N real processes per world size. Each gets a mock `needle.parallel.distributed` whose
all_reduce really sums across processes (multiprocessing queues) and enforces the
agreed contract: NDArray in, NEW NDArray out, same device. No MPI needed.

Checks, on every rank:
  1. all_reduce contract    sums across ranks; rejects Tensor; returns a new array on the same device
  2. one TPMLP block        forward, loss, every shard's grad, and dX match dense
  3. two stacked blocks     same checks for TPMLP -> TPMLP (g directly followed by f)
  4. Adam training          K steps: loss matches every step; shards equal dense slices;
                            replicated row bias stays identical on every rank
  5. direct init            shards use the full fan_in; row bias identical across ranks;
                            column shards differ across ranks (WARN only, see below)

Run from needle-tp/:
    NEEDLE_BACKEND=nd python tests_parallel/test_tp.py            # world sizes 1, 2, 4
    NEEDLE_BACKEND=nd python tests_parallel/test_tp.py 2          # just N = 2
"""
import math
import queue
import sys
import traceback
import types
import multiprocessing as mp

import numpy as np

# Keep needle imports out of module scope: with the "spawn" start method each child
# re-imports this file, and the mock must be installed before needle.parallel loads.

B, D, H = 4, 8, 16          # H must be divisible by every world size tested
K_STEPS = 5
LR = 1e-2
RTOL, ATOL = 1e-4, 1e-5
COMM_TIMEOUT = 60
DEFAULT_WORLD_SIZES = (1, 2, 4)


# --------------------------------------------------------------------------- mock comm

def build_mock_dist(rank, world_size, queues):
    """Mock needle.parallel.distributed. all_reduce follows the agreed contract strictly."""
    mock = types.ModuleType("needle.parallel.distributed")
    mock.is_initialized = lambda: True
    mock.get_world_size = lambda: world_size
    mock.get_rank = lambda: rank

    def all_reduce(x):
        import needle.backend_ndarray as nd

        if not isinstance(x, nd.NDArray):
            raise TypeError(
                f"all_reduce takes an NDArray, got {type(x).__name__}. "
                "In gradient(), unwrap with .realize_cached_data() and rewrap with Tensor.make_const()."
            )
        local = x.numpy().astype(np.float32)          # .numpy() handles non-compact views

        parts = [None] * world_size
        parts[rank] = local
        for dst in range(world_size):
            if dst != rank:
                queues[rank][dst].put(local)
        for src in range(world_size):
            if src != rank:
                parts[src] = queues[src][rank].get(timeout=COMM_TIMEOUT)

        total = parts[0].copy()                        # sum in rank order -> bit-identical on all ranks
        for p in parts[1:]:
            total = total + p
        return nd.array(total, device=x.device)       # always a NEW array on x's device

    mock.all_reduce = all_reduce
    return mock


# --------------------------------------------------------------------------- per-rank tests

def worker(rank, world_size, queues, result_queue):
    log, state = [], {"ok": True}

    def section(title):
        log.append(f"  -- {title}")

    def fail(msg):
        state["ok"] = False
        log.append(f"  ❌ {msg}")

    def check(name, actual, expected):
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        if actual.shape != expected.shape:
            return fail(f"{name}: shape {actual.shape} != expected {expected.shape}")
        diff = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
        if np.allclose(actual, expected, rtol=RTOL, atol=ATOL):
            log.append(f"  ✅ {name} (max diff {diff:.1e})")
        else:
            fail(f"{name}: max diff {diff:.1e}")

    def check_true(name, cond, detail=""):
        if cond:
            log.append(f"  ✅ {name}")
        else:
            fail(f"{name} {detail}")

    try:
        sys.modules["needle.parallel.distributed"] = build_mock_dist(rank, world_size, queues)

        import needle as ndl
        import needle.nn as nn
        import needle.backend_ndarray as nd
        from needle.autograd import Tensor
        from needle.parallel.layers import ColumnParallelLinear, RowParallelLinear, TPMLP

        dist = sys.modules["needle.parallel.distributed"]
        shard = H // world_size
        sl = slice(rank * shard, (rank + 1) * shard)

        def same_on_all_ranks(arr):
            """True on this rank iff all_reduce(arr) == N * arr. If that holds on every
            rank, every rank's arr equals the mean, i.e. they are all identical.
            Collective: every rank must call it the same number of times."""
            a = np.asarray(arr, dtype=np.float32)
            total = dist.all_reduce(nd.array(a)).numpy()
            return np.allclose(total, world_size * a, rtol=RTOL, atol=ATOL)

        def set_param(p, arr):
            p.data = Tensor(np.asarray(arr, dtype=np.float32).reshape(p.shape))

        def dense_pair(rs):
            """Two nn.Linear layers (D->H, H->D) with weights from RandomState rs."""
            l1, l2 = nn.Linear(D, H), nn.Linear(H, D)
            set_param(l1.weight, rs.randn(D, H) * 0.3)
            set_param(l1.bias, rs.randn(H) * 0.1)
            set_param(l2.weight, rs.randn(H, D) * 0.3)
            set_param(l2.bias, rs.randn(D) * 0.1)
            return l1, l2

        def dense_block(l1, l2, x):
            return l2(nn.ReLU()(l1(x)))

        def check_block_grads(prefix, tp, l1, l2):
            check(f"{prefix} col W grad = dense[:, shard]", tp.col_linear.weight.grad.numpy(), l1.weight.grad.numpy()[:, sl])
            check(f"{prefix} col b grad = dense[shard]", tp.col_linear.bias.grad.numpy().reshape(-1), l1.bias.grad.numpy().reshape(-1)[sl])
            check(f"{prefix} row W grad = dense[shard, :]", tp.row_linear.weight.grad.numpy(), l2.weight.grad.numpy()[sl, :])
            check(f"{prefix} row b grad = dense (full)", tp.row_linear.bias.grad.numpy().reshape(-1), l2.bias.grad.numpy().reshape(-1))

        # ---- 1. all_reduce contract ------------------------------------------------
        section("1. all_reduce contract")
        x = nd.array(np.full((3, 4), rank + 1, dtype=np.float32))
        out = dist.all_reduce(x)
        check("sums across ranks", out.numpy(), np.full((3, 4), world_size * (world_size + 1) / 2))
        check_true("returns a new array", out is not x and out._handle is not x._handle)
        check_true("keeps the device", out.device == x.device)
        check("input left unchanged", x.numpy(), np.full((3, 4), rank + 1))
        noncompact = x[:, 1:3]
        check_true("input is non-compact (test setup)", not noncompact.is_compact())
        check("non-compact input", dist.all_reduce(noncompact).numpy(), np.full((3, 2), world_size * (world_size + 1) / 2))
        try:
            dist.all_reduce(Tensor(np.zeros((1,), dtype=np.float32)))
            fail("Tensor input should be rejected")
        except TypeError:
            log.append("  ✅ rejects Tensor input")

        # ---- 2. one block: forward + backward --------------------------------------
        section("2. one TPMLP block")
        rs = np.random.RandomState(0)
        l1, l2 = dense_pair(rs)
        X_np, T_np = rs.randn(B, D).astype(np.float32), rs.randn(B, D).astype(np.float32)
        tp = TPMLP.from_dense(l1, l2)
        check_true("col W is a shard", tp.col_linear.weight.shape == (D, shard), f"got {tp.col_linear.weight.shape}")
        check_true("row W is a shard", tp.row_linear.weight.shape == (shard, D), f"got {tp.row_linear.weight.shape}")

        Xd, Xt, T = Tensor(X_np), Tensor(X_np), Tensor(T_np, requires_grad=False)
        yd, yt = dense_block(l1, l2, Xd), tp(Xt)
        check("forward output", yt.numpy(), yd.numpy())
        ld, lt = ((yd - T) ** 2).sum(), ((yt - T) ** 2).sum()
        ld.backward()
        lt.backward()
        check("loss", lt.numpy().reshape(-1), ld.numpy().reshape(-1))
        check_block_grads("", tp, l1, l2)
        check("dX (needs f's backward all-reduce)", Xt.grad.numpy(), Xd.grad.numpy())

        # ---- 3. two stacked blocks ---------------------------------------------------
        section("3. two stacked blocks (g then f)")
        rs = np.random.RandomState(1)
        a1, a2 = dense_pair(rs)
        b1, b2 = dense_pair(rs)
        X_np, T_np = rs.randn(B, D).astype(np.float32), rs.randn(B, D).astype(np.float32)
        tpa, tpb = TPMLP.from_dense(a1, a2), TPMLP.from_dense(b1, b2)

        Xd, Xt, T = Tensor(X_np), Tensor(X_np), Tensor(T_np, requires_grad=False)
        yd = dense_block(b1, b2, dense_block(a1, a2, Xd))
        yt = tpb(tpa(Xt))
        check("forward output", yt.numpy(), yd.numpy())
        ld, lt = ((yd - T) ** 2).sum(), ((yt - T) ** 2).sum()
        ld.backward()
        lt.backward()
        check("loss", lt.numpy().reshape(-1), ld.numpy().reshape(-1))
        check_block_grads("block 1", tpa, a1, a2)
        check_block_grads("block 2", tpb, b1, b2)
        check("dX", Xt.grad.numpy(), Xd.grad.numpy())

        # ---- 4. K Adam steps -----------------------------------------------------------
        section(f"4. {K_STEPS} Adam steps")
        rs = np.random.RandomState(2)
        l1, l2 = dense_pair(rs)
        tp = TPMLP.from_dense(l1, l2)
        opt_d = ndl.optim.Adam(l1.parameters() + l2.parameters(), lr=LR)
        opt_t = ndl.optim.Adam(tp.parameters(), lr=LR)
        losses_d, losses_t = [], []
        for _ in range(K_STEPS):
            X_np, T_np = rs.randn(B, D).astype(np.float32), rs.randn(B, D).astype(np.float32)
            X, T = Tensor(X_np, requires_grad=False), Tensor(T_np, requires_grad=False)
            for model_fn, opt, losses in ((lambda x: dense_block(l1, l2, x), opt_d, losses_d),
                                          (tp, opt_t, losses_t)):
                opt.reset_grad()
                loss = ((model_fn(X) - T) ** 2).sum()
                loss.backward()
                opt.step()
                losses.append(float(loss.numpy().reshape(-1)[0]))
        check("loss at every step", losses_t, losses_d)
        check("col W after training = dense[:, shard]", tp.col_linear.weight.numpy(), l1.weight.numpy()[:, sl])
        check("col b after training = dense[shard]", tp.col_linear.bias.numpy().reshape(-1), l1.bias.numpy().reshape(-1)[sl])
        check("row W after training = dense[shard, :]", tp.row_linear.weight.numpy(), l2.weight.numpy()[sl, :])
        check("row b after training = dense", tp.row_linear.bias.numpy().reshape(-1), l2.bias.numpy().reshape(-1))
        check_true("row b identical on every rank after training", same_on_all_ranks(tp.row_linear.bias.numpy()))

        # ---- 5. direct construction (no from_dense) --------------------------------------
        section("5. direct init")
        d_in, h_big = 64, 256                       # big enough for the bound check to be meaningful
        np.random.seed(123)                          # same seed on every rank
        col = ColumnParallelLinear(d_in, h_big)
        row = RowParallelLinear(h_big, d_in)
        cw, rw = np.abs(col.weight.numpy()), np.abs(row.weight.numpy())
        col_bound, row_bound = math.sqrt(6 / d_in), math.sqrt(6 / h_big)
        check_true("col W within full-fan_in bound", cw.max() <= col_bound and cw.max() > 0.8 * col_bound,
                   f"max {cw.max():.4f}, bound {col_bound:.4f}")
        check_true("row W within full-fan_in bound (fan_in = h, not h/N)", rw.max() <= row_bound and rw.max() > 0.8 * row_bound,
                   f"max {rw.max():.4f}, bound {row_bound:.4f}")
        check_true("row b identical on every rank", same_on_all_ranks(row.bias.numpy()))
        col_same = same_on_all_ranks(col.weight.numpy())
        if world_size > 1 and col_same:
            # Known open issue: with one shared seed every rank draws the same shard, so paired
            # hidden units never diverge. Harmless for benchmarks; matters for training from
            # direct init. Turn this into a failure once the seeding fix lands.
            log.append("  ⚠️  WARN column shards are identical across ranks (shared seed -> symmetric hidden units)")
        else:
            log.append("  ✅ column shards differ across ranks" if world_size > 1 else "  ✅ (single rank: shard uniqueness n/a)")

    except Exception:
        state["ok"] = False
        log.append("  ❌ exception:\n" + traceback.format_exc())

    result_queue.put((rank, state["ok"], log))


# --------------------------------------------------------------------------- driver

def run(world_size):
    ctx = mp.get_context("spawn")
    queues = [[ctx.Queue() if s != d else None for d in range(world_size)] for s in range(world_size)]
    result_queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, world_size, queues, result_queue)) for r in range(world_size)]
    for p in procs:
        p.start()

    print(f"=== world size {world_size}  (B={B}, d={D}, h={H}, rtol={RTOL}, atol={ATOL}) ===")
    results = {}
    try:
        for _ in range(world_size):
            rank, ok, log = result_queue.get(timeout=COMM_TIMEOUT * 3)
            results[rank] = (ok, log)
    except queue.Empty:
        print("❌ Timed out waiting for workers (a rank crashed or skipped a collective -> deadlock).")
        for p in procs:
            p.terminate()
        return False
    finally:
        for p in procs:
            p.join(timeout=10)

    all_ok = len(results) == world_size
    for rank in sorted(results):
        ok, log = results[rank]
        all_ok &= ok
        # Print rank 0 in full; other ranks only if they failed (keeps output short).
        if rank == 0 or not ok:
            print(f"[rank {rank}] {'PASSED' if ok else 'FAILED'}")
            print("\n".join(log))
    others = [r for r in sorted(results) if r != 0 and results[r][0]]
    if others:
        print(f"[ranks {others}] PASSED")
    return all_ok


def main():
    if H % math.lcm(*DEFAULT_WORLD_SIZES) != 0:
        raise ValueError("H must be divisible by every tested world size")
    sizes = [int(a) for a in sys.argv[1:]] or list(DEFAULT_WORLD_SIZES)
    outcome = {n: run(n) for n in sizes}
    print("\n" + "  ".join(f"N={n}: {'PASS' if ok else 'FAIL'}" for n, ok in outcome.items()))
    sys.exit(0 if all(outcome.values()) else 1)


if __name__ == "__main__":
    main()
