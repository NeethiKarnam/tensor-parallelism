"""
@file test_allreduce.py
@brief Correctness checks for all_reduce in needle.parallel.distributed

test_allreduce runs ten checks on all_reduce, each on every rank. It runs with
or without mpirun, from needle-tp/:
    NEEDLE_BACKEND=nd python tests_parallel/test_allreduce.py --device cpu
    NEEDLE_BACKEND=nd mpirun -np 2 python tests_parallel/test_allreduce.py --device cuda
Only rank 0 prints. Every rank exits nonzero if any check failed on any rank.

@author Neethi Karnam <nkarnam@andrew.cmu.edu>
"""
import argparse
import os
import sys
import traceback

os.environ.setdefault("NEEDLE_BACKEND", "nd")
if os.environ["NEEDLE_BACKEND"] != "nd":
    print("These tests require NEEDLE_BACKEND=nd.")
    sys.exit(1)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))  # the needle package lives here

import numpy as np
import needle as ndl
from needle.backend_ndarray import NDArray
from needle.parallel import distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
args = parser.parse_args()

dist.init()
RANK, WORLD = dist.get_rank(), dist.get_world_size()

device = ndl.cpu() if args.device == "cpu" else ndl.cuda()
if not device.enabled():
    if RANK == 0:
        print(f"Device {args.device} is not available (backend not built).")
    sys.exit(1)

failures = []  # names of checks that failed on this rank


def log(msg):
    """
    @brief This function prints a message from rank 0 only.
    @param msg text to print
    """
    if RANK == 0:
        print(msg, flush=True)


def check(name, fn):
    """
    @brief This function runs one check and records PASS or FAIL.
    @param name name to report
    @param fn check to run; raises on failure
    """
    # Collectives inside fn run before its asserts, so a failing assert on one
    # rank doesn't leave the other ranks waiting.
    try:
        fn()
    except Exception:
        failures.append(name)
        log(f"FAIL  {name}")
        if RANK == 0:
            traceback.print_exc(file=sys.stdout)
    else:
        log(f"PASS  {name}")


def rank_array(r, shape, seed=0):
    """
    @brief This function builds the array that rank r contributes.
    @param r rank number
    @param shape array shape
    @param seed per-check seed
    @return float32 numpy array; any rank can rebuild any other rank's array
    """
    return np.random.default_rng(1000 * seed + r).standard_normal(shape).astype(np.float32)


def expected_sum(shape, seed=0, view=lambda a: a):
    """
    @brief This function computes the correct all_reduce result locally.
    @param shape array shape
    @param seed per-check seed
    @param view function applied to each rank's array before summing
    @return sum over all ranks of view(rank_array(r))
    """
    return sum(view(rank_array(r, shape, seed)) for r in range(WORLD))


def nd(a):
    """
    @brief This function copies a numpy array onto the test device.
    @param a numpy array
    @return NDArray on the selected device
    """
    return NDArray(a, device=device)


def close(got, want, msg=""):
    """
    @brief This function asserts two arrays match within 1e-5.
    @param got actual array
    @param want expected array
    @param msg message shown on failure
    """
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5, err_msg=msg)


def test_constant():
    """
    @brief This function checks that rank r's value r + 1 sums to N(N+1)/2.
    """
    x = device.full((3, 5), float(RANK + 1))
    out = dist.all_reduce(x).numpy()
    close(out, np.full((3, 5), WORLD * (WORLD + 1) / 2, dtype=np.float32))


def test_random():
    """
    @brief This function checks the sum of different random arrays from each rank.
    """
    shape = (4, 6)
    out = dist.all_reduce(nd(rank_array(RANK, shape, seed=1))).numpy()
    close(out, expected_sum(shape, seed=1))


def test_column_slice():
    """
    @brief This function checks a non-compact column slice as input.
    """
    shape = (4, 8)
    x = nd(rank_array(RANK, shape, seed=2))[:, 2:6]
    assert not x.is_compact(), "test setup: slice should be a non-compact view"
    out = dist.all_reduce(x).numpy()
    close(out, expected_sum(shape, seed=2, view=lambda a: a[:, 2:6]))


def test_transpose():
    """
    @brief This function checks a transposed (permuted) array as input.
    """
    shape = (3, 5)
    x = nd(rank_array(RANK, shape, seed=3)).permute((1, 0))
    assert not x.is_compact(), "test setup: permute should be a non-compact view"
    out = dist.all_reduce(x).numpy()
    close(out, expected_sum(shape, seed=3, view=lambda a: a.T))


def test_broadcast_view():
    """
    @brief This function checks a broadcast array (stride 0) as input.
    """
    row = rank_array(RANK, (1, 4), seed=4)
    x = nd(row).broadcast_to((3, 4))
    assert 0 in x.strides, "test setup: broadcast should have a stride-0 axis"
    out = dist.all_reduce(x).numpy()
    close(out, np.broadcast_to(expected_sum((1, 4), seed=4), (3, 4)))


def test_large():
    """
    @brief This function checks a 4 MB array, large enough that MPI splits it up.
    """
    shape = (512, 2048)
    out = dist.all_reduce(nd(rank_array(RANK, shape, seed=5))).numpy()
    close(out, expected_sum(shape, seed=5))


def test_interface():
    """
    @brief This function checks the return type, shape, device, and that x is unchanged.
    """
    before = rank_array(RANK, (2, 3), seed=6)
    x = nd(before)
    out = dist.all_reduce(x)
    assert isinstance(out, NDArray), f"returned {type(out).__name__}, not NDArray"
    assert out.shape == x.shape, f"shape {out.shape} != {x.shape}"
    assert out.device == x.device, f"device {out.device} != {x.device}"
    np.testing.assert_array_equal(x.numpy(), before, err_msg="input was modified")
    if WORLD == 1:
        assert out is x, "single-process all_reduce should return its input unchanged"
    else:
        assert out is not x, "all_reduce should return a new array"


def test_ranks_agree():
    """
    @brief This function checks that every rank gets a bit-identical result.
    """
    out = dist.all_reduce(nd(rank_array(RANK, (8, 8), seed=7))).numpy()
    if WORLD > 1:
        from mpi4py import MPI

        results = MPI.COMM_WORLD.allgather(out.tobytes())
        assert all(r == results[0] for r in results), "ranks got different results"


def test_rejects_tensor():
    """
    @brief This function checks that passing a Tensor raises TypeError.
    """
    t = ndl.Tensor(np.ones((2, 2), dtype=np.float32), device=device)
    try:
        dist.all_reduce(t)
    except TypeError:
        return
    raise AssertionError("all_reduce accepted a Tensor; it should raise TypeError")


def test_stats():
    """
    @brief This function checks that the timing counters count calls and bytes, and reset.
    """
    dist.reset_comm_stats()
    dist.all_reduce(device.full((10, 10), 1.0))
    s = dist.get_comm_stats()
    if WORLD == 1:
        assert s["calls"] == 0, f"single-process mode should not count calls: {s}"
    else:
        assert s["calls"] == 1 and s["bytes"] == 400, f"unexpected stats: {s}"
        assert min(s["d2h_s"], s["allreduce_s"], s["h2d_s"]) >= 0, f"negative time: {s}"
    dist.reset_comm_stats()
    assert dist.get_comm_stats()["calls"] == 0, "reset_comm_stats did not reset"


def main():
    """
    @brief This function runs every check and combines results from all ranks.
    @return 0 if every check passed on every rank, otherwise 1
    """
    log(f"=== all_reduce: world size {WORLD}, device {args.device} ===")
    for test in (
        test_constant,
        test_random,
        test_column_slice,
        test_transpose,
        test_broadcast_view,
        test_large,
        test_interface,
        test_ranks_agree,
        test_rejects_tensor,
        test_stats,
    ):
        check(test.__name__, test)
    dist.barrier()

    # Collect every rank's failures so rank 0 reports them and all ranks agree on the exit code.
    if WORLD > 1:
        from mpi4py import MPI

        per_rank = MPI.COMM_WORLD.allgather(failures)
    else:
        per_rank = [failures]
    total = sum(len(f) for f in per_rank)
    for r, names in enumerate(per_rank):
        if r != 0 and names:
            log(f"FAIL  on rank {r}: {', '.join(names)}")
    log(f"{10 - len(failures)}/10 checks passed on rank 0; {total} failure(s) across {WORLD} rank(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
