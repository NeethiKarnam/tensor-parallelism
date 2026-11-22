"""Show where each shard of a tensor-parallel MLP lives, and what it holds.

Builds a tiny dense MLP (same seed on every rank), shards it with TPMLP.from_dense,
and prints, rank by rank:
  - the machine and device each rank runs on
  - that rank's column shard of W1 and row shard of W2 (small enough to read)
  - its partial output before the all-reduce, and the summed output after it
  - whether everything matches the dense model

Run from needle-tp/:
    NEEDLE_BACKEND=nd python apps/tp_shard_demo.py                      # 1 rank
    NEEDLE_BACKEND=nd mpirun -np 2 python apps/tp_shard_demo.py         # 2 ranks, one machine
    # one rank per GHC machine, each on its own GPU (see docs/GHC_SETUP.md):
    mpirun -np 2 --host ghc29,ghc32 --mca plm_rsh_args -oBatchMode=yes -x NEEDLE_BACKEND=nd \
        ~/private/tp-venv/bin/python apps/tp_shard_demo.py --device cuda
    # the integer example from tests_parallel/test_forward.py, checked against its hand-computed answers:
    ... apps/tp_shard_demo.py --device cuda --example handcheck
"""
import argparse
import os
import socket
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

import numpy as np
import needle as ndl
from needle import nn
import needle.parallel.distributed as dist
from needle.parallel.layers import TPMLP

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
parser.add_argument("--d", type=int, default=4, help="model width")
parser.add_argument("--h", type=int, default=8, help="hidden width (divisible by the number of ranks)")
parser.add_argument("--batch", type=int, default=2)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--example", choices=["random", "handcheck"], default="random",
                    help="handcheck: the integer example from tests_parallel/test_forward.py "
                         "(d=2, h=4, no bias), with its hand-computed partials and output")
args = parser.parse_args()

# The hand-checkable example from tests_parallel/test_forward.py (no bias).
HAND_X = np.array([[1, 2], [-1, 1]], dtype=np.float32)
HAND_W1 = np.array([[1, 0, -1, 2], [0, 1, 1, -1]], dtype=np.float32)
HAND_W2 = np.array([[3, 0], [0, 1], [1, 2], [2, -1]], dtype=np.float32)
HAND_PARTIALS_2_RANKS = [np.array([[3, 2], [0, 1]], dtype=np.float32),
                         np.array([[1, 2], [2, 4]], dtype=np.float32)]
HAND_OUT = np.array([[4, 4], [2, 5]], dtype=np.float32)
if args.example == "handcheck":
    args.d, args.h = HAND_W1.shape

dist.init()
rank, world = dist.get_rank(), dist.get_world_size()
host = socket.gethostname().split(".")[0]
device = ndl.cuda() if args.device == "cuda" else ndl.cpu()
if args.h % world:
    sys.exit(f"--h {args.h} must be divisible by the number of ranks ({world})")


def show(name, a):
    text = np.array2string(np.asarray(a), precision=3, suppress_small=True, max_line_width=100)
    return f"{name}, shape {np.asarray(a).shape}:\n  " + text.replace("\n", "\n  ")


def in_rank_order(text):
    """Print one block per rank, rank 0 first (barriers keep the blocks from interleaving)."""
    for r in range(world):
        dist.barrier()
        if r == rank:
            print(text, flush=True)
    dist.barrier()


# Every rank builds the identical dense model and input (same seed, or the fixed example).
np.random.seed(args.seed)
if args.example == "handcheck":
    lin1, lin2 = nn.Linear(args.d, args.h, bias=False), nn.Linear(args.h, args.d, bias=False)
    lin1.weight.data = ndl.Tensor(HAND_W1)
    lin2.weight.data = ndl.Tensor(HAND_W2)
    X = HAND_X
else:
    lin1, lin2 = nn.Linear(args.d, args.h), nn.Linear(args.h, args.d)
    X = np.random.randn(args.batch, args.d).astype("float32")
W1, W2 = lin1.weight.numpy(), lin2.weight.numpy()
b1 = lin1.bias.numpy().reshape(1, -1) if lin1.bias is not None else np.zeros((1, args.h), "float32")
b2 = lin2.bias.numpy().reshape(1, -1) if lin2.bias is not None else np.zeros((1, args.d), "float32")
dense_out = np.maximum(X @ W1 + b1, 0) @ W2 + b2

tp = TPMLP.from_dense(lin1, lin2, device=device)
tp_out = tp(ndl.Tensor(X, device=device)).numpy()

# What this rank holds (read back from its device)
k = args.h // world
cols = slice(rank * k, (rank + 1) * k)
W1_r, W2_r = tp.col_linear.weight.numpy(), tp.row_linear.weight.numpy()
b1_r = tp.col_linear.bias.numpy() if tp.col_linear.bias is not None else b1[:, cols]
partial = np.maximum(X @ W1_r + b1_r, 0) @ W2_r      # this rank's term, before g sums them
stats = dist.get_comm_stats()

ok_shards = np.array_equal(W1_r, W1[:, cols]) and np.array_equal(W2_r, W2[cols, :])
err = float(np.abs(tp_out - dense_out).max())
ok = ok_shards and err < 1e-5

# The hand example also has known answers: the output, and each partial when N = 2.
hand_lines = []
if args.example == "handcheck":
    out_ok = np.array_equal(tp_out, HAND_OUT)
    ok = ok and out_ok
    if world == 2:
        expected = HAND_PARTIALS_2_RANKS[rank]
        part_ok = np.array_equal(partial, expected)
        ok = ok and part_ok
        hand_lines.append(show("expected partial (from test_forward.py)", expected))
        hand_lines.append(f"partial matches the hand-computed one: {'yes' if part_ok else 'NO'}")

if rank == 0:
    print(f"=== Tensor-parallel MLP: d={args.d}, h={args.h}, {world} rank(s), device {args.device}, "
          f"example {args.example} ===")
    print("Every rank built the same dense model:")
    print(show("dense W1 (d x h)", W1))
    print(show("dense W2 (h x d)", W2))
    print(show("input X", X))
    print(flush=True)

in_rank_order("\n".join([
    f"--- rank {rank} of {world} on {host}, device {device!r} ---",
    show(f"column shard of W1 = W1[:, {cols.start}:{cols.stop}]", W1_r),
    show(f"row shard of W2    = W2[{cols.start}:{cols.stop}, :]", W2_r),
    f"shards match the dense slices: {'yes' if ok_shards else 'NO'}",
    show("partial output relu(X W1_r + b1_r) W2_r (before all-reduce)", partial),
    *hand_lines,
    f"all-reduces on this rank: {stats['calls']} ({stats['bytes']} bytes sent)",
    "",
]))

if rank == 0:
    if lin2.bias is not None:
        print(show("b2 (full on every rank, added once after the all-reduce)", b2))
        print(show("after all-reduce + b2 (TPMLP output) = sum of partials + b2", tp_out))
    else:
        print(show("after all-reduce (TPMLP output) = sum of partials", tp_out))
    print(show("dense model output", dense_out))
    if args.example == "handcheck":
        print(show("hand-computed output (from test_forward.py)", HAND_OUT))
        print(f"TP output matches it exactly: {'yes' if np.array_equal(tp_out, HAND_OUT) else 'NO'}")
    print(f"max |TP - dense| = {err:.1e}  ->  {'PASS' if ok else 'FAIL'}", flush=True)

ok_all = dist.all_reduce(np.array([0.0 if ok else 1.0], dtype=np.float32))[0] == 0
sys.exit(0 if ok_all else 1)
