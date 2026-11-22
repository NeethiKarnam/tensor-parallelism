"""Char-level language model on Tiny Shakespeare, dense or tensor-parallel.

Model:  one-hot (context*65) -> ColumnParallelLinear(context*65, h) -> ReLU
        -> RowParallelLinear(h, 65) -> logits -> SoftmaxLoss
The dense model is the same two nn.Linear layers. Both start from the same weights:
every rank builds the dense layers on CPU with --seed, and TP mode slices them with
from_dense. Every rank draws the same batches (--data-seed), so with correct TP layers
the two loss curves match up to float rounding.

Run from needle-tp/:
    NEEDLE_BACKEND=nd python apps/train_lm.py --mode dense --out results/train_dense.json
    NEEDLE_BACKEND=nd mpirun -np 2 python apps/train_lm.py --mode tp --out results/train_tp.json
    # GHC, one rank per machine (docs/GHC_SETUP.md):
    mpirun -np 2 --host ghcA,ghcB --mca plm_rsh_args -oBatchMode=yes -x NEEDLE_BACKEND=nd \
        ~/private/tp-venv/bin/python apps/train_lm.py --mode tp --device cuda --out results/train_tp.json

Rank 0 writes the results JSON (format in report/final_report.ipynb) and prints progress.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import numpy as np
import needle as ndl
from needle import nn
import needle.parallel.distributed as dist
from needle.parallel.layers import ColumnParallelLinear, RowParallelLinear

from char_data import CharDataset, download

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--mode", choices=["dense", "tp"], default="tp")
p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
p.add_argument("--steps", type=int, default=500)
p.add_argument("--hidden", type=int, default=512)
p.add_argument("--context", type=int, default=8)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=0, help="weight init seed (same on every rank)")
p.add_argument("--data-seed", type=int, default=1, help="batch sampling seed (same on every rank)")
p.add_argument("--log-every", type=int, default=10)
p.add_argument("--eval-every", type=int, default=100)
p.add_argument("--eval-batches", type=int, default=10)
p.add_argument("--prompt", default="ROMEO:\n")
p.add_argument("--sample-len", type=int, default=300)
p.add_argument("--out", default=None, help="results JSON path (written by rank 0)")
args = p.parse_args()

dist.init()
rank, world = dist.get_rank(), dist.get_world_size()
device = ndl.cuda() if args.device == "cuda" else ndl.cpu()
if args.mode == "dense" and world > 1:
    sys.exit("--mode dense runs on one process; drop mpirun")
if args.hidden % world:
    sys.exit(f"--hidden {args.hidden} must be divisible by the number of ranks ({world})")


def log(*a):
    if rank == 0:
        print(*a, flush=True)


# ---- data: rank 0 downloads, the rest wait (AFS is shared across GHC machines) ----
if rank == 0:
    download()
dist.barrier()
data = CharDataset(context=args.context)
log(f"Tiny Shakespeare: {len(data.train):,} train / {len(data.val):,} val chars, vocab {data.vocab}")

# ---- model: dense layers built on CPU from one seed, then moved or sharded ----
np.random.seed(args.seed)
lin1 = nn.Linear(data.input_dim, args.hidden)
lin2 = nn.Linear(args.hidden, data.vocab)

if args.mode == "tp":
    layers = [ColumnParallelLinear.from_dense(lin1, device=device), nn.ReLU(),
              RowParallelLinear.from_dense(lin2, device=device)]
else:
    def to_device(lin):
        new = nn.Linear(lin.in_features, lin.out_features, device=device)
        new.weight.data = ndl.Tensor(lin.weight.numpy(), device=device)
        new.bias.data = ndl.Tensor(lin.bias.numpy(), device=device)
        return new
    layers = [to_device(lin1), nn.ReLU(), to_device(lin2)]
model = nn.Sequential(*layers)
loss_fn = nn.SoftmaxLoss()
opt = ndl.optim.Adam(model.parameters(), lr=args.lr)

n_local = sum(int(np.prod(q.shape)) for q in model.parameters())
n_full = sum(int(np.prod(q.shape)) for q in lin1.parameters() + lin2.parameters())
log(f"mode {args.mode}, {world} rank(s), device {args.device}: "
    f"{n_local:,} params on this rank of {n_full:,} total")


def forward(X_np):
    return model(ndl.Tensor(X_np, device=device, requires_grad=False))


def evaluate(rng):
    """Mean val loss over a fixed set of batches (same on every rank)."""
    model.eval()
    tot = 0.0
    for _ in range(args.eval_batches):
        X, y = data.batch(rng, args.batch, "val")
        tot += float(loss_fn(forward(X), ndl.Tensor(y, device=device, requires_grad=False)).numpy().reshape(-1)[0])
    model.train()
    return tot / args.eval_batches


def sample(prompt, n):
    """Greedy decoding. Every rank runs it (each forward has an all-reduce); rank 0 prints."""
    ids = list(data.encode(prompt))
    pad = [data.stoi["\n"]] * max(0, args.context - len(ids))
    ids = pad + ids
    for _ in range(n):
        window = np.array(ids[-args.context:])[None, :]
        logits = forward(data.one_hot(window)).numpy()
        ids.append(int(logits[0].argmax()))
    return data.decode(ids[len(pad):])


# ---- training ----
rng = np.random.RandomState(args.data_seed)
val_rng_seed = args.data_seed + 1000
res = {"mode": args.mode, "world_size": world, "hardware": None,
       "config": {"context": args.context, "hidden": args.hidden, "lr": args.lr, "seed": args.seed,
                  "data_seed": args.data_seed, "batch": args.batch, "steps": args.steps},
       "params_per_rank": n_local, "step": [], "loss": [], "val_step": [], "val_loss": []}

dist.reset_comm_stats()
step_times = []
t_start = time.time()
for step in range(args.steps + 1):
    if step % args.eval_every == 0 or step == args.steps:
        v = evaluate(np.random.RandomState(val_rng_seed))  # same val batches every time
        res["val_step"].append(step)
        res["val_loss"].append(v)
        log(f"step {step:5d}  val loss {v:.4f}")
    if step == args.steps:
        break
    X, y = data.batch(rng, args.batch)
    t0 = time.time()
    opt.reset_grad()
    loss = loss_fn(forward(X), ndl.Tensor(y, device=device, requires_grad=False))
    loss.backward()
    opt.step()
    lv = float(loss.numpy().reshape(-1)[0])  # also syncs the device before stopping the clock
    step_times.append(time.time() - t0)
    if step % args.log_every == 0:
        res["step"].append(step)
        res["loss"].append(lv)
        log(f"step {step:5d}  loss {lv:.4f}  ({step_times[-1] * 1e3:.0f} ms/step)")

res["train_time_s"] = time.time() - t_start
res["step_ms_median"] = float(np.median(step_times[5:] or step_times)) * 1e3
res["comm"] = dist.get_comm_stats()
res["sample_prompt"] = args.prompt
res["sample"] = sample(args.prompt, args.sample_len)
log(f"\n--- greedy sample ---\n{res['sample']}\n---------------------")

if rank == 0:
    host = socket.gethostname().split(".")[0]
    gpu = "CPU"
    if args.device == "cuda":
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        except Exception:
            gpu = "GPU"
    res["hardware"] = f"{host} (rank 0), {world}x {gpu}"
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        log(f"wrote {args.out}")
