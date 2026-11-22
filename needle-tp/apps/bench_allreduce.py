"""
@file bench_allreduce.py
@brief All-reduce latency and bandwidth benchmark

bench_allreduce times needle.parallel.distributed.all_reduce on its own, with
no model, over a range of message sizes. A barrier before every timed call
keeps waiting for other ranks out of the numbers, so they show the cost of the
host copies and the network. Latency is the time of the smallest message and
bandwidth is the best bytes per second measured. Needs 2 or more ranks.
Run from needle-tp/:
    mpirun -np 2 python apps/bench_allreduce.py --device cpu
    mpirun -np 2 --host ghc29,ghc32 ... python apps/bench_allreduce.py --device cuda
Rank 0 prints a table and writes the results to --out.

@author Neethi Karnam <nkarnam@andrew.cmu.edu>
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time

os.environ.setdefault("NEEDLE_BACKEND", "nd")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

import numpy as np
import needle as ndl
from needle.backend_ndarray import NDArray
from needle.parallel import distributed as dist

DEFAULT_SIZES = [4 << 10, 16 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20]


def parse_args():
    """
    @brief This function reads the command line flags.
    @return parsed arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES, help="message sizes in bytes")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--out", default=os.path.join(ROOT, "results", "allreduce.json"))
    return p.parse_args()


def gather(obj):
    """
    @brief This function collects one Python object from every rank onto rank 0.
    @param obj this rank's value
    @return list indexed by rank on rank 0, None on other ranks
    """
    if dist.get_world_size() == 1:
        return [obj]
    from mpi4py import MPI

    return MPI.COMM_WORLD.gather(obj, root=0)


def gpu_name(device_name):
    """
    @brief This function returns the GPU model, or "cpu".
    @param device_name "cpu" or "cuda"
    @return GPU model name from nvidia-smi
    """
    if device_name != "cuda":
        return "cpu"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def latency_bandwidth(rows):
    """
    @brief This function estimates latency and bandwidth from the sweep.
    @param rows results from bench_size, sorted by size
    @return (latency in ms from the smallest message, peak bandwidth in GB/s)
    """
    return rows[0]["allreduce_ms"], max(r["bandwidth_gbs"] for r in rows)


def bench_size(nbytes, device, warmup, iters):
    """
    @brief This function times all_reduce for one message size.
    @param nbytes message size in bytes
    @param device needle device
    @param warmup untimed calls before measuring
    @param iters timed calls
    @return dict of median times in ms for each phase, and bandwidth
    """
    n = max(nbytes // 4, 1)
    rank = dist.get_rank()
    x = NDArray(np.full(n, rank + 1, dtype=np.float32), device=device)

    for _ in range(warmup):
        dist.all_reduce(x)

    phases = {k: [] for k in ("sync_s", "d2h_s", "allreduce_s", "h2d_s", "total_s")}
    for _ in range(iters):
        dist.barrier()  # start together, so allreduce_s is not waiting for a slow rank
        dist.reset_comm_stats()
        t = time.perf_counter()
        dist.all_reduce(x)
        phases["total_s"].append(time.perf_counter() - t)
        s = dist.get_comm_stats()
        for k in ("sync_s", "d2h_s", "allreduce_s", "h2d_s"):
            phases[k].append(s[k])

    med = {k: float(np.median(v)) for k, v in phases.items()}
    return {
        "bytes": n * 4,
        "sync_ms": med["sync_s"] * 1e3,
        "d2h_ms": med["d2h_s"] * 1e3,
        "allreduce_ms": med["allreduce_s"] * 1e3,
        "h2d_ms": med["h2d_s"] * 1e3,
        "total_ms": med["total_s"] * 1e3,
        # Bytes each rank contributes divided by the time of the MPI call alone.
        "bandwidth_gbs": n * 4 / med["allreduce_s"] / 1e9 if med["allreduce_s"] > 0 else 0.0,
    }


def main():
    """
    @brief This function runs the sweep and writes the results from rank 0.
    @return process exit code
    """
    args = parse_args()
    dist.init()
    rank, world = dist.get_rank(), dist.get_world_size()
    if world < 2:
        print("bench_allreduce needs 2 or more ranks, e.g. mpirun -np 2 python apps/bench_allreduce.py")
        return 1

    device = ndl.cpu() if args.device == "cpu" else ndl.cuda()
    if not device.enabled():
        if rank == 0:
            print(f"Device {args.device} is not available (backend not built).")
        return 1

    hosts = gather(socket.gethostname())
    rows = [bench_size(b, device, args.warmup, args.iters) for b in sorted(args.sizes)]

    if rank != 0:
        return 0
    latency_ms, bandwidth_gbs = latency_bandwidth(rows)
    result = {
        "world_size": world,
        "device": args.device,
        "gpu": gpu_name(args.device),
        "hosts": hosts,
        "two_machines": len(set(hosts)) > 1,
        "command": " ".join(sys.argv),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latency_ms": latency_ms,
        "peak_bandwidth_gbs": bandwidth_gbs,
        "rows": rows,
    }

    print(f"all_reduce, {world} ranks on {sorted(set(hosts))}, device {args.device} ({result['gpu']})")
    print(f"{'bytes':>10} {'sync':>8} {'d2h':>8} {'allred':>8} {'h2d':>8} {'total':>8} {'GB/s':>7}   (ms, medians)")
    for r in rows:
        print(
            f"{r['bytes']:>10} {r['sync_ms']:>8.3f} {r['d2h_ms']:>8.3f} {r['allreduce_ms']:>8.3f} "
            f"{r['h2d_ms']:>8.3f} {r['total_ms']:>8.3f} {r['bandwidth_gbs']:>7.3f}"
        )
    print(f"latency (smallest message) {latency_ms:.3f} ms, peak bandwidth {bandwidth_gbs:.3f} GB/s")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
