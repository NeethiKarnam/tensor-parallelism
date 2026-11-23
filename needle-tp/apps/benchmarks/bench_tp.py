"""Benchmarks for Needle tensor parallelism: one script, one suite per invocation.

Run from needle-tp/, one `mpirun` per world size N (see benchmarks/run_benchmarks.sh):

    mpirun -np 2 python benchmarks/bench_tp.py headline    --device cuda   # memory, step time, comm split, utilization
    mpirun -np 2 python benchmarks/bench_tp.py scaling     --device cuda   # weak scaling: h = 4096 * N
    mpirun -np 2 python benchmarks/bench_tp.py correctness --device cuda   # max |difference| from the dense model
    mpirun -np 2 python benchmarks/bench_tp.py allreduce   --device cuda   # latency / bandwidth vs. message size

Rank 0 writes JSON to --out-dir (default results/tp_bench/). benchmarks/make_tables.py turns those files
into the report tables. Without mpirun the script runs as world size 1.

Measurement notes
  * Step time: wall clock around zero_grad -> forward -> loss -> backward -> Adam step, ended by a device sync.
  * Comm split: `needle.parallel.distributed.all_reduce` already times the GPU->host copy, the MPI Allreduce and
    the host->GPU copy. On cuda, the GPU->host copy also waits for queued kernels, which would be charged to
    "comm". We therefore wrap all_reduce with a device sync first, so queued compute is counted as compute.
  * Peak memory: a background thread samples NVML (this process's memory, or the whole device as a fallback)
    every ~1 ms. It can only under-estimate the true peak. The CUDA context is excluded ("model" peak) and
    also reported ("total" peak, which decides whether the run fits).
  * Utilization: NVML's utilization.gpu averaged over the timed steps (same number nvidia-smi shows).
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suite", choices=["headline", "scaling", "correctness", "allreduce"])
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--d", type=int, default=4096, help="model width d")
    p.add_argument("--h", type=int, default=16384, help="hidden width h (headline suite)")
    p.add_argument("--h-base", type=int, default=4096, help="scaling suite uses h = h_base * N")
    p.add_argument("--blocks", type=int, default=4, help="number of stacked TPMLP blocks")
    p.add_argument("--batch", type=int, default=128, help="batch size B")
    p.add_argument("--steps", type=int, default=10, help="timed steps")
    p.add_argument("--warmup", type=int, default=3, help="untimed warm-up steps")
    p.add_argument("--peak-tflops", type=float, default=10.0, help="fp32 peak of the GPU (RTX 2080 ~ 10)")
    p.add_argument("--out-dir", default="results/tp_bench")
    p.add_argument("--tag", default="", help="extra suffix for the output file name")
    # correctness suite sizes (small on purpose: float32 error scales with the loss magnitude)
    p.add_argument("--c-d", type=int, default=64)
    p.add_argument("--c-h", type=int, default=256)
    p.add_argument("--c-batch", type=int, default=16)
    p.add_argument("--tol", type=float, default=1e-5)
    return p.parse_args()


ARGS = parse_args()

# One GPU per rank on a multi-GPU machine (Kaggle 2xT4): BENCH_GPUS_PER_NODE=2 picks GPU = local rank.
if ARGS.device == "cuda" and os.environ.get("BENCH_GPUS_PER_NODE") and "CUDA_VISIBLE_DEVICES" not in os.environ:
    _local = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("MPI_LOCALRANKID", "0")))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_local % int(os.environ["BENCH_GPUS_PER_NODE"]))

os.environ.setdefault("NEEDLE_BACKEND", "nd")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

import numpy as np  # noqa: E402

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"

sys.path.insert(0, str(PYTHON_DIR))

import needle as ndl  # noqa: E402
import needle.backend_ndarray as nd  # noqa: E402
import needle.nn as nn  # noqa: E402
import needle.parallel.distributed as dist  # noqa: E402
from needle.autograd import Tensor  # noqa: E402
from needle.parallel.layers import TPMLP  # noqa: E402

dist.init()
RANK, N = dist.get_rank(), dist.get_world_size()


def log(*a):
    if RANK == 0:
        print(*a, flush=True)


# --------------------------------------------------------------------------- helpers

def allgather(obj):
    c = dist._comm
    return c.allgather(obj) if (c is not None and N > 1) else [obj]


def get_device():
    if ARGS.device == "cpu":
        return ndl.cpu()
    dev = ndl.cuda()
    if not dev.enabled():
        sys.exit("cuda requested but the CUDA backend is not available (was it built? see docs/GHC_SETUP.md)")
    return dev


def make_sync(dev):
    """Block until all queued kernels finished. A D2H copy on the default stream waits for earlier work."""
    probe = nd.array(np.zeros(1, dtype=np.float32), device=dev)
    return lambda: probe.numpy()


def is_oom(e):
    s = (str(e) or "").lower()
    return isinstance(e, MemoryError) or "out of memory" in s or "bad_alloc" in s


class Monitor:
    """Samples this process's GPU memory (and GPU utilization) in a background thread."""

    def __init__(self, device_kind):
        self.kind = device_kind
        self.mode = "none"
        self.peak = 0
        self.base = 0
        self.total = None
        self.gpu_name = "cpu"
        self.util = []
        self.running = False
        self._nvml = None
        self._handle = None
        if device_kind == "cuda":
            self._init_nvml()

    def _init_nvml(self):
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        idx = int(vis) if vis.isdigit() else 0
        try:
            import pynvml  # pip install nvidia-ml-py

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            name = pynvml.nvmlDeviceGetName(self._handle)
            self.gpu_name = name.decode() if isinstance(name, bytes) else name
            self.total = pynvml.nvmlDeviceGetMemoryInfo(self._handle).total
            self.mode = "nvml-device"
        except Exception:
            self._nvml = None
            self.mode = "nvidia-smi"
            self._smi_idx = idx
            out = self._smi(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits", f"-i={idx}"])
            if out:
                name, tot = [s.strip() for s in out.splitlines()[0].split(",")]
                self.gpu_name, self.total = name, int(float(tot) * 2**20)

    @staticmethod
    def _smi(args):
        try:
            return subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True, timeout=5).stdout
        except Exception:
            return ""

    def read_mem(self):
        if self.kind == "cpu":
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
            return 0
        if self._nvml is not None:
            try:
                for p in self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle):
                    if p.pid == os.getpid() and p.usedGpuMemory is not None:
                        self.mode = "nvml-process"
                        return int(p.usedGpuMemory)
            except Exception:
                pass
            return int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)  # whole device (shared GPU!)
        out = self._smi(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
        for line in out.splitlines():
            parts = [s.strip() for s in line.split(",")]
            if len(parts) == 2 and parts[0] == str(os.getpid()):
                return int(float(parts[1]) * 2**20)
        out = self._smi(["--query-gpu=memory.used", "--format=csv,noheader,nounits", f"-i={self._smi_idx}"])
        return int(float(out.split()[0]) * 2**20) if out.strip() else 0

    def read_util(self):
        if self._nvml is not None:
            try:
                return float(self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
            except Exception:
                return None
        out = self._smi(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", f"-i={self._smi_idx}"])
        return float(out.split()[0]) if out.strip() else None

    def _loop(self):
        next_util = 0.0
        fast = self.kind == "cpu" or self._nvml is not None
        while self.running:
            m = self.read_mem()
            self.peak = max(self.peak, m)
            now = time.perf_counter()
            if self.kind == "cuda" and now >= next_util:
                u = self.read_util()
                if u is not None:
                    self.util.append((now, u))
                next_util = now + 0.1
            time.sleep(0.001 if fast else 0.25)

    def start(self):
        sys.setswitchinterval(0.0005)  # let the sampler thread run between Needle ops
        self.running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self.running = False
        self._t.join(timeout=2)

    def util_mean(self, t0, t1):
        v = [u for t, u in self.util if t0 <= t <= t1]
        return statistics.mean(v) if v else None


def write_json(name, obj):
    if RANK != 0:
        return
    os.makedirs(ARGS.out_dir, exist_ok=True)
    path = os.path.join(ARGS.out_dir, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"wrote {path}", flush=True)


def common_meta(mon):
    hosts = allgather(socket.gethostname())
    return {
        "world_size": N,
        "device": ARGS.device,
        "gpu": mon.gpu_name if ARGS.device == "cuda" else "cpu",
        "hosts": hosts,
        "machines": len(set(hosts)),
        "command": " ".join(sys.argv),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# --------------------------------------------------------------------------- model + training benchmark

class TPStack(nn.Module):
    def __init__(self, d, h, blocks, device):
        super().__init__()
        self.blocks = [TPMLP(d, h, device=device) for _ in range(blocks)]

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def predicted_bytes(d, h, blocks, B, n):
    """16 B/param (weights + grads + Adam m + v) + activations. Needle's backward stores .grad on every
    graph node, so saved activations are counted twice. Per block per rank ~ 3*B*h/n + 3*B*d floats."""
    params = blocks * (2 * d * h // n + h // n + d)
    state = 16 * params
    acts = 2 * 4 * blocks * (3 * B * h // n + 3 * B * d)
    return params, state, acts


def train_bench(dev, d, h, blocks, B, out_name):
    mon = Monitor(ARGS.device)
    sync = make_sync(dev)  # creates the CUDA context before the baseline reading
    mon.base = mon.read_mem()
    mon.start()
    params_n, pred_state, pred_acts = predicted_bytes(d, h, blocks, B, N)
    rec = {
        **common_meta(mon),
        "config": {"d": d, "h": h, "blocks": blocks, "batch": B, "steps": ARGS.steps, "warmup": ARGS.warmup,
                   "optimizer": "adam"},
        "predicted_params_per_rank": params_n,
        "predicted_state_bytes": pred_state,
        "predicted_activation_bytes": pred_acts,
        "mem_mode": mon.mode,
        "gpu_total_bytes": mon.total,
        "status": "ok",
    }
    steps_ms, d2h, ar, h2d, calls, nbytes, losses = [], [], [], [], [], [], []
    t_win0 = t_win1 = None
    orig_all_reduce = dist.all_reduce

    def synced_all_reduce(x):
        sync()  # charge queued kernels to compute, not to the GPU->host copy
        return orig_all_reduce(x)

    dist.all_reduce = synced_all_reduce
    try:
        np.random.seed(0)
        model = TPStack(d, h, blocks, dev)
        params = model.parameters()
        rec["params_per_rank"] = int(sum(int(np.prod(p.shape)) for p in params))
        opt = ndl.optim.Adam(params, lr=1e-3)
        X = Tensor(np.random.randn(B, d).astype(np.float32), device=dev, requires_grad=False)
        T = Tensor(np.random.randn(B, d).astype(np.float32), device=dev, requires_grad=False)
        scale = 1.0 / (B * d)

        def one_step():
            # Locals die on return, so the graph (activations + their grads) is freed before the next step.
            opt.reset_grad()
            loss = ((model(X) - T) ** 2).sum() * scale
            loss.backward()
            opt.step()
            return loss.realize_cached_data()

        for i in range(ARGS.warmup + ARGS.steps):
            sync()
            dist.barrier()
            dist.reset_comm_stats()
            t0 = time.perf_counter()
            loss = one_step()
            sync()
            t1 = time.perf_counter()
            if i == ARGS.warmup:
                t_win0 = t0
            t_win1 = t1
            if i >= ARGS.warmup:
                st = dist.get_comm_stats()
                steps_ms.append((t1 - t0) * 1e3)
                d2h.append(st["d2h_s"] * 1e3)
                ar.append(st["allreduce_s"] * 1e3)
                h2d.append(st["h2d_s"] * 1e3)
                calls.append(st["calls"])
                nbytes.append(st["bytes"])
            losses.append(float(loss.numpy().reshape(-1)[0]))
            log(f"  [{out_name}] step {i - ARGS.warmup:>3} {(t1 - t0) * 1e3:9.1f} ms  loss {losses[-1]:.4f}")
    except Exception as e:
        if not is_oom(e):
            raise
        rec["status"] = "OOM"
        rec["error"] = str(e)
        print(f"[rank {RANK}] OOM: {e}", flush=True)
    finally:
        dist.all_reduce = orig_all_reduce
        mon.stop()

    rec["peak_total_bytes"] = mon.peak
    rec["peak_model_bytes"] = max(0, mon.peak - mon.base)
    rec["baseline_bytes"] = mon.base
    if rec["status"] == "ok":
        comm = [a + b + c for a, b, c in zip(d2h, ar, h2d)]
        med = statistics.median
        rec.update({
            "step_ms": med(steps_ms),
            "compute_ms": med([s - c for s, c in zip(steps_ms, comm)]),
            "d2h_ms": med(d2h), "allreduce_ms": med(ar), "h2d_ms": med(h2d),
            "comm_ms": med(comm),
            "calls_per_step": med(calls),
            "bytes_per_call": (med(nbytes) / med(calls)) if med(calls) else 0,
            "step_ms_all": steps_ms,
            "final_loss": losses[-1],
            "gpu_util_pct": mon.util_mean(t_win0, t_win1) if t_win0 else None,
            "flops_per_gpu_step": 12 * B * d * h * blocks / N,
        })
    if rec["status"] == "OOM" and N > 1:
        # Other ranks are inside a collective, so a gather would deadlock: record this rank and stop the job.
        os.makedirs(ARGS.out_dir, exist_ok=True)
        with open(os.path.join(ARGS.out_dir, out_name.replace(".json", f"_rank{RANK}_oom.json")), "w") as fh:
            json.dump(rec, fh, indent=1)
        sys.stdout.flush()
        dist._comm.Abort(2)
    per_rank = allgather(rec)
    if RANK == 0:
        agg = dict(per_rank[0])
        agg["per_rank"] = [{k: r.get(k) for k in ("status", "peak_total_bytes", "peak_model_bytes", "step_ms",
                                                     "comm_ms", "gpu_util_pct")} for r in per_rank]
        if any(r["status"] == "OOM" for r in per_rank):
            agg["status"] = "OOM"
        else:
            for k in ("step_ms", "compute_ms", "d2h_ms", "allreduce_ms", "h2d_ms", "comm_ms"):
                agg[k] = max(r[k] for r in per_rank)  # slowest rank
            utils = [r["gpu_util_pct"] for r in per_rank if r.get("gpu_util_pct") is not None]
            agg["gpu_util_pct"] = statistics.mean(utils) if utils else None
        agg["peak_total_bytes"] = max(r["peak_total_bytes"] for r in per_rank)
        agg["peak_model_bytes"] = max(r["peak_model_bytes"] for r in per_rank)
        write_json(out_name, agg)
    return rec


# --------------------------------------------------------------------------- correctness

def correctness(dev):
    d, h, B = ARGS.c_d, ARGS.c_h, ARGS.c_batch
    assert h % 8 == 0 and h % N == 0, "c-h must be divisible by every world size"
    shard = h // N
    sl = slice(RANK * shard, (RANK + 1) * shard)
    relu = nn.ReLU()

    def set_param(p, arr):
        p.data = Tensor(np.asarray(arr, dtype=np.float32).reshape(p.shape), device=dev)

    def dense_pair(rs):
        l1, l2 = nn.Linear(d, h, device=dev), nn.Linear(h, d, device=dev)
        set_param(l1.weight, rs.randn(d, h) / np.sqrt(d))  # keeps activations and the loss O(1)
        set_param(l1.bias, rs.randn(h) * 0.1)
        set_param(l2.weight, rs.randn(h, d) / np.sqrt(h))
        set_param(l2.bias, rs.randn(d) * 0.1)
        return l1, l2

    def dense_fwd(pairs, x):
        for l1, l2 in pairs:
            x = l2(relu(l1(x)))
        return x

    def tp_fwd(blocks, x):
        for b in blocks:
            x = b(x)
        return x

    def mse(y, T):
        return ((y - T) ** 2).sum() * (1.0 / (B * d))

    def adiff(a, b):
        return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))

    def grad_diffs(tp, pair):
        """(abs diff, ref magnitude) over the 4 parameter grads of one block vs. the dense slices."""
        l1, l2 = pair
        pairs = [
            (tp.col_linear.weight.grad.numpy(), l1.weight.grad.numpy()[:, sl]),
            (tp.col_linear.bias.grad.numpy().reshape(-1), l1.bias.grad.numpy().reshape(-1)[sl]),
            (tp.row_linear.weight.grad.numpy(), l2.weight.grad.numpy()[sl, :]),
            (tp.row_linear.bias.grad.numpy().reshape(-1), l2.bias.grad.numpy().reshape(-1)),
        ]
        return max(adiff(a, b) for a, b in pairs), max(float(np.abs(b).max()) for _, b in pairs)

    def one_case(n_blocks, seed):
        rs = np.random.RandomState(seed)
        pairs = [dense_pair(rs) for _ in range(n_blocks)]
        X_np, T_np = rs.randn(B, d).astype(np.float32), rs.randn(B, d).astype(np.float32)
        tps = [TPMLP.from_dense(l1, l2, device=dev) for l1, l2 in pairs]
        Xd, Xt, T = Tensor(X_np, device=dev), Tensor(X_np, device=dev), Tensor(T_np, device=dev, requires_grad=False)
        ld, lt = mse(dense_fwd(pairs, Xd), T), mse(tp_fwd(tps, Xt), T)
        ld.backward()
        lt.backward()
        gd = [grad_diffs(tp, pr) for tp, pr in zip(tps, pairs)]
        ref = max(g[1] for g in gd)
        gmax = max(g[0] for g in gd)
        return {"dloss": adiff(lt.numpy(), ld.numpy()), "dgrad": gmax, "dgrad_rel": gmax / ref,
                "ddx": adiff(Xt.grad.numpy(), Xd.grad.numpy())}

    def adam_case(steps=5):
        rs = np.random.RandomState(2)
        l1, l2 = dense_pair(rs)
        tp = TPMLP.from_dense(l1, l2, device=dev)
        od = ndl.optim.Adam(l1.parameters() + l2.parameters(), lr=1e-2)
        ot = ndl.optim.Adam(tp.parameters(), lr=1e-2)
        worst = {"dloss": 0.0, "dgrad": 0.0, "dgrad_rel": 0.0}
        for _ in range(steps):
            X = Tensor(rs.randn(B, d).astype(np.float32), device=dev, requires_grad=False)
            T = Tensor(rs.randn(B, d).astype(np.float32), device=dev, requires_grad=False)
            od.reset_grad()
            ot.reset_grad()
            ld = mse(dense_fwd([(l1, l2)], X), T)
            lt = mse(tp_fwd([tp], X), T)
            ld.backward()
            lt.backward()
            g, ref = grad_diffs(tp, (l1, l2))
            worst["dloss"] = max(worst["dloss"], adiff(lt.numpy(), ld.numpy()))
            worst["dgrad"] = max(worst["dgrad"], g)
            worst["dgrad_rel"] = max(worst["dgrad_rel"], g / ref)
            od.step()
            ot.step()
        return worst

    results = {"one_block": one_case(1, 0), "two_blocks": one_case(2, 1), "adam5": adam_case()}
    for r in results.values():  # worst case over all ranks
        for k in list(r):
            r[k] = max(allgather(r[k]))
        r["pass"] = all(r[k] <= ARGS.tol for k in ("dloss", "dgrad", "ddx") if k in r)
    mon = Monitor(ARGS.device)
    rec = {**common_meta(mon), "config": {"d": d, "h": h, "batch": B, "tol": ARGS.tol,
                                           "loss": "mean squared error (sum / (B*d))"}, "results": results}
    log(json.dumps(results, indent=1))
    write_json(f"correctness_N{N}{ARGS.tag}.json", rec)
    if not all(r["pass"] for r in results.values()):
        sys.exit(1)


# --------------------------------------------------------------------------- all-reduce latency

def allreduce_latency(dev):
    mon = Monitor(ARGS.device)
    sync = make_sync(dev)
    if N == 1:
        log("allreduce suite needs at least 2 ranks; nothing to do")
        return
    sizes = [("4 B", 1), ("1 KB", 256), ("64 KB", 16 * 1024), ("1 MB", 256 * 1024), ("16 MB", 4 * 1024 * 1024),
             ("B x d x 4", ARGS.batch * ARGS.d)]
    comm, MPI = dist._comm, dist._MPI
    rows = []
    for label, n in sizes:
        iters = 200 if n * 4 <= 64 * 1024 else (100 if n * 4 <= 2**20 else 30)
        host = np.ones(n, dtype=np.float32)
        out = np.empty_like(host)
        dev_x = nd.array(host, device=dev)
        t_mpi, t_e2e = [], []
        for i in range(10 + iters):
            comm.Barrier()
            t0 = time.perf_counter()
            comm.Allreduce(host, out, op=MPI.SUM)
            t1 = time.perf_counter()
            if i >= 10:
                t_mpi.append(t1 - t0)
        for i in range(10 + iters):
            sync()
            comm.Barrier()
            t0 = time.perf_counter()
            dist.all_reduce(dev_x)
            t1 = time.perf_counter()
            if i >= 10:
                t_e2e.append(t1 - t0)
        row = {"label": label, "floats": n, "bytes": 4 * n,
               "mpi_us": max(allgather(statistics.median(t_mpi) * 1e6)),
               "e2e_us": max(allgather(statistics.median(t_e2e) * 1e6))}
        for k in ("mpi", "e2e"):
            row[f"{k}_algbw_gbs"] = row["bytes"] / (row[f"{k}_us"] * 1e-6) / 1e9
            row[f"{k}_busbw_gbs"] = row[f"{k}_algbw_gbs"] * 2 * (N - 1) / N
        rows.append(row)
        log(f"  {label:>10}  MPI {row['mpi_us']:10.1f} us  {row['mpi_algbw_gbs']:7.3f} GB/s"
            f"   device e2e {row['e2e_us']:10.1f} us  {row['e2e_algbw_gbs']:7.3f} GB/s")
    rec = {**common_meta(mon), "config": {"batch": ARGS.batch, "d": ARGS.d}, "rows": rows}
    write_json(f"allreduce_N{N}{ARGS.tag}.json", rec)


# --------------------------------------------------------------------------- main

def main():
    dev = get_device()
    log(f"suite={ARGS.suite} N={N} device={ARGS.device} host={socket.gethostname()}")
    try:
        if ARGS.suite == "headline":
            name = f"headline_b{ARGS.blocks}_N{N}{ARGS.tag}.json"
            train_bench(dev, ARGS.d, ARGS.h, ARGS.blocks, ARGS.batch, name)
        elif ARGS.suite == "scaling":
            h = ARGS.h_base * N
            train_bench(dev, ARGS.d, h, ARGS.blocks, ARGS.batch, f"scaling_N{N}{ARGS.tag}.json")
        elif ARGS.suite == "correctness":
            correctness(dev)
        else:
            allreduce_latency(dev)
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        if N > 1:
            dist._comm.Abort(1)
        sys.exit(1)


if __name__ == "__main__":
    main()
