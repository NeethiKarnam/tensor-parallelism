"""
@file bench_tp.py
@brief Training benchmark for the tensor parallel MLP

bench_tp trains a stack of TPMLP blocks on random data with Adam and records,
for one configuration: peak memory per rank (measured and predicted), step
time and throughput, the split of each step into compute and communication,
and achieved FLOP/s. Weak scaling efficiency comes from comparing runs (N = 1
at h, N = 2 at 2h). Run each configuration as its own process, from needle-tp/:
    python apps/bench_tp.py --device cuda --hidden 16384
    mpirun -np 2 --host ghc29,ghc32 ... python apps/bench_tp.py --device cuda --hidden 16384
Rank 0 appends one row to --out. A single-process run that runs out of GPU
memory records a row with oom set to true instead of crashing.

@author Neethi Karnam <nkarnam@andrew.cmu.edu>
"""
import argparse
import json
import os
import resource
import socket
import subprocess
import sys
import threading
import time

os.environ.setdefault("NEEDLE_BACKEND", "nd")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

import numpy as np
import needle as ndl
from needle import nn
from needle.parallel import distributed as dist

BYTES_PER_PARAM = 16  # fp32 weight + gradient + Adam's two moment estimates


def parse_args():
    """
    @brief This function reads the command line flags.
    @return parsed arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--d", type=int, default=4096, help="model width")
    p.add_argument("--hidden", type=int, default=16384, help="hidden size h of each block")
    p.add_argument("--blocks", type=int, default=1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=10, help="timed steps")
    p.add_argument("--warmup", type=int, default=3, help="untimed steps before timing")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--peak-tflops", type=float, default=10.0, help="GPU fp32 peak (RTX 2080 is about 10)")
    p.add_argument("--mem-interval", type=float, default=0.02, help="seconds between GPU memory samples")
    p.add_argument("--out", default=os.path.join(ROOT, "results", "bench.json"))
    return p.parse_args()


class StackedTPMLP(nn.Module):
    """
    @brief Stack of TPMLP blocks, each d -> h -> d.
    """

    def __init__(self, d, h, blocks, device):
        """
        @brief This function builds the blocks.
        @param d model width
        @param h hidden size of each block
        @param blocks number of blocks
        @param device needle device
        """
        super().__init__()
        from needle.parallel.layers import TPMLP

        self.blocks = [TPMLP(d, h, bias=True, device=device) for _ in range(blocks)]

    def forward(self, x):
        """
        @brief This function runs the blocks in order.
        @param x input Tensor of shape (batch, d)
        @return output Tensor of shape (batch, d)
        """
        for block in self.blocks:
            x = block(x)
        return x


class PeakMemory:
    """
    @brief Background sampler of this process's peak GPU memory.
    """

    def __init__(self, device_name, interval):
        """
        @brief This function sets up the sampler.
        @param device_name "cpu" or "cuda"
        @param interval seconds between samples
        """
        self.device_name = device_name
        self.interval = interval
        self.pid = os.getpid()
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = None
        self._nvml = None
        if device_name == "cuda":
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml = pynvml
            except Exception:
                self._nvml = None  # fall back to nvidia-smi

    def current_mb(self):
        """
        @brief This function reads this process's memory use now.
        @return megabytes of GPU memory (cuda) or peak resident memory (cpu)
        """
        if self.device_name != "cuda":
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return rss / 2**20 if sys.platform == "darwin" else rss / 2**10  # bytes on macOS, KB on Linux
        if self._nvml is not None:
            total = 0
            for i in range(self._nvml.nvmlDeviceGetCount()):
                handle = self._nvml.nvmlDeviceGetHandleByIndex(i)
                for proc in self._nvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    if proc.pid == self.pid and proc.usedGpuMemory:
                        total += proc.usedGpuMemory
            return total / 2**20
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        total = 0.0
        for line in out.stdout.strip().splitlines():
            parts = [s.strip() for s in line.split(",")]
            if len(parts) == 2 and parts[0] == str(self.pid):
                total += float(parts[1])
        return total

    def _run(self):
        """
        @brief This function samples until stopped, keeping the maximum.
        """
        while not self._stop.is_set():
            try:
                self.peak_mb = max(self.peak_mb, self.current_mb())
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        """
        @brief This function starts sampling in a background thread.
        """
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """
        @brief This function stops sampling and takes one last sample.
        @return peak megabytes seen
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        try:
            self.peak_mb = max(self.peak_mb, self.current_mb())
        except Exception:
            pass
        return self.peak_mb


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


def params_per_rank(d, h, blocks, world):
    """
    @brief This function counts the parameters each rank holds.
    @param d model width
    @param h hidden size
    @param blocks number of blocks
    @param world number of ranks
    @return column weight and bias shards, row weight shard, and the full row bias, per block
    """
    return blocks * (d * h // world + h // world + h // world * d + d)


def flops_per_gpu(batch, d, h, blocks, world):
    """
    @brief This function counts the FLOPs one GPU does per training step.
    @param batch batch size
    @param d model width
    @param h hidden size
    @param blocks number of blocks
    @param world number of ranks
    @return 12 * batch * d * h per block (forward plus backward), divided by world
    """
    return 12 * batch * d * h * blocks / world


def is_oom(err):
    """
    @brief This function checks whether an exception is an out-of-memory error.
    @param err exception raised during the run
    @return True for CUDA out of memory or MemoryError
    """
    return isinstance(err, MemoryError) or "out of memory" in str(err).lower()


def run_steps(args, device):
    """
    @brief This function builds the model and times the training steps.
    @param args parsed arguments
    @param device needle device
    @return (list of per-step timing dicts, actual parameter count, final loss)
    """
    np.random.seed(args.seed)  # same weights on every rank for replicated parameters
    model = StackedTPMLP(args.d, args.hidden, args.blocks, device)
    opt = ndl.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(int(np.prod(p.shape)) for p in model.parameters())

    data = np.random.default_rng(args.seed + 1)  # same batch on every rank
    X = ndl.Tensor(data.standard_normal((args.batch, args.d)).astype(np.float32), device=device)
    Y = ndl.Tensor(data.standard_normal((args.batch, args.d)).astype(np.float32), device=device)

    steps, loss_value = [], float("nan")
    for i in range(args.warmup + args.steps):
        dist.barrier()
        dist.reset_comm_stats()
        t = time.perf_counter()
        opt.reset_grad()
        loss = ((model(X) - Y) ** 2).sum() / args.batch
        loss.backward()
        opt.step()
        loss_value = float(loss.numpy().reshape(-1)[0])  # waits for all queued GPU work
        step_s = time.perf_counter() - t
        if i >= args.warmup:
            steps.append(dict(dist.get_comm_stats(), step_s=step_s))
    return steps, n_params, loss_value


def summarize(steps, args, world):
    """
    @brief This function turns per-step timings into the reported numbers.
    @param steps per-step timing dicts from run_steps
    @param args parsed arguments
    @param world number of ranks
    @return dict of median times (ms), throughput and FLOP/s
    """
    def med(key):
        return float(np.median([s[key] for s in steps]))

    comm = [s["d2h_s"] + s["allreduce_s"] + s["h2d_s"] for s in steps]
    step_s, comm_s = med("step_s"), float(np.median(comm))
    compute_s = float(np.median([s["step_s"] - c for s, c in zip(steps, comm)]))
    flops = flops_per_gpu(args.batch, args.d, args.hidden, args.blocks, world)
    tflops = flops / step_s / 1e12
    return {
        "step_ms": step_s * 1e3,
        "compute_ms": compute_s * 1e3,  # includes sync_ms
        "sync_ms": med("sync_s") * 1e3,
        "d2h_ms": med("d2h_s") * 1e3,
        "allreduce_ms": med("allreduce_s") * 1e3,
        "h2d_ms": med("h2d_s") * 1e3,
        "host_copy_ms": (med("d2h_s") + med("h2d_s")) * 1e3,
        "comm_ms": comm_s * 1e3,
        "comm_pct": 100.0 * comm_s / step_s,
        "allreduce_calls_per_step": int(steps[0]["calls"]),
        "allreduce_bytes_per_step": int(steps[0]["bytes"]),
        "samples_per_s": args.batch / step_s,
        "flops_per_step_per_gpu": flops,
        "tflops_per_gpu": tflops,
        "tflops_per_gpu_compute_only": flops / compute_s / 1e12 if compute_s > 0 else 0.0,
        "tflops_total": tflops * world,  # weak scaling: total(N=2, 2h) / (2 * total(N=1, h))
        "pct_of_peak": 100.0 * tflops / args.peak_tflops,
    }


def write_row(path, row):
    """
    @brief This function appends one result row to a JSON list file.
    @param path results file
    @param row dict to append
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = json.load(f)
    rows.append(row)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def main():
    """
    @brief This function runs one benchmark configuration and writes the result from rank 0.
    @return process exit code
    """
    args = parse_args()
    dist.init()  # before building layers, so they shard by the real world size
    rank, world = dist.get_rank(), dist.get_world_size()
    device = ndl.cpu() if args.device == "cpu" else ndl.cuda()
    if not device.enabled():
        if rank == 0:
            print(f"Device {args.device} is not available (backend not built).")
        return 1

    mem = PeakMemory(args.device, args.mem_interval)
    if args.device == "cuda":
        device.empty((1,)).numpy()  # start the CUDA context so it counts in the baseline
    baseline_mb = mem.current_mb()
    mem.start()

    oom, steps, n_params, loss_value = False, [], 0, float("nan")
    try:
        steps, n_params, loss_value = run_steps(args, device)
    except Exception as err:
        # With 2+ ranks the other rank would hang waiting for this one, so only a
        # single-process run records out-of-memory as a result.
        if world > 1 or not is_oom(err):
            raise
        oom = True
    peak_mb = mem.stop()

    hosts = gather(socket.gethostname())
    peaks = gather(peak_mb)
    if rank != 0:
        return 0

    predicted_params = params_per_rank(args.d, args.hidden, args.blocks, world)
    row = {
        "world_size": world,
        "hidden": args.hidden,
        "d": args.d,
        "blocks": args.blocks,
        "batch": args.batch,
        "params_per_rank": n_params or predicted_params,
        "predicted_gb": predicted_params * BYTES_PER_PARAM / 1e9,
        "peak_mem_mb": max(peaks),
        "peak_mem_mb_per_rank": peaks,
        "baseline_mem_mb": baseline_mb,
        "oom": oom,
        "final_loss": loss_value,
        "device": args.device,
        "gpu": gpu_name(args.device),
        "hosts": hosts,
        "two_machines": len(set(hosts)) > 1,
        "steps": args.steps,
        "warmup": args.warmup,
        "command": " ".join(sys.argv),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if not oom:
        row.update(summarize(steps, args, world))

    write_row(args.out, row)
    if oom:
        print(f"N={world} h={args.hidden} blocks={args.blocks}: OUT OF MEMORY "
              f"(predicted {row['predicted_gb']:.2f} GB per rank)")
    else:
        print(
            f"N={world} d={args.d} h={args.hidden} blocks={args.blocks} B={args.batch} on {sorted(set(hosts))}\n"
            f"  memory   peak {row['peak_mem_mb']:.0f} MB/rank (baseline {baseline_mb:.0f}), "
            f"predicted {row['predicted_gb']:.2f} GB\n"
            f"  step     {row['step_ms']:.1f} ms, {row['samples_per_s']:.1f} samples/s\n"
            f"  split    compute {row['compute_ms']:.1f} (sync {row['sync_ms']:.1f}) | d2h {row['d2h_ms']:.1f} | "
            f"allreduce {row['allreduce_ms']:.1f} | h2d {row['h2d_ms']:.1f} ms | comm {row['comm_pct']:.0f}%\n"
            f"  flops    {row['tflops_per_gpu']:.3f} TFLOPS/GPU ({row['pct_of_peak']:.1f}% of {args.peak_tflops} peak)"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
