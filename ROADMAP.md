# Tensor Parallelism Final Project — Roadmap

_Last updated Nov 21, 2026. Edit this file directly; it replaces the Claude Docs version._

## Summary

Video due **Nov 28** (25% of grade). Final report (Colab notebook + private Drive code link) due **Dec 7** (65%, half prose, half code).

Done: single-process sharded forward pass, and Needle running on GHC GPUs (ops pass forward and gradient checks on cpu and cuda). Still to do: the tensor-parallel code itself (comm backend, f/g ops, parallel layers), the backward pass across processes, the LM demo, and the benchmarks. **Current focus (Phase 1):** everyone on the core TP path until M1; the LM and benchmarks wait for Phase 2. See Interfaces and Milestones. We build and test on CPU with `mpirun`, then run benchmarks and record the demo on GHC.

What the check-in commits us to (Megatron-style tensor parallelism for MLP blocks, in Needle):

- One process per GPU, with a process-group API (rank, world size, init)
- A communication backend (MPI; NCCL is a stretch goal)
- Two autograd ops: `CopyToParallelRegion` (f: identity forward, all-reduce backward) and `ReduceFromParallelRegion` (g: all-reduce forward, identity backward)
- `ColumnParallelLinear` and `RowParallelLinear`
- Milestones: **M1** backward pass across processes · **M2** GPU benchmarks (memory, runtime, communication vs. one GPU) · **M3** LM trained on Tiny Shakespeare across 2 processes with a loss curve matching single-device

## Decisions already made

- **Comm:** MPI only, through `mpi4py`. GPU data goes to the host, through `Allreduce(SUM)`, and back. NCCL, if we get to it, sits behind the same `all_reduce` interface.
- **Backend:** always `NEEDLE_BACKEND=nd`. The HW4 ops call NDArray methods (`permute`, `compact`), so the `np` backend no longer works for backward passes. Each machine runs `make` once.
- **GPU selection:** GHC has one GPU per machine, so a 2-GPU run is one rank on each of two GHC machines (see Compute → Two-machine runs). On a machine with several GPUs (Kaggle backup), each rank sets `CUDA_VISIBLE_DEVICES=<rank>`.
- **LM:** char-level MLP. Last k chars → one-hot in numpy → column-parallel → ReLU → row-parallel → logits (vocab ≈ 65) → `SoftmaxLoss`. No `nn.Embedding`, no HW4 conv/RNN/transformer code, output layer not sharded.
- **M2 headline:** a model that runs out of memory on one GPU trains on two. On GHC (RTX 2080, 8 GB): 4 blocks of d = 4096, h = 16384 (537M params, ≈ 8.6 GB with Adam; ≈ 4.3 GB per GPU on two). On the Kaggle backup (T4, 16 GB): d = 8192, h = 65536 (1.07B params, ≈ 17 GB).
- **Code location:** `python/needle/parallel/`, `apps/`, `tests_parallel/`. Avoid editing homework files unless unavoidable, and say so in the commit.

## Interfaces (agree before splitting up)

These are the points where one person's code calls another's. Write code against these signatures; if one needs to change, tell the team before pushing.

**1. Comm: `needle/parallel/distributed.py`** (Neethi → everyone)

```python
init()                              # call once at the top of each script; calling twice is harmless; never runs on import
get_rank() -> int
get_world_size() -> int
all_reduce(x: NDArray) -> NDArray   # SUM; takes a raw NDArray (not a Tensor), compacts it,
                                    # returns a NEW array on x's device (no in-place update)
barrier()
```

Without `mpirun`, rank = 0, world size = 1, and `all_reduce` returns its input unchanged. Then f and g are identity and every test runs in a single process before MPI works.

**`all_reduce` inside autograd (decided Nov 21):** `all_reduce` is a plain function, **not** a `TensorOp`. f and g in `parallel/ops.py` are the only TensorOps. g calls it directly in `compute` (which already receives NDArrays). f's `gradient` unwraps, reduces, and rewraps:

```python
class CopyToParallelRegion(TensorOp):          # f
    def compute(self, x):
        return x
    def gradient(self, out_grad, node):
        return Tensor.make_const(all_reduce(out_grad.realize_cached_data()))

class ReduceFromParallelRegion(TensorOp):      # g
    def compute(self, x):
        return all_reduce(x)
    def gradient(self, out_grad, node):
        return out_grad
```

Why: a standalone AllReduce TensorOp would have all-reduce as its own gradient. g's backward must be identity (every rank already holds the full gradient), so using that op for g would silently multiply gradients by N. This keeps the communication rule only in f and g, and keeps `all_reduce` free of autograd code. Trade-off: `make_const` cuts the graph, so there are no gradients of f's gradient; we don't need them.

**2. Parallel layers: `needle/parallel/layers.py`** (Anishka → Evelyn, Neethi)

- Weights stay in Needle's `(in, out)` layout with `X @ W`. Column-parallel splits along axis 1, row-parallel along axis 0 (same as `sharding.py`).
- `ColumnParallelLinear(in_features, out_features, bias=True, device=None, dtype="float32")`, and the same arguments for `RowParallelLinear`. Layers read rank and world size from `distributed`; callers don't pass them in.
- Shapes: column output stays sharded at `(B, h/N)` and is never gathered. Row input is already sharded; row output is the full `(B, d)` after g. Row bias is added after g.
- `TPMLP(d, h, bias=True, device=None)`: f → column → ReLU → row → g. This is the block the LM and the benchmark model are built from.

**3. Matching dense weights** (everyone's correctness numbers)

- `ColumnParallelLinear.from_dense(linear)` / `RowParallelLinear.from_dense(linear)` (and `TPMLP.from_dense(lin1, lin2)`) slice an existing `nn.Linear` into this rank's shard.
- Tests and the LM comparison build the dense model after `np.random.seed(S)` (same S on every rank), then shard it with `from_dense`. Data uses its own seed, also the same on every rank.
- float32 everywhere. Tolerance: max absolute difference 1e-5 for loss, gradients, and dX.

**4. Scripts**

- Tests are plain scripts: `mpirun -np 2 python tests_parallel/test_*.py`, and they also run without `mpirun`. Only rank 0 prints; the exit code is nonzero on failure.
- Every script takes `--device cpu|cuda`. Training and benchmark scripts also take `--steps --hidden --out`, and rank 0 writes JSON in the format in `report/final_report.ipynb`.

**5. Git**

- Branches: `initializers`, `comm-backend`, `tp-layers`, `tp-tests`. Merge in that order. One reviewer per PR.
- In `parallel/__init__.py`, each person adds only their own exports. Don't edit `nn/` or other homework files without saying so in the PR.

## Milestones

Work is split into two phases. Phase 1 puts all three of us on the core TP path, because nothing in the LM or the benchmarks can be tested until it works across ranks. The LM and the benchmarks move to Phase 2.

### Phase 1: core TP path, now → M1 (target Nov 22)

| Person | Works on | Done when |
| --- | --- | --- |
| **Neethi** | `distributed.py` (push it early; Anishka builds on it), `tests_parallel/test_allreduce.py`, and the two-machine GHC setup from Compute in your own account (CPU all-reduce across ghc29 + ghc32 already works for Evelyn) | `mpirun -np 2` all-reduce passes on cpu and cuda, including across two GHC machines with one GPU each |
| **Anishka** | `parallel/ops.py` (f, g) and `parallel/layers.py` (`ColumnParallelLinear`, `RowParallelLinear`, `TPMLP`, `from_dense`) | `tests_parallel/test_tp.py` passes at N = 1, 2, 4 |
| **Evelyn** | Initializers (done). Correctness tests in `tests_parallel/test_tp.py`, written from the interfaces above (done; see Status). Next: add an MPI mode to `test_tp.py` so the same checks run on the real `distributed.py` under `mpirun` | `test_tp.py` runs with the mock (N = 1, 2, 4) and under `mpirun -np 2` |

**M1 is done when** `test_tp.py` passes at N = 1, 2, 4 with the mock comm, the same checks pass under `mpirun -np 2` on the real `distributed.py`, and `test_allreduce.py` passes under `mpirun -np 2`.

`test_tp.py` (run from `needle-tp/`: `NEEDLE_BACKEND=nd python tests_parallel/test_tp.py`, or append `2` for N = 2 only) spawns N processes with a strict mock `all_reduce` that really sums across processes. No MPI needed. It checks, on every rank:

1. `all_reduce` contract: sums across ranks; rejects `Tensor`; returns a new array on the same device; input unchanged; non-compact input OK
2. One `TPMLP` block: forward, loss, each shard's gradient = the dense gradient's slice, dX
3. Two stacked blocks (g directly followed by f): same checks
4. 5 Adam steps: loss matches at every step; shards = dense slices after training; row bias identical on every rank
5. Direct construction: shards use the full fan_in; row bias identical across ranks; column shards differ across ranks (**warning only** until the seeding fix; see Known traps)

### Phase 2: after M1 (Nov 23–26)

| Person | Works on |
| --- | --- |
| Evelyn | Tiny Shakespeare loader + LM training, starting from the Adam-steps section of `test_tp.py` → M3 |
| Neethi | Benchmark script + GHC GPU runs → M2 |
| Anishka | Stacked multi-block model for the memory headline, then help whoever is behind and start report sections |

### Dates

| Date | Milestone | Owner |
| --- | --- | --- |
| **Nov 22** | **M1:** forward, backward, and a few training steps across 2 ranks match dense | All |
| Nov 23–24 | LM on Tiny Shakespeare | Evelyn |
| **Nov 24** | **M3:** 2-rank LM loss matches dense | Evelyn |
| Nov 24–26 | GPU benchmark runs on two GHC machines (Kaggle 2 × T4 as backup) | Neethi, Anishka |
| **Nov 26** | **M2:** benchmarks done | Neethi |
| Nov 26–28 | Record + edit video | All |
| **Nov 28** | **Video due** | |
| **Dec 7** | **Final report due** | All |

GPU time is only needed Nov 24–26; everything before that runs on CPU.

**If M1 slips past Nov 22:** drop the 4-block headline model and benchmark a single wide block instead. Don't defer the GHC two-machine check: if it fails we need the Kaggle backup before Nov 24.

## Status

| Piece | State | What's left |
| --- | --- | --- |
| Autograd, ops, SGD, Adam | Done. HW4 `ops_mathematic.py` + fixes to the autograd default device and `init.constant`. `tests_base/test_ops.py` passes 46/46 on GHC (cpu + cuda) | Optional: remove unused HW4 ops (Stack, Split, Flip, Dilate, Conv) |
| CPU/CUDA backends | Done. `src/`, `CMakeLists.txt` (no `-march=native`), `Makefile` in repo; builds on GHC | — |
| Initializers | Done. Only `kaiming_uniform` is kept (the only one used); it takes `shape=` so a shard can use the full layer's fan_in. The unused, broken `xavier_*` / `kaiming_normal` were removed | — |
| Sharding helpers | Done: `split_column`, `split_row` | — |
| Single-process sharded forward | Done: `tests_parallel/test_forward.py` (8/8) | Switch it from `NEEDLE_BACKEND=np` to `nd` |
| `parallel/distributed.py` | Not started (Neethi, Phase 1) | See Interfaces §1. mpi4py `Allreduce(SUM)`, compact before MPI, world size 1 without `mpirun`. Time the all-reduce and host copies. **Settle the path:** `ops.py` / `layers.py` import `needle.distributed` and call `is_initialized()`; Interfaces §1 says `needle/parallel/distributed.py` with no `is_initialized` |
| `parallel/ops.py` (f, g) | Written (Anishka) | **Fix:** `F_Op.gradient` passes a `Tensor` to `all_reduce`; change to `Tensor.make_const(dist.all_reduce(out_grad.realize_cached_data()))` (Interfaces §1). `test_tp.py` fails at N ≥ 2 until then; with this fix it passes at N = 1, 2, 4 |
| `parallel/layers.py` | Written (Anishka); init uses `kaiming_uniform` with the full fan_in | **Open:** seeding for direct construction. With one shared seed every rank draws identical column shards (symmetric hidden units). Fine for benchmarks; for the LM use `from_dense` on CPU-built dense layers, or seed with `seed + rank` for shards |
| Correctness tests | Written (Evelyn): `tests_parallel/test_tp.py`. Replaces `test_mock.py`, `test_tp_correctness.py`, `test_two_processes.py` (removed). Verified to fail on 3 injected bugs (g backward all-reducing, row bias before g, wrong column slice) | Add an MPI mode that uses the real `distributed.py` under `mpirun`. Report notebook cell still points at `test_backward.py`; update it |
| LM demo | Phase 2 (Evelyn) | Tiny Shakespeare loader, one-hot windows, TP MLP, training loop, greedy sampling |
| Benchmarks | Phase 2 (Neethi, Anishka) | Per-rank memory, step time, all-reduce time (incl. host copies) across widths |

**Branches:** `forward-demo` and `build-setup` are merged into `main`. Phase 1 branches: `initializers`, `comm-backend`, `tp-layers`, `tp-tests`.

## Owners

| Person | Owns | First deliverable |
| --- | --- | --- |
| Evelyn | Phase 1: initializers, TP correctness tests (`test_tp.py`). Phase 2: Tiny Shakespeare dataset, char-level LM, loss-curve comparison vs. dense. Also: video, report notebook | Initializers PR merged; tests run with and without `mpirun` |
| Neethi | Phase 1: `needle/parallel/distributed.py` (mpi4py), all-reduce test, multi-machine `mpirun` on GHC. Phase 2: benchmark script (memory, comm timing) and GHC GPU runs | `mpirun -np 2` all-reduce test passes; two-machine `mpirun` works |
| Anishka | Phase 1: f/g ops, column- and row-parallel linear layers, `TPMLP`, `from_dense`. Phase 2: stacked multi-block benchmark model, then help where needed | Layers pass the correctness tests under `mpirun -np 2` |

All three: write the report sections for your own piece, and review each other's PRs before merging to `main`.

## Compute: GHC machines

Develop and run CPU tests on a laptop or Colab (`mpirun -np 2`; on Colab add `--allow-run-as-root --oversubscribe`). Run the 2-GPU benchmarks on two GHC machines (one rank each); Kaggle 2 × T4 is the backup.

### Specs (checked Nov 17 on ghc29)

| Item | Value | What it means for us |
| --- | --- | --- |
| GPUs per machine | 1 × NVIDIA GeForce RTX 2080 | 2 ranks = 2 machines over the network; report comm cost as including the network |
| GPU memory | 8 GB | Sets the "too big for one GPU" model |
| Driver / max CUDA | 550.67 / CUDA 12.4 | Build against CUDA ≤ 12.4 |
| nvcc | CUDA 11.7 at `/usr/local/cuda-11.7/bin`, not on PATH | Needs gcc ≤ 11; system gcc is 13.3, so build with `/usr/bin/gcc-11` |
| MPI | Open MPI 4.1.6 | `mpi4py` builds against it |
| Python | System 3.12.3 | We use Python 3.13 from `uv` in a venv |
| CPU | 8 cores, no AVX-512 | Never build with `-march=native` on another machine and copy the `.so` over |
| Shared lab machine | Others may use the GPU | Check `nvidia-smi` is idle before benchmarking |

### Setup and multi-machine runs: see [docs/GHC_SETUP.md](docs/GHC_SETUP.md)

The guide covers everything from a fresh account: Python 3.13 venv and backend build (`gcc-11`, CUDA 11.7), the `~/.ssh/config` that lets `mpirun` reach other machines, picking machines with idle GPUs, the per-session login, run commands for 2 or 3 ranks, the check scripts (`scripts/check_allreduce.py`, `scripts/check_gpu.py`), and a table of every error we hit and its fix.

In short:

- **Once:** build in `~/private` on **one** machine only (the Makefile wipes `build/` on a hostname change, and the files are shared). Set up `~/.ssh/config` and put `HWLOC_COMPONENTS=-gl` at the top of `~/.bashrc`.
- **Each session:** `ssh ghcNN true` with your password for each remote machine (connections last 4 h). Then:
  `mpirun -np 2 --host ghc29,ghc32 --mca plm_rsh_args "-o BatchMode=yes" -x NEEDLE_BACKEND=nd ~/private/tp-venv/bin/python <script>`
- **Why not Kerberos or SSH keys:** the GHC servers reject Kerberos logins, and SSH-key sessions can't read `~/private` on AFS. SSH connection sharing gets around both.
- **Status (Nov 21):** CPU all-reduce works across ghc29 + ghc32. The 3-machine run (+ ghc41) is not yet confirmed.

- [ ] `scripts/check_gpu.py` across two machines: each rank on its own RTX 2080
- [ ] Each teammate repeats the setup in their own account (connections and `~/private` are per person)
- [ ] Measure all-reduce latency and bandwidth between machines at several message sizes (for the report)

### Backup: Kaggle 2 × T4

Both GPUs are on one machine: `mpirun -np 2`, each rank sets `CUDA_VISIBLE_DEVICES=<rank>`, no SSH, and no network in the comm cost.

- [ ] Kaggle account with GPU enabled (Kaggle may require phone verification), and the "GPU T4 × 2" accelerator selected
- [ ] `nvidia-smi` shows 2 GPUs; `which nvcc mpirun` (install Open MPI with `apt-get` if missing); `pip install mpi4py cmake pybind11`
- [ ] `make` builds both backends; `NEEDLE_BACKEND=nd python tests_base/test_ops.py` passes on cuda
- [ ] `mpirun --allow-run-as-root -np 2` all-reduce test passes, with each rank on its own GPU
- [ ] Note the weekly GPU quota and plan runs to fit it

| Option | GPUs | Cost | Notes |
| --- | --- | --- | --- |
| **GHC, two machines (plan)** | 2 × RTX 2080, 8 GB each | Free | Connection-sharing setup above |
| Kaggle Notebooks | 2 × T4, 16 GB each | Free, weekly quota | Run `!mpirun` from a cell |
| GCP VM | 2 × T4 | Billed per hour | Last resort; stop it between sessions |

## Demo and benchmark plan

Two models: the LM proves training is correct; wide synthetic MLP blocks prove the memory savings. The dataset doesn't affect memory, only model size does. The char-level LM has only ≈ 585 × h params (≈ 38M even at h = 65,536), too small to stress a GPU.

| Job | Model | Data | Proves |
| --- | --- | --- | --- |
| LM demo (M3) | Small char-level LM | Tiny Shakespeare | Training works, loss matches dense, text samples for the video |
| Memory benchmark (M2) | Wide MLP blocks, d = 4096, h = 16384, stacked | Random tensors | Per-GPU memory drops ≈ 2×; too big for one GPU, fits on two |

Sizing (fp32 + Adam ≈ 16 bytes/param, activations extra):

| Model | Params | One GPU | Per GPU, 2 ranks |
| --- | --- | --- | --- |
| 1 block, d = 4096, h = 16384 | 134M | ≈ 2.1 GB | ≈ 1.1 GB |
| **4 blocks (headline on GHC, 8 GB GPUs)** | **537M** | **≈ 8.6 GB (doesn't fit in 8 GB)** | **≈ 4.3 GB** |
| d = 8192, h = 65536 (headline on the Kaggle T4 backup, 16 GB) | 1.07B | ≈ 17 GB (doesn't fit in 16 GB) | ≈ 8.6 GB |

Benchmarks to produce (modeled on the Megatron-LM paper's measurements, adapted to our setup):

| # | Metric | Shown as | Notes |
| --- | --- | --- | --- |
| 1 | **Peak memory per GPU vs. h**, N = 1 vs. 2 | **Headline plot** | The N = 1 line ends at an out-of-memory marker. Megatron doesn't report this; for us it's the main point |
| 2 | **Step-time breakdown:** compute, GPU→host copy, all-reduce, host→GPU copy | Stacked bar | Explains why 2 GPUs are slower per step. Timings come from `all_reduce` |
| 3 | **Communication share vs. h** | Line plot | Each block does 2 all-reduces per step of B × d floats, independent of h, so the share should shrink as h grows |
| 4 | **Achieved FLOP/s per GPU**, and % of RTX 2080 fp32 peak (≈ 10 TFLOPS) | Table | Megatron: 39 TFLOPS, ≈ 30% of peak. Expect a few % for us (homework matmul kernel, not cuBLAS); say why in the report. Use the T4 peak (≈ 8 TFLOPS) only for Kaggle runs |
| 5 | **Weak-scaling efficiency**, N = 1 → 2 | One number + explanation | Grow h with N (h₂ = 2 h₁). Efficiency = throughput₂ / (2 × throughput₁). Megatron: 95% at 2 GPUs over NVLink. Per-GPU compute and comm are both constant here, so this number is the comm overhead (host copies + network); pair it with #2 |
| 6 | **Model configs** | Table | d, h, blocks, params, N for every benchmark point (like Megatron Table 1) |
| 7 | **Time per iteration / tokens per second**, single GPU vs. TP | Table | Like Megatron Table 2, per iteration instead of per epoch |
| 8 | **Convergence:** Tiny Shakespeare loss curves, single vs. TP | Overlay plot | From the M3 runs |
| 9 | **Correctness:** max difference from dense in loss and gradients | Table | Target ≈ 1e-5 |
| — | Downstream quality (WikiText103, LAMBADA, RACE) | Skip | Matching the single-GPU baseline is enough for us |

Items 4–7 come from the same runs as 1–3; the benchmark script only needs to log the config and FLOP count with each result.

**FLOP counting** (use the same formula for every point):

- One block, forward: 2 matmuls × 2·B·d·h = **4·B·d·h**
- Training step (forward + backward ≈ 3 × forward): **≈ 12·B·d·h per block**, **÷ N per GPU** under TP
- Achieved FLOP/s per GPU = FLOPs per GPU per step ÷ measured step time (median over steps, after warm-up)

Measure memory per rank with `nvidia-smi --query-gpu=memory.used` or `pynvml` at the peak of a step. Record the GPU model with every run.

## Final report notebook

Scaffold: `report/final_report.ipynb`. Colab has one GPU, so the notebook shows three levels of evidence:

| Evidence | Where it runs | What it shows |
| --- | --- | --- |
| numpy simulation of 2 ranks | Notebook kernel | The math, with a hand-checkable forward and backward example |
| `mpirun -np 2` cells | Colab CPU (real processes, real MPI) | Our code works across processes and matches dense; per-rank parameter memory halves |
| Results files | Two GHC machines (or Kaggle 2 × T4) | Memory savings, the out-of-memory headline, comm cost, full LM loss curves |

What the notebook needs from the code:

- [ ] `distributed.py` falls back to world size 1 without `mpirun`, so layers can be imported in a notebook cell
- [ ] Training, test and benchmark code are scripts with flags (`--steps`, `--hidden`, `--out`); rank 0 writes JSON
- [ ] Results JSON follows the format in the notebook (`train_dense.json`, `train_tp.json`, `bench.json`)
- [ ] Each GHC run records the command, GPU model, and whether the 2 ranks are on one machine or two

Figures in the report must be our own (the plagiarism policy covers figures, even with citation). Code goes in as a private Drive link, not a public GitHub repo.

## Risks

| Risk | Fallback |
| --- | --- |
| GHC two-machine runs stop working (machines down, shared connections) | Kaggle 2 × T4; then GCP 2 × T4 for a few hours; otherwise CPU-only demo with per-rank memory/comm |
| Activations push the 4-block model over ≈ 4.3 GB per RTX 2080 | Lower the batch size; the weights + Adam state still show the 2× saving |
| Network between GHC machines makes the comm cost dominate | Expected and reportable: measure bandwidth separately and compare with the theoretical cost |
| Host-side all-reduce makes GPU runs look slow | Report it as overhead, compare with the theoretical cost, list NCCL as future work |
| Model too small to show memory savings | Scale width until one GPU runs out of memory |

## Known traps

Each one gives wrong numbers or a hang, not an error message.

- Row-parallel bias goes **after** g; before it, it gets added N times.
- Every rank builds the **full** weight from a shared seed, then slices. Initializing the shard at its own shape uses the wrong fan_in.
- Column slices are non-compact; compact or copy before handing to MPI.
- h must be divisible by N.
- Every rank must see the same batch (same data seed).
- Every rank must call every collective; one rank skipping one = hang.
- Sharded and dense results match up to float rounding, not bit for bit. Compare generated text with greedy decoding and the same seeds.

## Open questions

- [ ] Which day is our in-person presentation slot?
- [x] Does `mpirun` across two GHC machines work? Yes (Nov 21), via SSH connection sharing; Kerberos login is refused by the servers. See Compute.
- [ ] Is the GitHub repo private?
