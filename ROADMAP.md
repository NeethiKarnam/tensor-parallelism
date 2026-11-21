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
- **GPU selection:** `CUDA_VISIBLE_DEVICES=<rank>` per process if a machine has several GPUs. GHC has one GPU per machine, so 2 ranks = 2 machines.
- **LM:** char-level MLP. Last k chars → one-hot in numpy → column-parallel → ReLU → row-parallel → logits (vocab ≈ 65) → `SoftmaxLoss`. No `nn.Embedding`, no HW4 conv/RNN/transformer code, output layer not sharded.
- **M2 headline:** a model that runs out of memory on one RTX 2080 (8 GB) trains on two. Target: 4 blocks of d = 4096, h = 16384 (537M params, ≈ 8.6 GB with Adam).
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
| **Neethi** | `distributed.py` (push it early; Anishka builds on it), `tests_parallel/test_allreduce.py`, and the GHC two-machine check: passwordless SSH, then `mpirun -np 2 --host ghc29,ghcYY hostname` (if it hangs, try `--mca btl_tcp_if_include <interface>`) | `mpirun -np 2` all-reduce passes on cpu and cuda, and the two-machine `hostname` test works |
| **Anishka** | `parallel/ops.py` (f, g) and `parallel/layers.py` (`ColumnParallelLinear`, `RowParallelLinear`, `TPMLP`, `from_dense`) | Layers import and run with world size 1; then pass Evelyn's tests under `mpirun -np 2` |
| **Evelyn** | Initializers PR first (unblocks the dense `nn.Linear` baseline). Then the correctness tests, written from the interfaces above: switch `test_forward.py` to `nd`; `test_backward.py` (each shard's gradient and dX vs. dense); `test_train.py` (a few Adam steps on random data, TP loss matches dense) | Tests run without `mpirun` and with `mpirun -np 2` |

**M1 is done when** `test_forward.py`, `test_backward.py`, and `test_train.py` all pass under `mpirun -np 2`.

### Phase 2: after M1 (Nov 23–26)

| Person | Works on |
| --- | --- |
| Evelyn | Tiny Shakespeare loader + LM training, starting from `test_train.py` → M3 |
| Neethi | Benchmark script + GHC GPU runs → M2 |
| Anishka | Stacked multi-block model for the memory headline, then help whoever is behind and start report sections |

### Dates

| Date | Milestone | Owner |
| --- | --- | --- |
| **Nov 22** | **M1:** forward, backward, and a few training steps across 2 ranks match dense | All |
| Nov 23–24 | LM on Tiny Shakespeare | Evelyn |
| **Nov 24** | **M3:** 2-rank LM loss matches dense | Evelyn |
| Nov 24–26 | GPU benchmark runs on GHC | Neethi, Anishka |
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
| Initializers | `kaiming_uniform`, `xavier_*` still raise `NotImplementedError` | Copy in our HW solutions. Needed by plain `nn.Linear` (the dense baseline) |
| Sharding helpers | Done: `split_column`, `split_row` | — |
| Single-process sharded forward | Done: `tests_parallel/test_forward.py` (8/8) | Switch it from `NEEDLE_BACKEND=np` to `nd` |
| `parallel/distributed.py` | Not started (Neethi, Phase 1) | See Interfaces §1. mpi4py `Allreduce(SUM)`, compact before MPI, world size 1 without `mpirun`. Time the all-reduce and host copies |
| `parallel/ops.py` (f, g) | Not started (Anishka, Phase 1) | f: identity forward, all-reduce backward. g: all-reduce forward, identity backward |
| `parallel/layers.py` | Not started (Anishka, Phase 1) | See Interfaces §2–3. Full weight from a shared seed, then slice. Row-parallel bias added after g |
| Correctness tests | Not started (Evelyn, Phase 1) | `test_backward.py`, `test_train.py`: every shard's gradient and dX, and the loss over a few steps, match dense within 1e-5 under `mpirun -np 2` |
| LM demo | Phase 2 (Evelyn) | Tiny Shakespeare loader, one-hot windows, TP MLP, training loop, greedy sampling |
| Benchmarks | Phase 2 (Neethi, Anishka) | Per-rank memory, step time, all-reduce time (incl. host copies) across widths |

**Branches:** `forward-demo` and `build-setup` are merged into `main`. Phase 1 branches: `initializers`, `comm-backend`, `tp-layers`, `tp-tests`.

## Owners

| Person | Owns | First deliverable |
| --- | --- | --- |
| Evelyn | Phase 1: initializers, TP correctness tests (`test_forward`, `test_backward`, `test_train`). Phase 2: Tiny Shakespeare dataset, char-level LM, loss-curve comparison vs. dense. Also: video, report notebook | Initializers PR merged; tests run with and without `mpirun` |
| Neethi | Phase 1: `needle/parallel/distributed.py` (mpi4py), all-reduce test, multi-machine `mpirun` on GHC. Phase 2: benchmark script (memory, comm timing) and GHC GPU runs | `mpirun -np 2` all-reduce test passes; two-machine `mpirun` works |
| Anishka | Phase 1: f/g ops, column- and row-parallel linear layers, `TPMLP`, `from_dense`. Phase 2: stacked multi-block benchmark model, then help where needed | Layers pass the correctness tests under `mpirun -np 2` |

All three: write the report sections for your own piece, and review each other's PRs before merging to `main`.

## Compute: GHC machines

Develop and run CPU tests on a laptop or Colab (`mpirun -np 2`; on Colab add `--allow-run-as-root --oversubscribe`). Run GPU benchmarks on GHC.

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

### Build steps on GHC

Each person builds in their own `~/private` clone (only you can read `~/private`).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'source $HOME/.local/bin/env' >> ~/.bashrc && source ~/.bashrc
cd ~/private && uv python install 3.13 && uv venv --python 3.13 tp-venv
echo 'source ~/private/tp-venv/bin/activate' >> ~/.bashrc && source ~/private/tp-venv/bin/activate
uv pip install numpy mpi4py cmake pybind11
git clone <repo> && cd tensor-parallelism/needle-tp
export PATH=/usr/local/cuda-11.7/bin:$PATH
mkdir -p build && cd build && cmake -DCUDA_HOST_COMPILER=/usr/bin/gcc-11 .. && cd .. && make
NEEDLE_BACKEND=nd python tests_base/test_ops.py     # expect ALL PASS (cpu + cuda)
```

Build on one machine only: the `Makefile` wipes `build/` when the hostname changes, and your home directory is shared across GHC machines, so don't run `make` on the second machine during a 2-machine run.

### Still to check

- [ ] Passwordless SSH from one GHC machine to another, then `mpirun -np 2 --host ghc29,ghcYY hostname`

### Backups if GHC falls through

| Option | GPUs | Cost | Notes |
| --- | --- | --- | --- |
| Kaggle Notebooks | 2 × T4 | Free, weekly quota | Run `!mpirun` from a cell |
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
| **4 blocks (headline)** | **537M** | **≈ 8.6 GB (doesn't fit in 8 GB)** | **≈ 4.3 GB** |
| d = 8192, h = 65536 | 1.07B | ≈ 17 GB | ≈ 8.6 GB |

Benchmarks to produce:

1. **Peak memory per GPU vs. width**, 1 vs. 2 GPUs; the 1-GPU line ends at an out-of-memory marker. Headline plot.
2. **Step time broken down:** compute, GPU→host copy, MPI all-reduce, host→GPU copy. Expect 2 GPUs to be slower per step (host copies); the point is that the model fits.
3. **Communication share vs. width.** Each block does 2 all-reduces per step of batch × d floats, independent of h, so the comm share should shrink as h grows.
4. **Correctness table:** max difference from dense in loss and gradients (target ≈ 1e-5), plus the loss-curve overlay.

Measure memory per rank with `nvidia-smi --query-gpu=memory.used` or `pynvml` at the peak of a step.

## Final report notebook

Scaffold: `report/final_report.ipynb`. Colab has one GPU, so the notebook shows three levels of evidence:

| Evidence | Where it runs | What it shows |
| --- | --- | --- |
| numpy simulation of 2 ranks | Notebook kernel | The math, with a hand-checkable forward and backward example |
| `mpirun -np 2` cells | Colab CPU (real processes, real MPI) | Our code works across processes and matches dense; per-rank parameter memory halves |
| Results files | GHC GPUs | Memory savings, the out-of-memory headline, comm cost, full LM loss curves |

What the notebook needs from the code:

- [ ] `distributed.py` falls back to world size 1 without `mpirun`, so layers can be imported in a notebook cell
- [ ] Training, test and benchmark code are scripts with flags (`--steps`, `--hidden`, `--out`); rank 0 writes JSON
- [ ] Results JSON follows the format in the notebook (`train_dense.json`, `train_tp.json`, `bench.json`)
- [ ] Each GHC run records the command, GPU model, and whether the 2 ranks are on one machine or two

Figures in the report must be our own (the plagiarism policy covers figures, even with citation). Code goes in as a private Drive link, not a public GitHub repo.

## Risks

| Risk | Fallback |
| --- | --- |
| Two-machine `mpirun` doesn't work on GHC | Kaggle 2 × T4, or show the demo on 2 CPU processes and benchmark per-rank memory/comm on CPU |
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
- [ ] Does passwordless SSH + `mpirun` across two GHC machines work?
- [ ] Is the GitHub repo private?
