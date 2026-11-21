# Tensor Parallelism in Needle: Project Plan

10-414 Fall 2026 Final Project · Evelyn Lui, Neethi Karnam, Anishka Mohanty

This plan covers how the pieces fit together (forward pass, backward pass, all-reduce, MPI, TCP), and how the three of us can build them at the same time without blocking each other.

## 1. TCP, MPI and all-reduce are different layers

These three words describe different levels of the same system:

- **All-reduce** is *what* we need: every rank contributes an array, and every rank gets back the sum.
- **MPI** is a library that gives us all-reduce ready-made, plus a way to start processes on several machines. GHC has Open MPI 4.1.6, and we call it from Python through `mpi4py`.
- **TCP** is the network connection underneath that carries the raw bytes. It has no idea what an array, a rank or a sum is. On GHC, Open MPI itself sends its messages over TCP between machines.

If we use MPI, we call `Allreduce`. If we use TCP directly, we write the message format (shape, dtype, bytes), the connection setup and the all-reduce algorithm ourselves. With two ranks, all-reduce is just a swap and an add. With more ranks we can use a hub (rank 0 sums everything and sends it back) or a ring (each rank passes chunks to the next).

### The software stack, top to bottom

| Layer | What it does | File |
|---|---|---|
| Model | ColumnParallelLinear → ReLU → RowParallelLinear. Each rank stores only its shard of W₁ and W₂. | `layers.py` |
| Autograd comm ops | **f**: identity forward, all-reduce backward. **g**: all-reduce forward, identity backward. Plus broadcast. These are Needle `TensorOp`s, so autograd produces the backward pass on its own. | `ops.py` |
| Communicator | One interface: `rank`, `world_size`, `all_reduce(x)`, `broadcast(x, src)`, `barrier()`. Nothing above this layer knows which backend is running. | `comm.py` |
| Backends | `ThreadComm` (laptop tests), `MPIComm` (mpi4py, main option), `SocketComm` (our own TCP, fallback). | `comm.py` |
| Transport | TCP between GHC machines, shared memory between processes on one machine. GPU tensors are copied to host memory with `.numpy()` before they are sent. | — |

## 2. Day 1, all together: agree on the interface

Spend one short meeting agreeing on these signatures and the file layout. After that, all three tracks can work in parallel, because each track can test against `ThreadComm` on a laptop with `NEEDLE_BACKEND=np`.

```python
class Communicator:
    rank: int
    world_size: int
    def all_reduce(self, x: NDArray) -> NDArray: ...          # elementwise sum, result on every rank
    def broadcast(self, x: NDArray, src: int = 0) -> NDArray: ...
    def barrier(self) -> None: ...

def get_comm() -> Communicator:   # picks MPIComm / SocketComm / ThreadComm from env vars
    ...
```

File layout:

```
needle/parallel/
  comm.py        Communicator, ThreadComm, MPIComm, SocketComm, get_comm      (Track C)
  ops.py         copy_to_tp (f), reduce_from_tp (g), broadcast                (Track B)
  layers.py      ColumnParallelLinear, RowParallelLinear, ParallelMLP         (Track A)
  sharding.py    split_column / split_row (exists)
tests_parallel/
  test_comm.py (C) · test_ops.py, test_backward.py (B) · test_forward.py (A, exists)
apps/train_tp.py, benchmarks/                                                 (A + C)
```

## 3. Three parallel tracks

Assign a track to each person. Track C writes `ThreadComm` first, in about half a day, because Tracks A and B test against it.

### Track A: Forward pass & parallel layers

**Owner:** ________

Build:
- `ColumnParallelLinear(in, out, comm)`: holds an `(in, out/k)` shard; forward is `f(X) @ Wᵢ + bᵢ`
- `RowParallelLinear(in, out, comm)`: holds an `(in/k, out)` shard; forward is `g(Xᵢ @ Wᵢ) + b`
- Initialisation: same seed and slice on every rank, plus `from_full(W)` for tests
- `ParallelMLP`, replacing the loop in `mlp.py`
- `apps/train_tp.py`: replicated loss, local SGD on shards

Done when: the existing forward tests pass using the real layers on ThreadComm, for k = 1, 2, 4.

Files: `layers.py`, `mlp.py`, `test_forward.py`, `train_tp.py`

### Track B: Backward pass & autograd ops

**Owner:** ________

Build:
- `CopyToTP` (f): forward identity, gradient `reduce_from_tp(out_grad)`
- `ReduceFromTP` (g): forward `comm.all_reduce`, gradient `out_grad`
- `Broadcast(src)`: gradient sum-reduced back to src
- Gradient tests: compare `dAᵢ` (concatenated), `dBᵢ` (stacked) and `dX` against the dense numpy result
- Multi-step SGD test: sharded and dense weights stay equal

Done when: `test_backward.py` passes for k = 1, 2, 4 on ThreadComm, then under `mpirun`.

Files: `ops.py`, `test_ops.py`, `test_backward.py`

### Track C: Communication & cluster

**Owner:** ________

Build:
- `ThreadComm` first: threads plus a barrier, for laptop tests
- `MPIComm`: `Allreduce` / `Bcast` through mpi4py, with the `.numpy()` round trip for CUDA tensors
- `SocketComm`: our TCP message format, 2-rank swap-and-add, then hub and ring all-reduce
- GHC setup: mpi4py, passwordless ssh, a 2-machine hello all-reduce, Needle CUDA on both machines

Done when: `test_comm.py` passes on all three backends, and all-reduce runs on two GHC GPUs.

Files: `comm.py`, `test_comm.py`, `hello_allreduce.py`

## 4. Phases and integration points

| Phase | Track A | Track B | Track C |
|---|---|---|---|
| **0 · Day 1** | Agree on the interface and file layout. | ← same | Push `ThreadComm`. |
| **1 · Build** | Layers, initialisation, `from_full`. Temporarily use plain identity functions in place of f/g. | f, g and broadcast ops; gradient tests on ThreadComm. | MPIComm; run the GHC setup checks; start SocketComm. |
| **2 · Integrate** | Replace the stand-ins with Track B's ops. The full forward and backward run on ThreadComm. | ← same | Run the same tests under `mpirun -np 2/4` on a laptop or one GHC machine. |
| **3 · Multi-machine** | Everyone: run `train_tp.py` on two GHC machines, one GPU each, with MPIComm (SocketComm if MPI won't launch). | ← same | ← same |
| **4 · Measure** | Per-rank memory scaling; the model that only fits when sharded. | Runtime split into compute and communication; correctness tables for the report. | All-reduce benchmarks: MPI vs. our hub vs. our ring, across message sizes. |

Checkpoints:
- **After phase 0:** `get_comm()` returns a working ThreadComm.
- **After phase 2:** forward and backward tests pass for k = 1, 2, 4.
- **After phase 3:** sharded training on 2 GPUs matches the single-device reference.

## 5. Backward pass: what autograd should produce

Use this to check Track B's tests. Rank i computes Zᵢ = X·Aᵢ, Hᵢ = ReLU(Zᵢ), Pᵢ = Hᵢ·Bᵢ, and Y = g(Σ Pᵢ). The upstream gradient dY is the same on every rank.

```
dPᵢ = dY                          # g backward = identity
dBᵢ = Hᵢᵀ · dY                    # local, no communication
dZᵢ = (dY · Bᵢᵀ) ⊙ 1[Zᵢ > 0]
dAᵢ = Xᵀ · dZᵢ                    # local, no communication
dX  = Σᵢ dZᵢ · Aᵢᵀ                # all-reduce (f backward)
```

- Each MLP block needs one all-reduce in the forward pass and one in the backward pass.
- The row-parallel bias is replicated, so its gradient is identical on every rank and needs no communication.

## 6. Risks and fallbacks

| Risk | Fallback |
|---|---|
| mpi4py isn't available on GHC, or installing it isn't allowed | SocketComm uses only Python and numpy; the code above it is unchanged |
| `mpirun` can't start processes on the second machine (ssh keys on AFS) | Try Kerberos (`kinit`, GSSAPI); otherwise start each rank by hand with SocketComm |
| The GHC Python doesn't match the prebuilt `cp313` Needle backends | Rebuild the backend on GHC using the CUDA toolkit already installed |
| A firewall blocks TCP between two GHC machines | Try another port or pair of machines; ask course staff. Meanwhile run CPU ranks on one machine |
| The project summary promises "multiple GPUs" | Check with the TA that two machines with one GPU each (plus CPU ranks) is acceptable, and update the wording |
