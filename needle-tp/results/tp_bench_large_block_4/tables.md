# Tensor-parallel benchmark tables

Hardware: NVIDIA GeForce RTX 2080 (cuda); config d=4096, h=16384, blocks=4, batch=128, Adam, fp32. Machines in the N-rank runs: N=1: 1, N=4: 4, N=8: 8

## Memory

| N | Params per rank | Predicted per rank (GB) | Measured peak per rank (GB) | Fits 8 GB GPU? |
|---|---|---|---|---|
| 1 | 537.0 M | 8.84 | OOM (reached 6.27 GB) | No |
| 2 | 268.5 M | 4.45 | — | — |
| 4 | 134.3 M | 2.25 | 6.28 (total 6.40 incl. CUDA context) | Yes |
| 8 | 67.1 M | 1.15 | 3.23 (total 3.35 incl. CUDA context) | Yes |

Predicted = 16 B/param (weights, grads, Adam m and v) + activation estimate (Needle keeps `.grad` on every graph node, so saved activations count twice). GB = 1e9 bytes; an 8 GB card has 8.59e9 B. Measured = NVML peak of this process minus the CUDA-context baseline. Memory source: nvidia-smi (`nvml-device` = whole GPU, so other users' memory would be included).

## Correctness

| Test | N | max abs Δ loss | max abs Δ gradient | max abs Δ dX | Pass (≤ 1e-5) |
|---|---|---|---|---|---|
| One TPMLP block | 1 | 0.0e+00 | 0.0e+00 | 0.0e+00 | PASS |
| One TPMLP block | 2 | 0.0e+00 | 5.6e-09 | 1.9e-09 | PASS |
| One TPMLP block | 4 | 0.0e+00 | 7.5e-09 | 1.9e-09 | PASS |
| One TPMLP block | 8 | 0.0e+00 | 7.5e-09 | 2.1e-09 | PASS |
| Two stacked blocks | 1/2/4/8 | 1.2e-07 | 1.3e-08 | 2.8e-09 | PASS |
| 5 Adam steps (worst step) | 1/2/4/8 | 4.8e-07 | 9.2e-07 | n/a | PASS |

Run on cuda (NVIDIA GeForce RTX 2080), d=64, h=256, B=16, loss = mean squared error. Worst gradient error relative to the largest gradient entry: 2.0e-05. Worst case over all ranks. Δ = TP result vs. the dense model built from the same weights.

## Throughput

| N | Machines | Batch | Median step (ms) | Samples/s | Speedup vs. N=1 | Parallel efficiency |
|---|---|---|---|---|---|---|
| 1 | 1 | 128 | OOM | — | — | — |
| 2 | 2 | 128 | — | — | — | — |
| 4 | 4 | 128 | 659.2 | 194.2 | — | — |
| 8 | 8 | 128 | 633.2 | 202.1 | — | — |

Supplement: the same comparison with 1 block (134M params), which fits on one GPU, so every number is measured.

| N | Machines | Batch | Median step (ms) | Samples/s | Speedup vs. N=1 | Parallel efficiency |
|---|---|---|---|---|---|---|
| 1 | — | 128 | — | — | — | — |
| 2 | 2 | 128 | 172.7 | 741.0 | — | — |
| 4 | 4 | 128 | 137.5 | 931.2 | — | — |
| 8 | 8 | 128 | 112.5 | 1,137.5 | — | — |

## Communication time

| N | Compute (ms) | GPU→host (ms) | All-reduce (ms) | Host→GPU (ms) | Step total (ms) | Comm share (%) | All-reduces per step | Bytes per all-reduce |
|---|---|---|---|---|---|---|---|---|
| 1 | — | — | — | — | — | — | — | B × 4096 × 4 |
| 2 | — | — | — | — | — | — | — | B × 4096 × 4 |
| 4 | 329.7 | 2.19 | 346.04 | 1.98 | 659.2 | 53.1 % | 7 | 2.10 MB (= 128 × 4096 × 4) |
| 8 | 191.5 | 2.18 | 450.36 | 2.04 | 633.2 | 71.8 % | 7 | 2.10 MB (= 128 × 4096 × 4) |

Medians over timed steps, slowest rank. Compute = step − comm (queued kernels are charged to compute because all_reduce is preceded by a device sync). All-reduces per step are measured: block 0's input does not require a gradient, so autograd detaches its `f` and only 2·blocks − 1 fire (7 for 4 blocks, not 8). `* ` = N=1 row estimated from the 1-block run.

## Scaling (weak: h = 4096·N, per-rank work constant)

| N | h | Throughput (samples/s × N, model-size-normalized) | Raw samples/s | Ideal (N × throughput₁) | Efficiency |
|---|---|---|---|---|---|
| 1 | 4096 | 443.3 | 443.3 | 443.3 | 100 % |
| 2 | 8192 | — | — | — | — |
| 4 | 16384 | — | — | — | — |
| 8 | 32768 | — | — | — | — |

Raw samples/s is flat under perfect weak scaling, because the model grows N-fold while each rank's work stays constant. The normalized column multiplies by N (model FLOPs per sample grow ∝ h), which makes `Ideal = N × throughput₁` correct, and efficiency = raw₍N₎ / raw₍1₎ = step₁ / step_N.

## All-reduce latency

### Host-side MPI Allreduce only (network + MPI)

| Message size | Floats | N=2 latency (µs) | N=4 latency (µs) | N=8 latency (µs) | N=2 algo BW (GB/s) | N=4 algo BW (GB/s) | N=8 algo BW (GB/s) |
|---|---|---|---|---|---|---|---|
| 4 B | 1 | 112.5 | 217.1 | 230.5 | 0.000 | 0.000 | 0.000 |
| 1 KB | 256 | 110.3 | 243.1 | 292.9 | 0.009 | 0.004 | 0.003 |
| 64 KB | 16,384 | 1,002.7 | 1,785.4 | 2,357.0 | 0.065 | 0.037 | 0.028 |
| 1 MB | 262,144 | 10,364.3 | 30,275.5 | 31,818.5 | 0.101 | 0.035 | 0.033 |
| 16 MB | 4,194,304 | 147,318.1 | 257,350.8 | 348,799.4 | 0.114 | 0.065 | 0.048 |
| B × d × 4 | 524,288 | 19,273.6 | 38,352.7 | 55,380.0 | 0.109 | 0.055 | 0.038 |

### End-to-end `distributed.all_reduce` on a device array (GPU→host + MPI + host→GPU)

| Message size | Floats | N=2 latency (µs) | N=4 latency (µs) | N=8 latency (µs) | N=2 algo BW (GB/s) | N=4 algo BW (GB/s) | N=8 algo BW (GB/s) |
|---|---|---|---|---|---|---|---|
| 4 B | 1 | 143.8 | 284.9 | 309.6 | 0.000 | 0.000 | 0.000 |
| 1 KB | 256 | 157.8 | 288.5 | 357.3 | 0.006 | 0.004 | 0.003 |
| 64 KB | 16,384 | 1,030.0 | 1,822.7 | 2,419.7 | 0.064 | 0.036 | 0.027 |
| 1 MB | 262,144 | 10,626.9 | 20,655.2 | 26,156.0 | 0.099 | 0.051 | 0.040 |
| 16 MB | 4,194,304 | 156,966.3 | 230,669.7 | 374,987.8 | 0.107 | 0.073 | 0.045 |
| B × d × 4 | 524,288 | 19,834.9 | 30,589.8 | 53,531.9 | 0.106 | 0.069 | 0.039 |

Algo BW = message bytes / latency. Median over 30–200 iterations after 10 warm-ups, slowest rank. The last row is the message the model actually sends.

## Utilization

| N | FLOPs per GPU per step | Step time (s) | Achieved TFLOPS | % of peak (10 TFLOPS) | GPU util (%) |
|---|---|---|---|---|---|
| 1 | 412 GFLOP | — | — | — | — |
| 2 | 206 GFLOP | — | — | — | — |
| 4 | 103 GFLOP | 0.659 | 0.156 | 1.6 % | 49 |
| 8 | 52 GFLOP | 0.633 | 0.081 | 0.8 % | 27 |

FLOPs = 12·B·d·h per block (fwd + bwd ≈ 3 × fwd), ÷ N, × blocks. Achieved TFLOPS counts the whole step including communication. GPU util is NVML's utilization.gpu (fraction of time a kernel was running, not SM occupancy), averaged over the timed steps. N=1 uses the 1-block run.
