# Tensor-parallel benchmark tables

Hardware: NVIDIA GeForce RTX 2080 (cuda); config d=4096, h=8192, blocks=1, batch=128, Adam, fp32. Machines in the N-rank runs: N=1: 1, N=2: 2, N=4: 4, N=8: 8

## Memory

| N | Params per rank | Predicted per rank (GB) | Measured peak per rank (GB) | Fits 8 GB GPU? |
|---|---|---|---|---|
| 1 | 67.1 M | 1.11 | 4.54 (total 4.66 incl. CUDA context) | Yes |
| 2 | 33.6 M | 0.56 | 2.40 (total 2.52 incl. CUDA context) | Yes |
| 4 | 16.8 M | 0.29 | 1.46 (total 1.57 incl. CUDA context) | Yes |
| 8 | 8.4 M | 0.15 | 0.86 (total 0.98 incl. CUDA context) | Yes |

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
| 1 | 1 | 128 | 136.1 | 940.6 | 1.00 | 100 % |
| 2 | 2 | 128 | 106.9 | 1,197.5 | 1.27 | 64 % |
| 4 | 4 | 128 | 92.4 | 1,385.4 | 1.47 | 37 % |
| 8 | 8 | 128 | 92.0 | 1,392.0 | 1.48 | 18 % |

Speedup baseline: 136.1 ms, measured at N=1.

Supplement: one-block results. Successful-run timings are measured; speedup requires a successful one-block N=1 baseline.

| N | Machines | Batch | Median step (ms) | Samples/s | Speedup vs. N=1 | Parallel efficiency |
|---|---|---|---|---|---|---|
| 1 | 1 | 128 | 136.1 | 940.6 | 1.00 | 100 % |
| 2 | 2 | 128 | 106.9 | 1,197.5 | 1.27 | 64 % |
| 4 | 4 | 128 | 92.4 | 1,385.4 | 1.47 | 37 % |
| 8 | 8 | 128 | 92.0 | 1,392.0 | 1.48 | 18 % |

## Communication time

| N | Compute (ms) | GPU→host (ms) | All-reduce (ms) | Host→GPU (ms) | Step total (ms) | Comm share (%) | All-reduces per step | Bytes per all-reduce |
|---|---|---|---|---|---|---|---|---|
| 1 | 136.1 | 0 | 0 | 0 | 136.1 | 0 | 0 | n/a |
| 2 | 81.9 | 0.32 | 19.81 | 0.21 | 106.9 | 19.1 % | 1 | 2.10 MB (= 128 × 4096 × 4) |
| 4 | 48.5 | 0.31 | 43.36 | 0.23 | 92.4 | 47.6 % | 1 | 2.10 MB (= 128 × 4096 × 4) |
| 8 | 37.9 | 0.32 | 53.05 | 0.22 | 92.0 | 58.3 % | 1 | 2.10 MB (= 128 × 4096 × 4) |

Medians over timed steps, slowest rank. Compute = step − comm (queued kernels are charged to compute because all_reduce is preceded by a device sync). All-reduces per step are measured: block 0's input does not require a gradient, so autograd detaches its `f` and only 2·blocks − 1 fire (1 for 1 blocks, not 2). `* ` = N=1 row estimated from the 1-block run.

## Scaling (weak: h = 4096·N, per-rank work constant)

| N | h | Throughput (samples/s × N, model-size-normalized) | Raw samples/s | Ideal (N × throughput₁) | Efficiency |
|---|---|---|---|---|---|
| 1 | 4096 | 1,574.1 | 1,574.1 | 1,574.1 | 100 % |
| 2 | 8192 | 2,418.2 | 1,209.1 | 3,148.3 | 77 % |
| 4 | 16384 | 3,944.0 | 986.0 | 6,296.5 | 63 % |
| 8 | 32768 | 6,797.0 | 849.6 | 12,593.0 | 54 % |

Raw samples/s is flat under perfect weak scaling, because the model grows N-fold while each rank's work stays constant. The normalized column multiplies by N (model FLOPs per sample grow ∝ h), which makes `Ideal = N × throughput₁` correct, and efficiency = raw₍N₎ / raw₍1₎ = step₁ / step_N.

## All-reduce latency

### Host-side MPI Allreduce only (network + MPI)

| Message size | Floats | N=2 latency (µs) | N=4 latency (µs) | N=8 latency (µs) | N=2 algo BW (GB/s) | N=4 algo BW (GB/s) | N=8 algo BW (GB/s) |
|---|---|---|---|---|---|---|---|
| 4 B | 1 | 56.2 | 161.2 | 228.9 | 0.000 | 0.000 | 0.000 |
| 1 KB | 256 | 107.8 | 225.6 | 303.4 | 0.010 | 0.005 | 0.003 |
| 64 KB | 16,384 | 1,001.8 | 1,794.6 | 2,330.4 | 0.065 | 0.037 | 0.028 |
| 1 MB | 262,144 | 14,659.5 | 30,175.4 | 31,365.2 | 0.072 | 0.035 | 0.033 |
| 16 MB | 4,194,304 | 147,274.4 | 292,097.3 | 365,112.0 | 0.114 | 0.057 | 0.046 |
| B × d × 4 | 524,288 | 19,453.4 | 42,730.6 | 58,772.6 | 0.108 | 0.049 | 0.036 |

### End-to-end `distributed.all_reduce` on a device array (GPU→host + MPI + host→GPU)

| Message size | Floats | N=2 latency (µs) | N=4 latency (µs) | N=8 latency (µs) | N=2 algo BW (GB/s) | N=4 algo BW (GB/s) | N=8 algo BW (GB/s) |
|---|---|---|---|---|---|---|---|
| 4 B | 1 | 142.8 | 279.9 | 305.3 | 0.000 | 0.000 | 0.000 |
| 1 KB | 256 | 152.6 | 284.1 | 350.9 | 0.007 | 0.004 | 0.003 |
| 64 KB | 16,384 | 1,027.5 | 1,846.1 | 2,407.2 | 0.064 | 0.036 | 0.027 |
| 1 MB | 262,144 | 10,614.5 | 20,402.8 | 26,287.5 | 0.099 | 0.051 | 0.040 |
| 16 MB | 4,194,304 | 156,731.9 | 301,949.2 | 410,678.6 | 0.107 | 0.056 | 0.041 |
| B × d × 4 | 524,288 | 19,853.1 | 39,143.9 | 50,076.1 | 0.106 | 0.054 | 0.042 |

Algo BW = message bytes / latency. Median over 30–200 iterations after 10 warm-ups, slowest rank. The last row is the message the model actually sends.

## Utilization

| N | FLOPs per GPU per step | Step time (s) | Achieved TFLOPS | % of peak (10 TFLOPS) | GPU util (%) |
|---|---|---|---|---|---|
| 1 | 52 GFLOP | 0.136 | 0.379 | 3.8 % | 47 |
| 2 | 26 GFLOP | 0.107 | 0.241 | 2.4 % | 27 |
| 4 | 13 GFLOP | 0.092 | 0.139 | 1.4 % | 18 |
| 8 | 6 GFLOP | 0.092 | 0.070 | 0.7 % | 13 |

FLOPs = 12·B·d·h per block (fwd + bwd ≈ 3 × fwd), ÷ N, × blocks. Achieved TFLOPS counts the whole step including communication. GPU util is NVML's utilization.gpu (fraction of time a kernel was running, not SM occupancy), averaged over the timed steps. All rows use the selected headline configuration.
