# Project Title: Tensor Parallelism in Needle: Autograd-Aware Collectives and Sharded Linear Layers

10-414 Fall 2026 Final Project

By: Evelyn Lui, Neethi Karnam, and Anishka Mohanty

# Key Words/Methods In the Project
Tensor parallelism, Megatron-LM, column/row-parallel linear layers, all-reduce collective communication (NCCL / MPI), autograd-aware communication operators, distributed training, model sharding, multi-GPU CUDA backend, memory and communication benchmarking

# Project Summary
We extend Needle with Megatron-style tensor parallelism: autograd-aware communication operators (all-reduce, broadcast) and column- and row-parallel linear layers that shard a model’s weights across multiple GPUs. We validate correctness against a single-device reference and benchmark per-GPU memory, runtime, and communication cost, demonstrating training of a model too large for a single GPU.

# Run
```
cd needle-tp && NEEDLE_BACKEND=np python tests_parallel/test_forward.py
```


