"""Check that every MPI rank can use the CUDA backend on its own machine's GPU.

Each rank runs a matmul on the GPU and compares it with numpy.
Run from needle-tp/ (see docs/GHC_SETUP.md):
    mpirun -np 2 --host ghc29,ghc32 --mca plm_rsh_args "-o BatchMode=yes" -x NEEDLE_BACKEND=nd \
        ~/private/tp-venv/bin/python scripts/check_gpu.py
"""
import os
import socket
import sys

os.environ.setdefault("NEEDLE_BACKEND", "nd")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

import numpy as np
import needle as ndl
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
if not ndl.cuda().enabled():
    print(f"{socket.gethostname()}  rank {rank}  cuda NOT available (was the CUDA backend built?)", flush=True)
    sys.exit(1)

A = np.random.randn(256, 256).astype("float32")
a = ndl.Tensor(A, device=ndl.cuda())
err = np.abs((a @ a).numpy() - A @ A).max()
status = "ok" if err < 1e-3 else "WRONG"
print(f"{socket.gethostname()}  rank {rank}  cuda ok  max error {err:.1e}  {status}", flush=True)
