"""Check that MPI ranks can exchange data, e.g. across GHC machines.

Each rank contributes rank + 1, so every rank should print the sum 1 + 2 + ... + N.
Run (see docs/GHC_SETUP.md):
    mpirun -np 2 --host ghc29,ghc32 --mca plm_rsh_args "-o BatchMode=yes" \
        ~/private/tp-venv/bin/python scripts/check_allreduce.py
"""
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

x = np.full(4, rank + 1.0)
y = np.empty_like(x)
comm.Allreduce(x, y)

expected = size * (size + 1) / 2
status = "ok" if np.allclose(y, expected) else f"WRONG (expected {expected})"
print(f"{MPI.Get_processor_name()}  rank {rank}/{size}  {y}  {status}", flush=True)
