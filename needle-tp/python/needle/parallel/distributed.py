"""
@file distributed.py
@brief MPI communication helpers for tensor parallelism

distributed.py provides the process group used by the tensor parallel layers.
Each process learns its rank and the world size, and ranks sum arrays with
all_reduce. It uses mpi4py when the script is launched with mpirun, and falls
back to a single process (rank 0, world size 1) otherwise.

@author Neethi Karnam <nkarnam@andrew.cmu.edu>
"""
import os
import sys
import time

import numpy as np

from ..backend_ndarray import NDArray

__all__ = [
    "init",
    "is_initialized",
    "get_rank",
    "get_world_size",
    "all_reduce",
    "barrier",
    "get_comm_stats",
    "reset_comm_stats",
]

# Set by mpirun in every process it starts (Open MPI, then MPICH-style launchers).
_LAUNCHER_SIZE_VARS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE")
_LAUNCHER_RANK_VARS = ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK")

_initialized = False
_MPI = None  # mpi4py.MPI, imported in init()
_comm = None  # MPI.COMM_WORLD, or None without mpirun
_rank = 0
_world_size = 1
_debug = False

_STAT_KEYS = ("calls", "bytes", "sync_s", "d2h_s", "allreduce_s", "h2d_s")
_stats = dict.fromkeys(_STAT_KEYS, 0)


def _launched_by_mpi():
    """
    @brief This function checks whether mpirun started this process.
    @return True if a launcher environment variable is set
    """
    return any(v in os.environ for v in _LAUNCHER_SIZE_VARS + _LAUNCHER_RANK_VARS)


def _expected_world_size():
    """
    @brief This function reads the process count set by mpirun.
    @return number of processes, or None if unknown
    """
    for v in _LAUNCHER_SIZE_VARS:
        if v in os.environ:
            return int(os.environ[v])
    return None


def init():
    """
    @brief This function joins the MPI job, or uses one process without mpirun.
    @throws RuntimeError if mpi4py is missing or built for a different MPI
    """
    global _initialized, _MPI, _comm, _rank, _world_size, _debug
    if _initialized:
        return
    _debug = os.environ.get("NEEDLE_TP_DEBUG", "") not in ("", "0")

    if _launched_by_mpi():
        try:
            from mpi4py import MPI  # starts MPI
        except ImportError as e:
            # Falling back to world size 1 would make every rank train the full
            # model alone while the tests still pass, so fail instead.
            raise RuntimeError(
                "Launched by mpirun, but mpi4py could not be imported. Install it in "
                "this Python environment and run again."
            ) from e
        _MPI = MPI
        _comm = MPI.COMM_WORLD
        _rank = _comm.Get_rank()
        _world_size = _comm.Get_size()

        expected = _expected_world_size()
        if expected is not None and expected != _world_size:
            # mpi4py linked against a different MPI than this mpirun
            # Each process starts as its own world of size 1.
            raise RuntimeError(
                f"mpirun started {expected} processes, but mpi4py reports a world size "
                f"of {_world_size}. mpi4py is probably built against a different MPI "
                f"than this mpirun ({MPI.Get_library_version().splitlines()[0].strip()}). "
                "Reinstall mpi4py with the matching mpicc on PATH."
            )
        if _world_size > 1:
            _abort_job_on_uncaught_exception()

    _initialized = True


def _abort_job_on_uncaught_exception():
    """
    @brief This function stops every rank when one rank crashes.
    """
    previous_hook = sys.excepthook

    def hook(exc_type, exc, tb):
        previous_hook(exc_type, exc, tb)
        sys.stderr.flush()
        _comm.Abort(1)

    sys.excepthook = hook


def is_initialized():
    """
    @brief This function checks whether init() has run.
    @return True after init()
    """
    return _initialized


def get_rank():
    """
    @brief This function returns this process's rank.
    @return rank, from 0 to world size - 1
    """
    init()
    return _rank


def get_world_size():
    """
    @brief This function returns the number of ranks.
    @return world size
    """
    init()
    return _world_size


def barrier():
    """
    @brief This function waits until every rank reaches it.
    """
    init()
    if _comm is not None and _world_size > 1:
        _comm.Barrier()


def all_reduce(x):
    """
    @brief This function sums an array across all ranks.
    @param x NDArray to sum, the same shape on every rank
    @return new NDArray on x's device holding the sum
    @throws TypeError if x is a Tensor instead of an NDArray
    """
    init()
    if not isinstance(x, (NDArray, np.ndarray)):
        raise TypeError(
            f"all_reduce takes a raw NDArray, got {type(x).__name__}. Call it from an "
            "op's compute(), or pass tensor.realize_cached_data()."
        )
    if _world_size == 1:
        return x
    if _debug:
        _check_same_shape(x.shape)

    ts = time.perf_counter()
    if isinstance(x, NDArray) and x.device.name == "cuda":
        # GPU kernels run in the background. Copying one float from a fresh array
        # waits for them, so that wait is timed as sync_s (compute), not d2h_s.
        x.device.empty((1,)).numpy()
    t0 = time.perf_counter()
    if isinstance(x, NDArray):
        # Copy views (slices, transposes, broadcasts) into one contiguous block on
        # the device, so only x.size floats go to the host.
        x = x.compact()
        # On cpu this numpy array can share memory with x, so keep x alive and
        # never reduce into it in place.
        host = np.ascontiguousarray(x.numpy(), dtype=np.float32)
    else:  # numpy backend
        host = np.ascontiguousarray(x, dtype=np.float32)
    t1 = time.perf_counter()

    out = np.empty_like(host)
    _comm.Allreduce(host, out, op=_MPI.SUM)
    t2 = time.perf_counter()

    result = NDArray(out, device=x.device) if isinstance(x, NDArray) else out
    t3 = time.perf_counter()

    _stats["calls"] += 1
    _stats["bytes"] += host.nbytes
    _stats["sync_s"] += t0 - ts
    _stats["d2h_s"] += t1 - t0
    _stats["allreduce_s"] += t2 - t1
    _stats["h2d_s"] += t3 - t2
    return result


def _check_same_shape(shape):
    """
    @brief This function checks that every rank passed the same shape.
    @param shape shape of this rank's array
    @throws RuntimeError if the shapes differ
    """
    shapes = _comm.allgather(tuple(shape))
    if any(s != shapes[0] for s in shapes):
        raise RuntimeError(f"all_reduce shape mismatch across ranks: {shapes}")


def get_comm_stats():
    """
    @brief This function returns the communication timing totals.
    @return dict of calls, bytes, sync_s, d2h_s, allreduce_s and h2d_s since the last reset
    """
    return dict(_stats)


def reset_comm_stats():
    """
    @brief This function sets the communication timing totals to zero.
    """
    for k in _STAT_KEYS:
        _stats[k] = 0
