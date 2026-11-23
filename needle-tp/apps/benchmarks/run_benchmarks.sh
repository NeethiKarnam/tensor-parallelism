#!/usr/bin/env bash
# Run every benchmark suite for N = 1, 2, 4, 8, then build the tables.
# Run from needle-tp/. Needs idle GPUs (check nvidia-smi first) and, for N > 1, working mpirun (docs/GHC_SETUP.md).
#
# GHC (one RTX 2080 per machine, N machines for N ranks; log in to each remote machine first):
#   HOSTS=ghc29,ghc32,ghc33,ghc27,ghc41,ghc42,ghc43,ghc44 \
#   PYTHON=~/private/tp-venv/bin/python \
#   MPIRUN_EXTRA='--mca plm_rsh_args "-o BatchMode=yes"' \
#   ./benchmarks/run_benchmarks.sh
#
# Kaggle 2 x T4 (both GPUs on one machine, so only N = 1 and 2):
#   NS="1 2" BENCH_GPUS_PER_NODE=2 MPIRUN="mpirun --allow-run-as-root" PEAK_TFLOPS=8 ./benchmarks/run_benchmarks.sh
#
# CPU smoke test (no GPU; shrink the model):
#   DEVICE=cpu NS="1 2" MPIRUN="mpirun --oversubscribe" SIZE="--d 64 --h 256 --batch 16 --steps 3 --warmup 1" \
#   H_BASE="--h-base 64" ./benchmarks/run_benchmarks.sh
#
# Variables: NS, HOSTS, PYTHON, MPIRUN, MPIRUN_EXTRA, DEVICE, SIZE, H_BASE, OUT, SUITES, PEAK_TFLOPS
set -u
NS=${NS:-"1 2 4 8"}
PYTHON=${PYTHON:-python}
MPIRUN=${MPIRUN:-mpirun}
MPIRUN_EXTRA=${MPIRUN_EXTRA:-}
DEVICE=${DEVICE:-cuda}
SIZE=${SIZE:-}                    # extra flags for bench_tp.py: --d --h --batch --steps --warmup
H_BASE=${H_BASE:-}
OUT=${OUT:-results/tp_bench}
SUITES=${SUITES:-"correctness headline headline1 scaling allreduce"}
PEAK_TFLOPS=${PEAK_TFLOPS:-10}
HOSTS=${HOSTS:-}

run() {  # run <N> <bench_tp.py args...>
  local n=$1; shift
  local hostarg="" xarg=""
  if [ -n "${BENCH_GPUS_PER_NODE:-}" ]; then xarg="-x BENCH_GPUS_PER_NODE"; fi
  if [ -n "$HOSTS" ]; then hostarg="--host $(echo "$HOSTS" | cut -d, -f1-"$n")"; fi
  echo "=== N=$n: $*"
  if [ "$n" = 1 ]; then
    NEEDLE_BACKEND=nd $PYTHON apps/benchmarks/bench_tp.py "$@" --device "$DEVICE" --out-dir "$OUT" $SIZE
  else
    eval "$MPIRUN -np $n $hostarg $MPIRUN_EXTRA -x NEEDLE_BACKEND=nd $xarg $PYTHON \
     apps/benchmarks/bench_tp.py \"\$@\" --device $DEVICE --out-dir $OUT $SIZE"
  fi
  local rc=$?
  [ $rc -ne 0 ] && echo "!! N=$n '$*' exited with $rc (continuing)"
  return 0
}

mkdir -p "$OUT"
for n in $NS; do
  for s in $SUITES; do
    case $s in
      correctness) run "$n" correctness ;;
      headline)    run "$n" headline --blocks 4 ;;            # N=1 is expected to run out of memory
      headline1)   run "$n" headline --blocks 1 ;;            # 1-block baseline that fits on one GPU
      scaling)     run "$n" scaling $H_BASE ;;
      allreduce)   [ "$n" -gt 1 ] && run "$n" allreduce ;;
    esac
  done
done
$PYTHON apps/benchmarks/make_tables.py --dir "$OUT" --peak-tflops "$PEAK_TFLOPS"
