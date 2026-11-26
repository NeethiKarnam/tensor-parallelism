#!/usr/bin/env bash
# Show who is on the GPUs of the given GHC machines, right now.
# Run it just before and just after a benchmark, and save the output with the results,
# so we can tell whether someone else was sharing a GPU during the run.
#
#   bash scripts/gpu_status.sh ghc29,ghc32
#   bash scripts/gpu_status.sh ghc29,ghc32 >> results/gpu_status.log
#
# Remote machines use the shared ssh connections from docs/GHC_SETUP.md step 4
# (run find_gpus.sh first). Exits 1 if any GPU has a process other than our own.

if [ $# -ne 1 ]; then echo "usage: $0 host1,host2,..." >&2; exit 2; fi
me=$(hostname -s)
query='nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader;
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader |
  while IFS=", " read -r pid mem; do echo "  pid $pid ($(ps -o user= -p "$pid" 2>/dev/null || echo "?")): $mem"; done'

echo "== $(date '+%F %T')"
status=0
for h in ${1//,/ }; do
  if [ "$h" = "$me" ]; then out=$(bash -c "$query" 2>&1)
  else out=$(ssh -o BatchMode=yes "$h" "$query" 2>&1) || out="unreachable: $out"; fi
  echo "$h: $out"
  # Any process not owned by us means the GPU is shared.
  if echo "$out" | grep -q "unreachable" || echo "$out" | grep "^  pid" | grep -vq "($USER)"; then status=1; fi
done
[ "$status" = 0 ] || echo "WARNING: a GPU above is unreachable or has someone else's process on it."
exit "$status"
