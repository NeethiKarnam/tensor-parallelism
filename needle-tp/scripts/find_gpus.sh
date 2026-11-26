#!/usr/bin/env bash
# Find GHC machines with an idle GPU and get them ready for mpirun.
#
#   bash scripts/find_gpus.sh             # use only machines you're already logged in to
#   bash scripts/find_gpus.sh --login     # also ask for your password, one machine at a time,
#                                         # until enough idle GPUs are found
#   options: --want N (default 2, counting this machine)  --range 26-50
#
# For each machine in the range, in order, until --want idle GPUs are found:
# step 1 adds its host key to ~/.ssh/known_hosts (no password, no "yes" prompt), which
# fixes "Host key verification failed"; step 2 checks its GPU, through the shared
# connections from docs/GHC_SETUP.md step 4. At the end it prints the --host list for mpirun.

want=2; login=0; first=26; last=50
while [ $# -gt 0 ]; do
  case "$1" in
    --login) login=1 ;;
    --want)  want="$2"; shift ;;
    --range) first="${2%-*}"; last="${2#*-}"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

me=$(hostname -s)
max_mem=500   # MiB in use; above this, someone else is probably on the GPU
max_util=10   # % utilization

# Idle = no processes on the GPU at all, plus low memory and utilization as a backstop.
# This is a snapshot: nothing reserves the GPU, so recheck with gpu_status.sh around each run.
idle() {  # $1 = "mem, util, procs"; returns 0 if idle
  local mem util procs
  IFS=', ' read -r mem util procs <<< "$1"
  [ -n "$mem" ] && [ "$procs" = 0 ] && [ "$mem" -le "$max_mem" ] && [ "$util" -le "$max_util" ]
}
# One line: "MiB used, % util, number of GPU processes". Runs locally or over ssh.
query='echo "$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1), $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"'

# Machines are checked in order and the scan stops as soon as `want` idle GPUs are found.
mkdir -p ~/.ssh; touch ~/.ssh/known_hosts; chmod 600 ~/.ssh/known_hosts
picked=()
echo "[MiB used, % util, GPU processes]"
r=$(bash -c "$query" 2>/dev/null | head -1)
if idle "$r"; then picked+=("$me"); echo "$me (this machine): idle  [$r]";
else echo "$me (this machine): busy or no GPU  [$r]"; fi

echo "Scanning ghc$first-ghc$last until $want idle GPU(s) are found ..."
for i in $(seq "$first" "$last"); do
  [ "${#picked[@]}" -ge "$want" ] && break
  h=ghc$i; full=$h.ghc.andrew.cmu.edu
  [ "$h" = "$me" ] && continue
  # Step 1: host key and reachability
  keys=$(ssh-keyscan -T 3 "$full" 2>/dev/null)
  if [ -z "$keys" ]; then continue; fi          # down or unreachable
  # ssh looks keys up under the HostName (full name), per the ~/.ssh/config block
  ssh-keygen -F "$full" >/dev/null || echo "$keys" >> ~/.ssh/known_hosts
  # Step 2: idle GPU
  if ! ssh -O check "$h" 2>/dev/null; then     # no shared connection open yet
    [ "$login" = 1 ] || { echo "$h: not logged in (rerun with --login)"; continue; }
    echo "$h: password to log in (Ctrl-C to stop):"
    ssh "$h" true || continue
  fi
  r=$(ssh -o BatchMode=yes "$h" "$query" 2>/dev/null | head -1)
  if idle "$r"; then picked+=("$h"); echo "$h: idle  [$r]"; else echo "$h: busy  [$r]"; fi
done

echo
if [ "${#picked[@]}" -ge "$want" ]; then
  hosts=$(IFS=,; echo "${picked[*]:0:$want}")
  echo "Use:  mpirun -np $want --host $hosts --mca plm_rsh_args -oBatchMode=yes -x NEEDLE_BACKEND=nd -x HWLOC_COMPONENTS=-gl ..."
  echo "(Record these machines with the run.)"
else
  echo "Found only ${#picked[@]} idle GPU(s): ${picked[*]:-none}. Try --login, or --range with other machines."
  exit 1
fi
