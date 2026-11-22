#!/usr/bin/env bash
# Find GHC machines with an idle GPU and get them ready for mpirun.
#
#   bash scripts/find_gpus.sh             # use only machines you're already logged in to
#   bash scripts/find_gpus.sh --login     # also ask for your password, one machine at a time,
#                                         # until enough idle GPUs are found
#   options: --want N (default 2, counting this machine)  --range 26-50
#
# Step 1 adds every reachable machine's host key to ~/.ssh/known_hosts (no password,
# no "yes" prompt), which fixes "Host key verification failed".
# Step 2 checks GPUs, through the shared connections from docs/GHC_SETUP.md step 4.
# At the end it prints the --host list for mpirun.

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

idle() {  # $1 = "mem, util" from nvidia-smi; returns 0 if idle
  local mem=${1%%,*} util=${1##*, }
  [ -n "$mem" ] && [ "$mem" -le "$max_mem" ] && [ "$util" -le "$max_util" ]
}
query='nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits'

# ---- Step 1: host keys and reachability --------------------------------------
mkdir -p ~/.ssh; touch ~/.ssh/known_hosts; chmod 600 ~/.ssh/known_hosts
up=()
echo "Scanning ghc$first-ghc$last ..."
for i in $(seq "$first" "$last"); do
  h=ghc$i; full=$h.ghc.andrew.cmu.edu
  [ "$h" = "$me" ] && continue
  keys=$(ssh-keyscan -T 3 "$full" 2>/dev/null)
  if [ -z "$keys" ]; then continue; fi          # down or unreachable
  up+=("$h")
  # ssh looks keys up under the HostName (full name), per the ~/.ssh/config block
  ssh-keygen -F "$full" >/dev/null || echo "$keys" >> ~/.ssh/known_hosts
done
echo "Reachable: ${up[*]:-none}"

# ---- Step 2: idle GPUs -------------------------------------------------------
picked=()
r=$($query 2>/dev/null | head -1)
if idle "$r"; then picked+=("$me"); echo "$me (this machine): idle  [$r]";
else echo "$me (this machine): busy or no GPU  [$r]"; fi

for h in "${up[@]}"; do
  [ "${#picked[@]}" -ge "$want" ] && break
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
