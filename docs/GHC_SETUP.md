# Running on GHC: build, SSH, and multi-machine `mpirun`

How to go from a fresh Andrew account to a tensor-parallel run with one rank per GHC machine. Each GHC machine has one RTX 2080 (8 GB), so N GPUs means N machines. Verified Nov 21, 2026: CPU all-reduce across ghc29 + ghc32. Verified Nov 22: the TP layers on 4 GPUs across 4 machines (`apps/tp_shard_demo.py --device cuda` on ghc29, 27, 32, 33).

**Overview**

| Step | How often | What it does |
| --- | --- | --- |
| [1. Python env + build](#1-python-env-and-backend-build-once) | Once | Python 3.13 venv, `mpi4py`, compiled Needle backend |
| [2. SSH + shell config](#2-ssh-and-shell-config-once) | Once | Lets `mpirun` start processes on other machines |
| [3. Pick machines](#3-pick-machines-each-time-you-choose-new-ones) | When choosing machines | Find reachable machines with an idle GPU |
| [4. Log in](#4-log-in-to-each-remote-machine-each-session) | Each session (every 4 h) | One password login per remote machine |
| [5. Run](#5-run) | Every run | `mpirun` across the machines |
| [6. Checks](#6-checks-after-setup-or-when-something-breaks) | After setup or when something breaks | Confirms MPI and the GPUs work |

Your home directory is AFS, the same on every GHC machine. Steps 1 and 2 therefore apply everywhere at once.

## 1. Python env and backend build (once)

Work in `~/private`: only you can read it, and the code contains our homework solutions.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'source $HOME/.local/bin/env' >> ~/.bashrc && source ~/.bashrc
cd ~/private && uv python install 3.13 && uv venv --python 3.13 tp-venv
echo 'source ~/private/tp-venv/bin/activate' >> ~/.bashrc && source ~/private/tp-venv/bin/activate
uv pip install numpy mpi4py cmake pybind11

git clone <repo> && cd tensor-parallelism/needle-tp
export PATH=/usr/local/cuda-11.7/bin:$PATH
mkdir -p build && cd build && cmake -DCUDA_HOST_COMPILER=/usr/bin/gcc-11 .. && cd .. && make
NEEDLE_BACKEND=nd python tests_base/test_ops.py     # expect ALL PASS (cpu + cuda)
```

- `/usr/bin/gcc-11`: nvcc 11.7 rejects the system gcc 13.
- **Build on one machine only.** The `Makefile` wipes `build/` when the hostname changes, and every machine sees the same files. Running `make` on another machine during a run would rebuild the `.so` underneath it. One build works on all GHC machines, because we don't use `-march=native`.

## 2. SSH and shell config (once)

`mpirun` starts the other ranks by running `ssh <host> <command>`, and that ssh must not stop for any input. Three things make it work:

- **Connection sharing.** Your first password login to a machine stays open in the background for 4 hours, and later ssh commands (including `mpirun`'s) reuse it. We need this because the GHC servers refuse Kerberos login, and SSH keys don't work well with AFS: the remote side can't read `~/private`. A session that started with a password login can read `~/private`.
- **Short → full names.** Open MPI shortens `ghc32.ghc.andrew.cmu.edu` to `ghc32` before it runs ssh. The `HostName` line maps it back, so `known_hosts` and connection sharing both match.
- **No X11 noise.** hwloc (inside Open MPI) probes the desktop display and prints `Authorization required, but no authorization protocol specified`. It's harmless; the setting below turns it off. It must go at the **top** of `~/.bashrc`, because the non-interactive shells `mpirun` starts stop reading `.bashrc` early.

If `~/.ssh/config` already has other entries, add these blocks to it instead of overwriting the file.

```bash
mkdir -p ~/.ssh && cat > ~/.ssh/config <<'EOF'
# Short names (ghc32) -> full names; Open MPI shortens hostnames before it runs ssh
Host ghc? ghc??
  HostName %h.ghc.andrew.cmu.edu

# Reuse one password login per machine for 4 hours (needed for mpirun)
Host ghc*
  ControlMaster auto
  ControlPath /tmp/ssh-%u-%r@%h:%p
  ControlPersist 4h
EOF
chmod 600 ~/.ssh/config

grep -q HWLOC_COMPONENTS ~/.bashrc || sed -i '1i export HWLOC_COMPONENTS=-gl' ~/.bashrc
```

`ControlPath` must be in `/tmp` (local disk): this kind of shared connection can't be created on AFS. ssh makes it readable only by you.

## 3. Pick machines (each time you choose new ones)

The script does this step and step 4 for you. Run it from `needle-tp/` on the machine you'll launch from:

```bash
bash scripts/find_gpus.sh --login             # 2 machines (this one + 1)
bash scripts/find_gpus.sh --login --want 4    # 4 machines
```

1. It adds every reachable machine's host key to `~/.ssh/known_hosts`, with no password and no "yes" prompt.
2. It checks this machine's GPU, then asks for your password one machine at a time (each login opens the 4 h shared connection) until it has found enough idle GPUs (≤ 500 MiB used, ≤ 10% utilization).
3. It prints the `--host` list to pass to `mpirun`.

Without `--login`, it only checks machines you're already logged in to. Use `--range 26-50` to choose which machines to scan.

To do it by hand instead, find reachable machines from the one you're on:

```bash
for i in $(seq 26 50); do
  printf "%-8s " ghc$i
  ssh -o BatchMode=yes -o ConnectTimeout=3 ghc$i hostname 2>&1 | head -1
done
```

| Output | Meaning |
| --- | --- |
| a hostname | Reachable and already logged in this session |
| `Host key verification failed` or `Permission denied` | Reachable; do step 4 for it |
| `Connection timed out` / `No route to host` | Down; skip it |

On Nov 21, ghc27, 28, 32–34, 38–42, 46 and 48 were reachable from ghc29.

These are shared lab machines, so check the GPU is idle before using one:

```bash
ssh ghc32 nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
```

Write down which machines each run used, so it can be reproduced.

## 4. Log in to each remote machine (each session)

For every machine in your run **except the one you're typing on**:

```bash
ssh ghc32 true      # first time ever: answer "yes" to the fingerprint question; then your password
ssh ghc41 true
```

The first-time `yes` saves the machine to `~/.ssh/known_hosts`. After that, only the password is needed, once every 4 hours. To confirm `mpirun` will get in without a prompt:

```bash
ssh -o BatchMode=yes ghc32 hostname    # must print the hostname with no prompt
```

## 5. Run

Run from `needle-tp/` on the **first** machine in the list. Use **short** names. Give the venv's Python by its full path, because the remote shells don't activate the venv.

```bash
cd ~/private/tensor-parallelism/needle-tp

# 2 ranks (one per machine)
mpirun -np 2 --host ghc29,ghc32 --mca plm_rsh_args "-o BatchMode=yes" -x NEEDLE_BACKEND=nd \
  ~/private/tp-venv/bin/python <script.py> [args]

# 3 ranks
mpirun -np 3 --host ghc29,ghc32,ghc41 --mca plm_rsh_args "-o BatchMode=yes" -x NEEDLE_BACKEND=nd \
  ~/private/tp-venv/bin/python <script.py> [args]

# 4 ranks: the TP layers on 4 GPUs, compared with the dense model (verified Nov 22)
mpirun -np 4 --host ghc29,ghc27,ghc32,ghc33 --mca plm_rsh_args "-o BatchMode=yes" -x NEEDLE_BACKEND=nd \
  ~/private/tp-venv/bin/python apps/tp_shard_demo.py --device cuda --example handcheck
```

Or put the machines in a hostfile (one rank per machine):

```bash
printf "ghc29 slots=1\nghc32 slots=1\nghc41 slots=1\n" > hosts.txt
mpirun -np 3 --hostfile hosts.txt --mca plm_rsh_args "-o BatchMode=yes" -x NEEDLE_BACKEND=nd \
  ~/private/tp-venv/bin/python <script.py> [args]
```

- `-o BatchMode=yes`: if a login is missing, ssh fails with an error instead of hanging silently.
- `-x VAR`: passes an environment variable to every rank. Add any others the script needs.
- Each rank uses its own machine's only GPU (device 0); nothing to configure.
- **The hidden size must be divisible by the number of ranks**: h = 16384 works for 2 or 4 ranks; for 3, use e.g. h = 12288.
- The ranks communicate over the campus network, so the communication cost we report includes it. More machines means more of it.

## 6. Checks (after setup, or when something breaks)

Run these in order from `needle-tp/`. Each one depends on the one before.

```bash
# a) Processes start on every machine: prints each hostname once
mpirun -np 2 --host ghc29,ghc32 --mca plm_rsh_args "-o BatchMode=yes" hostname

# b) Ranks can exchange data: every line ends in "ok" ([3. 3. 3. 3.] for 2 ranks, [6. ...] for 3)
mpirun -np 2 --host ghc29,ghc32 --mca plm_rsh_args "-o BatchMode=yes" \
  ~/private/tp-venv/bin/python scripts/check_allreduce.py

# c) Every rank can use its GPU: "cuda ok" from each machine
mpirun -np 2 --host ghc29,ghc32 --mca plm_rsh_args "-o BatchMode=yes" -x NEEDLE_BACKEND=nd \
  ~/private/tp-venv/bin/python scripts/check_gpu.py
```

## Troubleshooting

These are the errors we actually hit, and their fixes:

| Symptom | Cause | Fix |
| --- | --- | --- |
| `mpirun` hangs with no output | ssh is waiting at a prompt you can't see | Always pass `--mca plm_rsh_args "-o BatchMode=yes"` so it fails with an error instead |
| `Host key verification failed` | A machine (sometimes the one you're on, when listed by its full name) isn't in `known_hosts` under the name ssh was given | `ssh <that host> true` once and answer `yes`. Make sure the `HostName` block is in `~/.ssh/config`. List the local machine by its short name |
| `Permission denied (publickey,gssapi-...,password)` | No shared connection open to that machine (expired after 4 h, or never opened) | `ssh <host> true` with your password (step 4) |
| `ORTE was unable to reliably start one or more daemons` | Generic. The real cause is the line printed above it | Fix that line (usually one of the two rows above) |
| `Authorization required, but no authorization protocol specified` | hwloc probing the X11 display | Harmless. `HWLOC_COMPONENTS=-gl` must be at the top of `~/.bashrc` (step 2) |
| `hostname` works but the all-reduce (check b) hangs | The ranks can't open a direct connection to each other (firewall or wrong network interface) | `ip -br addr` to find the interface with the 128.2.x.x address, then add `--mca btl_tcp_if_include <iface> --mca oob_tcp_if_include <iface>` |
| `cuda NOT available` on one rank | That process can't load the CUDA `.so` | Make sure the build finished (step 1) and you're using the venv's Python by full path |
| `Illegal instruction` | A `.so` built with CPU-specific flags | Rebuild from current `main` (no `-march=native`) |
| Kerberos: `klist` shows a valid ticket, but ssh still asks for a password | The GHC servers reject Kerberos (GSSAPI) logins, and only CMU's admins can fix that | Not needed: connection sharing replaces it |
