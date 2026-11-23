"""Plot dense vs tensor-parallel Tiny Shakespeare runs.

Expects (written by the run_suite loop):
    results/lm_strict/{dense,tp_n1,tp_n2,tp_n4,tp_n8}.json   (--steps 100)
    results/lm_full/{dense,tp_n1,tp_n2,tp_n4,tp_n8}.json     (--steps 5000)

Run from needle-tp/:
    ~/private/tp-venv/bin/python apps/plot_lm.py
    ~/private/tp-venv/bin/python apps/plot_lm.py --root results --ns 1 2 4 8
    ~/private/tp-venv/bin/python apps/plot_lm.py --per-n     # also one dense-vs-TP panel per N

Writes loss_curves.png and scaling.png into lm_full/, correctness.png into lm_strict/,
and with --per-n also per_N.png into lm_full/.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--root", default="results")
p.add_argument("--ns", type=int, nargs="+", default=[1, 2, 4, 8])
p.add_argument("--smooth", type=int, default=25, help="moving-average window (in logged points)")
p.add_argument("--tol", type=float, default=1e-5)
p.add_argument("--per-n", action="store_true", help="also write per_N.png: one dense-vs-TP panel per N")
args = p.parse_args()

STYLES = {1: "-", 2: "--", 4: "-.", 8: ":"}
COLORS = {1: "tab:blue", 2: "tab:orange", 4: "tab:green", 8: "tab:red"}


def load(tag):
    folder = os.path.join(args.root, f"lm_{tag}")
    dense = json.load(open(os.path.join(folder, "dense.json")))
    tp = {}
    for n in args.ns:
        path = os.path.join(folder, f"tp_n{n}.json")
        if os.path.exists(path):
            tp[n] = json.load(open(path))
        else:
            print(f"warning: {path} missing, skipping N={n}")
    # a comparison is only valid if every run used the same settings
    for n, r in tp.items():
        a, b = dict(dense["config"]), dict(r["config"])
        if a != b:
            print(f"warning: config of N={n} differs from dense: {a} vs {b}")
    return folder, dense, tp


def smooth(y, k):
    y = np.asarray(y, dtype=float)
    if k <= 1 or len(y) < k:
        return y
    ker = np.ones(k) / k
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, ker, mode="valid")


# ---------------- Figure 1: convergence (full run) ----------------
folder, d, t = load("full")
step = np.array(d["step"])
dense_loss = np.array(d["loss"])

fig, ax = plt.subplots(1, 4, figsize=(21, 4.4))

ax[0].plot(step, smooth(dense_loss, args.smooth), "k", lw=3, alpha=.3, label="dense")
for n, r in t.items():
    ax[0].plot(r["step"], smooth(r["loss"], args.smooth), STYLES[n], color=COLORS[n], lw=1.4, label=f"TP N={n}")
ax[0].set(xlabel="step", ylabel=f"train loss (moving avg, {args.smooth} pts)", title="Training loss")
ax[0].legend()

ax[1].plot(d["val_step"], d["val_loss"], "k", lw=3, alpha=.3, label="dense")
for n, r in t.items():
    ax[1].plot(r["val_step"], r["val_loss"], STYLES[n], color=COLORS[n], lw=1.4, label=f"TP N={n}")
ax[1].set(xlabel="step", ylabel="val loss", title="Validation loss (same batches every eval)")
ax[1].legend()

# zoom on the last 40% of validation, where any real gap would be visible
cut = int(0.6 * len(d["val_step"]))
ax[2].plot(d["val_step"][cut:], d["val_loss"][cut:], "k", lw=3, alpha=.3, label="dense")
for n, r in t.items():
    ax[2].plot(r["val_step"][cut:], r["val_loss"][cut:], STYLES[n], color=COLORS[n], lw=1.4, marker="o", ms=3, label=f"N={n}")
ax[2].set(xlabel="step", ylabel="val loss", title="Validation loss, zoomed (last 40%)")
ax[2].legend()

for n, r in t.items():
    diff = np.abs(dense_loss - np.array(r["loss"])) + 1e-9
    ax[3].semilogy(step, diff, STYLES[n], color=COLORS[n], lw=1, alpha=.8, label=f"N={n}")
ax[3].axhline(args.tol, color="gray", ls="--", lw=1, label=f"{args.tol:g}")
ax[3].set(xlabel="step", ylabel="|train loss - dense|", title="Difference from dense (float-order drift)")
ax[3].legend()

plt.tight_layout()
out = os.path.join(folder, "loss_curves.png")
plt.savefig(out, dpi=150); plt.close()
print("saved", out)

# ---------------- Optional: one panel per N ----------------
if args.per_n:
    ncols = len(t)
    fig, ax = plt.subplots(2, ncols, figsize=(4.6 * ncols, 7), sharex="row", squeeze=False)
    for i, (n, r) in enumerate(t.items()):
        ax[0, i].plot(step, smooth(dense_loss, args.smooth), "k", lw=3, alpha=.3, label="dense")
        ax[0, i].plot(r["step"], smooth(r["loss"], args.smooth), color=COLORS[n], lw=1.3, label=f"TP N={n}")
        ax[0, i].set(title=f"N={n}: train loss", xlabel="step")
        ax[0, i].legend()
        ax[1, i].plot(d["val_step"], d["val_loss"], "k", lw=3, alpha=.3, label="dense")
        ax[1, i].plot(r["val_step"], r["val_loss"], color=COLORS[n], lw=1.3, marker="o", ms=2.5, label=f"TP N={n}")
        ax[1, i].set(title=f"N={n}: val loss", xlabel="step")
        ax[1, i].legend()
    ax[0, 0].set_ylabel(f"train loss (moving avg, {args.smooth} pts)")
    ax[1, 0].set_ylabel("val loss")
    # shared y range per row so panels are directly comparable
    for row in (0, 1):
        lo = min(a.get_ylim()[0] for a in ax[row]); hi = max(a.get_ylim()[1] for a in ax[row])
        for a in ax[row]:
            a.set_ylim(lo, hi)
    plt.tight_layout()
    out = os.path.join(folder, "per_N.png")
    plt.savefig(out, dpi=150); plt.close()
    print("saved", out)

# ---------------- Figure 2: scaling bars (full run) ----------------
labels = ["dense"] + [f"N={n}" for n in t]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].bar(labels, [d["params_per_rank"]] + [r["params_per_rank"] for r in t.values()])
ax[0].set(ylabel="parameters per rank", title="Memory: params per rank")
ax[1].bar(labels, [d["step_ms_median"]] + [r["step_ms_median"] for r in t.values()])
ax[1].set(ylabel="median ms / step", title="Step time")
ax[2].bar([f"N={n}" for n in t], [r["comm"]["allreduce_s"] for r in t.values()])
ax[2].set(ylabel="seconds", title="Total all-reduce time")
plt.tight_layout()
out = os.path.join(folder, "scaling.png")
plt.savefig(out, dpi=150); plt.close()
print("saved", out)

# ---------------- Figure 3: strict correctness (first 100 steps) ----------------
folder, d, t = load("strict")
s = np.array(d["step"])
a = np.array(d["loss"])
plt.figure(figsize=(6.5, 4.4))
worst = {}
for n, r in t.items():
    diff = np.abs(a - np.array(r["loss"]))
    worst[n] = diff.max()
    plt.semilogy(s, diff + 1e-9, STYLES[n], color=COLORS[n], lw=1.4, label=f"N={n}  (max {diff.max():.1e})")
plt.axhline(args.tol, color="gray", ls="--", lw=1, label=f"tolerance {args.tol:g}")
plt.xlabel("step"); plt.ylabel("|train loss - dense|")
plt.title("Correctness vs dense, first 100 steps")
plt.legend(fontsize=8); plt.tight_layout()
out = os.path.join(folder, "correctness.png")
plt.savefig(out, dpi=150); plt.close()
print("saved", out)

print("\nstrict check (first 100 steps):")
for n, w in worst.items():
    print(f"  N={n}: max |diff| = {w:.2e}  {'PASS' if w < args.tol else 'FAIL'}")
if any(w >= args.tol for w in worst.values()):
    raise SystemExit(1)
