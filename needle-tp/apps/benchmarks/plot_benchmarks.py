"""
@file plot_benchmarks.py
@brief Makes the presentation graphs from the bench_tp.py result files

plot_benchmarks reads the JSON files that bench_tp.py writes under results/ and
draws every benchmark graph: memory, correctness, step time breakdown, strong
and weak scaling, utilization, and all-reduce latency and bandwidth. Each graph
is saved as a PNG, and the numbers behind it are saved as a CSV next to it so
they can be checked or pasted into a spreadsheet. It can be run from any folder:
    python apps/benchmarks/plot_benchmarks.py
    python apps/benchmarks/plot_benchmarks.py --results results --out results/plots
By default it reads needle-tp/results and writes needle-tp/results/plots.

@author Neethi Karnam <nkarnam@andrew.cmu.edu>
"""
import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

WORLD = (1, 2, 4, 8)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # needle-tp/
GB = 1e9
GPU_LIMIT_GB = 8 * 2**30 / GB  # an 8 GB card holds 8.59e9 bytes
ETHERNET_GBPS = 0.125  # 1 Gb/s link = 0.125 GB/s

# Colors. Each N keeps its color on every graph.
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"
N_COLOR = {1: BLUE, 2: ORANGE, 4: AQUA, 8: YELLOW}
PREDICTED = "#c9c8c1"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style():
    """
    @brief This function sets one plain style for every graph.
    """
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": INK_2, "axes.edgecolor": "#bdbcb5",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 9.5,
        "lines.linewidth": 2.2, "lines.markersize": 7,
    })


# ---------------------------------------------------------------- reading results

def load(run_dir, name):
    """
    @brief This function reads one result file, or reports an out-of-memory run.
    @param run_dir folder holding the result files of one run
    @param name file name without .json, for example headline_b1_N2
    @return the result dict, a dict with status OOM if only per-rank OOM files exist, or None
    """
    path = os.path.join(run_dir, name + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # A multi-rank run that runs out of memory writes only *_rank*_oom.json, one per failing rank.
    ooms = sorted(glob.glob(os.path.join(run_dir, name + "_rank*_oom.json")))
    if not ooms:
        return None
    ranks = []
    for p in ooms:
        with open(p) as f:
            ranks.append(json.load(f))
    r = max(ranks, key=lambda x: x.get("peak_model_bytes", 0))  # report the rank that reached the most
    r["status"] = "OOM"
    return r


def ok(r):
    """
    @brief This function checks whether a run finished.
    @param r result dict or None
    @return True if the run finished without running out of memory
    """
    return r is not None and r.get("status") == "ok"


def setup_text(r):
    """
    @brief This function describes the model, hardware and network of a run.
    @param r result dict
    @return text such as "d=4096, h=8192, 1 block"
    """
    c = r["config"]
    blocks = f"{c['blocks']} block" + ("s" if c["blocks"] > 1 else "")
    return f"d={c['d']}, h={c['h']}, {blocks}"


def figure(title, subtitle, ncols=1, width=None, height=4.6):
    """
    @brief This function makes a figure with a headline title and a gray setup line.
    @param title takeaway sentence shown in bold
    @param subtitle setup line shown under the title
    @param ncols number of side-by-side panels
    @param width figure width in inches
    @param height figure height in inches
    @return (figure, list of axes)
    """
    fig, axes = plt.subplots(1, ncols, figsize=(width or 6.8 * ncols, height), squeeze=False)
    fig.suptitle(title, x=0.01, y=1.0, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, 0.94, subtitle, ha="left", va="top", fontsize=9.5, color=INK_2)
    fig.subplots_adjust(top=0.80 if ncols > 1 else 0.86, wspace=0.28)
    return fig, list(axes[0])


def n_axis(ax, label="Number of GPUs (N)", extra=None):
    """
    @brief This function puts N = 1, 2, 4, 8 evenly spaced on the x axis.
    @param ax axes
    @param label axis title
    @param extra optional dict N -> second tick line, for example the model width
    """
    ax.set_xscale("log", base=2)
    ax.set_xticks(WORLD, [f"{n}" + (f"\n{extra[n]}" if extra else "") for n in WORLD])
    ax.minorticks_off()
    ax.set_xlim(0.75, 10.5)
    ax.set_xlabel(label)


def bar_labels(ax, xs, heights, texts, inside=False):
    """
    @brief This function writes a value above (or inside) each bar.
    @param ax axes
    @param xs bar positions
    @param heights bar heights
    @param texts label for each bar
    @param inside True to put the label inside the top of the bar
    """
    top = ax.get_ylim()[1]
    for x, h, t in zip(xs, heights, texts):
        ax.text(x, h + top * 0.015, t, ha="center", va="bottom", fontsize=9.5, color=INK)


def save(fig, out_dir, name, header, rows):
    """
    @brief This function saves a graph as PNG and its numbers as CSV.
    @param fig matplotlib figure
    @param out_dir output folder
    @param name file name without extension
    @param header CSV column names
    @param rows CSV rows
    """
    fig.savefig(os.path.join(out_dir, name + ".png"))
    plt.close(fig)
    with open(os.path.join(out_dir, name + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {name}.png")


def plot_memory_large(run_dir, prefix, out_dir):
    """
    @brief This function compares predicted and measured memory per GPU for the large model.
    @param run_dir folder of the large-model run
    @param prefix result file prefix, for example headline_b4
    @param out_dir output folder
    """
    res = {n: load(run_dir, f"{prefix}_N{n}") for n in WORLD}
    meta = next((r for r in res.values() if r), None)
    if meta is None:
        print("skipped memory_large_model: no results in", run_dir)
        return
    fig, (ax,) = figure("Large-model memory per GPU",
                        setup_text(meta), width=8.2)
    ax.set_ylim(0, 10.5)
    w = 0.38
    rows = []
    for j, n in enumerate(WORLD):
        r = res[n]
        if r is None:
            rows.append([n, "", "", "not run"])
            continue
        pred = (r["predicted_state_bytes"] + r["predicted_activation_bytes"]) / GB
        meas = r["peak_model_bytes"] / GB
        ax.bar(j - w / 2, pred, w * 0.94, color=PREDICTED)
        ax.text(j - w / 2, pred + 0.12, f"{pred:.1f}", ha="center", va="bottom", fontsize=9, color=INK_2)
        if ok(r):
            ax.bar(j + w / 2, meas, w * 0.94, color=BLUE)
            ax.text(j + w / 2, meas + 0.12, f"{meas:.2f}", ha="center", va="bottom", fontsize=9.5,
                    color=INK, fontweight="bold")
        else:
            ax.bar(j + w / 2, meas, w * 0.94, color="white", edgecolor=RED, hatch="////", linewidth=1.2)
            ax.text(j + w / 2, meas + 0.12, f"OOM\nat {meas:.2f}", ha="center", va="bottom", fontsize=9,
                    color=RED, fontweight="bold")
        rows.append([n, f"{pred:.3f}", f"{meas:.3f}", r["status"]])
    ax.axhline(GPU_LIMIT_GB, color=RED, linestyle="--", linewidth=1.3)
    ax.text(3.55, GPU_LIMIT_GB + 0.12, "8 GB GPU capacity", ha="right", va="bottom", fontsize=9, color=RED)
    ax.set_xticks(range(len(WORLD)), [f"N={n}" for n in WORLD])
    ax.set_xlim(-0.6, 3.6)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Memory per GPU (GB)")
    ax.legend(handles=[Patch(color=PREDICTED, label="predicted"),
                       Patch(color=BLUE, label="measured peak"),
                       Patch(facecolor="white", edgecolor=RED, hatch="////", label="OOM (memory reached)")],
              loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, out_dir, "01_memory_large_model", ["N", "predicted_GB", "measured_or_reached_GB", "status"], rows)


def plot_memory_runs(runs, out_dir):
    """
    @brief This function draws peak memory per GPU for every model size, marking out-of-memory runs.
    @param runs list of (label, run folder, file prefix)
    @param out_dir output folder
    """
    colors = [BLUE, ORANGE, AQUA]
    fig, (ax,) = figure("Memory per GPU for each model size",
                        "d=4096", width=8.6)
    ax.set_ylim(0, 10.5)
    width = 0.8 / len(runs)
    rows = []
    for i, (label, run_dir, prefix) in enumerate(runs):
        for j, n in enumerate(WORLD):
            r = load(run_dir, f"{prefix}_N{n}")
            x = j + (i - (len(runs) - 1) / 2) * width
            if ok(r):
                gb = r["peak_model_bytes"] / GB
                ax.bar(x, gb, width * 0.9, color=colors[i])
                ax.text(x, gb + 0.12, f"{gb:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)
                rows.append([label, n, f"{gb:.3f}", "ok"])
            elif r is not None:
                ax.bar(x, GPU_LIMIT_GB, width * 0.9, color="white", edgecolor=colors[i], hatch="////", linewidth=1)
                ax.text(x, GPU_LIMIT_GB / 2, "OOM", ha="center", va="center", rotation=90, fontsize=9,
                        color=INK, fontweight="bold", bbox=dict(fc="white", ec="none", pad=1))
                rows.append([label, n, "", "OOM"])
            else:
                rows.append([label, n, "", "not run"])
    ax.axhline(GPU_LIMIT_GB, color=RED, linestyle="--", linewidth=1.3)
    ax.text(3.55, GPU_LIMIT_GB + 0.12, "8 GB GPU capacity", ha="right", va="bottom", fontsize=9, color=RED)
    ax.set_xticks(range(len(WORLD)), [f"N={n}" for n in WORLD])
    ax.set_xlim(-0.6, 3.6)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Peak memory per GPU (GB)")
    ax.legend(handles=[Patch(color=c, label=l) for c, (l, _, _) in zip(colors, runs)] +
                      [Patch(facecolor="white", edgecolor=INK_2, hatch="////", label="ran out of memory")],
              loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, out_dir, "02_memory_all_models", ["model", "N", "peak_model_GB", "status"], rows)


def plot_correctness(run_dir, out_dir, tol=1e-5):
    """
    @brief This function shows the largest difference between tensor-parallel and dense results for each test.
    @param run_dir folder holding correctness_N*.json
    @param out_dir output folder
    @param tol pass threshold
    """
    res = {n: load(run_dir, f"correctness_N{n}") for n in WORLD}
    have = {n: r["results"] for n, r in res.items() if r}
    if not have:
        print("skipped 03_correctness: no correctness_N*.json in", run_dir)
        return
    tests = [(f"1 block, N={n}", have[n]["one_block"]) for n in WORLD if n in have]
    for label, key in (("2 blocks (worst N)", "two_blocks"), ("5 Adam steps (worst N)", "adam5")):
        cs = [have[n][key] for n in have]
        tests.append((label, {k: max(c[k] for c in cs) for k in ("dloss", "dgrad", "ddx") if k in cs[0]}))
    metrics = [("dloss", "Loss"), ("dgrad", "Weight gradients"), ("ddx", "Input gradient (dX)")]
    meta = next(r for r in res.values() if r)
    cfg = meta["config"]
    fig, axes = figure("Numerical correctness: tensor-parallel vs. single process",
                       f"Pass if below {tol:g}",
                       ncols=3, width=13, height=4.4)
    fig.subplots_adjust(wspace=0.14)
    ys = list(range(len(tests)))[::-1]
    lo, hi = 1e-10, 1e-4
    for i, (ax, (key, name)) in enumerate(zip(axes, metrics)):
        ax.axvspan(tol, hi, color=RED, alpha=0.07, linewidth=0)
        ax.axvline(tol, color=RED, linestyle="--", linewidth=1.2)
        for y, (_, c) in zip(ys, tests):
            if key not in c:
                ax.text(lo * 1.6, y, "not measured", va="center", fontsize=9, color=INK_2, style="italic")
            elif c[key] == 0:
                ax.plot(lo, y, marker="o", markersize=8, color=BLUE, clip_on=False)
                ax.text(lo * 1.8, y, "0 (exact)", va="center", fontsize=9, color=INK)
            else:
                ax.plot([lo / 4, c[key]], [y, y], color=BLUE, linewidth=1.2, alpha=0.4)
                ax.plot(c[key], y, marker="o", markersize=8, color=BLUE)
                ax.text(c[key] * 1.8, y, f"{c[key]:.0e}", va="center", fontsize=9, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(lo / 4, hi)
        ax.set_xticks([1e-9, 1e-7, 1e-5], ["1e-9", "1e-7", "1e-5"])
        ax.tick_params(axis="y", pad=8)
        ax.minorticks_off()
        ax.set_ylim(-0.6, len(tests) - 0.4)
        ax.set_yticks(ys, [t[0] for t in tests] if i == 0 else [""] * len(ys))
        ax.grid(axis="y", visible=False)
        ax.set_title(name)
        ax.set_xlabel("Absolute difference")
    for ax in axes:
        ax.text(tol * 1.3, len(tests) - 0.55, "fail", color=RED, fontsize=9, va="top")
    rows = [[label] + [f"{c[k]:.3e}" if k in c else "" for k, _ in metrics] +
            ["PASS" if all(c[k] <= tol for k, _ in metrics if k in c) else "FAIL"] for label, c in tests]
    save(fig, out_dir, "03_correctness", ["test", "max_abs_d_loss", "max_abs_d_grad", "max_abs_d_dX", "result"], rows)


def plot_breakdown(run_dir, prefix, out_dir):
    """
    @brief This function draws a stacked bar of compute and communication time per training step.
    @param run_dir folder of the run to show
    @param prefix result file prefix, for example headline_b1
    @param out_dir output folder
    """
    res = {n: load(run_dir, f"{prefix}_N{n}") for n in WORLD}
    ns = [n for n in WORLD if ok(res[n])]
    if not ns:
        print("skipped 04_step_breakdown: no finished runs in", run_dir)
        return
    fig, (ax,) = figure("Step-time breakdown",
                        setup_text(res[ns[0]]), width=8.2, height=4.8)
    parts = [("compute", BLUE), ("all-reduce (MPI)", RED), ("GPU↔host copies", YELLOW)]
    rows = []
    tops = []
    for j, n in enumerate(ns):
        r = res[n]
        if n == 1:
            vals = [r["step_ms"], 0, 0]
        else:
            vals = [r["compute_ms"], r["allreduce_ms"], r["d2h_ms"] + r["h2d_ms"]]
        bottom = 0
        for (label, color), v in zip(parts, vals):
            ax.bar(j, v, 0.58, bottom=bottom, color=color, edgecolor="white", linewidth=1.5)
            bottom += v
        tops.append(bottom)
        share = 0 if n == 1 else r["comm_ms"] / r["step_ms"] * 100
        rows.append([n, f"{vals[0]:.2f}", f"{vals[1]:.2f}", f"{r.get('d2h_ms', 0):.2f}", f"{r.get('h2d_ms', 0):.2f}",
                     f"{r['step_ms']:.2f}", f"{share:.1f}"])
    ax.set_ylim(0, max(tops) * 1.22)
    for j, (n, top) in enumerate(zip(ns, tops)):
        share = float(rows[j][6])
        ax.text(j, top + max(tops) * 0.02, f"{float(rows[j][5]):.0f} ms", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=INK)
        if share:
            ax.text(j, top + max(tops) * 0.075, f"comm {share:.0f}%", ha="center", va="bottom",
                    fontsize=9, color=RED)
    ax.set_xticks(range(len(ns)), [f"N={n}" for n in ns])
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Time per training step (ms)")
    ax.legend(handles=[Patch(color=c, label=l) for l, c in parts], loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, out_dir, "04_step_breakdown",
         ["N", "compute_ms", "allreduce_ms", "d2h_ms", "h2d_ms", "step_ms", "comm_share_pct"], rows)


def plot_strong_scaling(run_dir, prefix, out_dir):
    """
    @brief This function draws strong scaling: speedup and parallel efficiency for a fixed model.
    @param run_dir folder of the run to show
    @param prefix result file prefix, for example headline_b1
    @param out_dir output folder
    """
    res = {n: load(run_dir, f"{prefix}_N{n}") for n in WORLD}
    if not ok(res[1]):
        print("skipped 05_strong_scaling: no finished N=1 run in", run_dir)
        return
    ns = [n for n in WORLD if ok(res[n])]
    sp = [res[1]["step_ms"] / res[n]["step_ms"] for n in ns]
    eff = [s / n * 100 for s, n in zip(sp, ns)]
    fig, (a1, a2) = figure("Strong scaling",
                           setup_text(res[1]), ncols=2, width=12.5)
    a1.plot(WORLD, WORLD, linestyle="--", color=INK_2, linewidth=1.3)
    a1.text(5.6, 5.6 * 1.08, "ideal", rotation=0, fontsize=9, color=INK_2, ha="right")
    a1.plot(ns, sp, marker="o", color=BLUE)
    for n, s in zip(ns, sp):
        a1.annotate(f"{s:.2f}×", (n, s), textcoords="offset points", xytext=(0, -16), ha="center",
                    fontsize=10, color=INK, fontweight="bold")
    n_axis(a1)
    a1.set_yscale("log", base=2)
    a1.set_yticks([1, 2, 4, 8], ["1×", "2×", "4×", "8×"])
    a1.minorticks_off()
    a1.set_ylim(0.7, 10)
    a1.set_ylabel("Speedup over 1 GPU")
    a1.set_title("Speedup")

    xs = range(len(ns))
    a2.bar(xs, eff, 0.58, color=BLUE)
    a2.set_ylim(0, 115)
    bar_labels(a2, xs, eff, [f"{e:.0f}%" for e in eff])
    a2.set_xticks(list(xs), [f"N={n}" for n in ns])
    a2.set_xlabel("Number of GPUs")
    a2.set_ylabel("Parallel efficiency (%)")
    a2.set_title("Parallel efficiency")
    save(fig, out_dir, "05_strong_scaling", ["N", "step_ms", "speedup", "efficiency_pct"],
         [[n, f"{res[n]['step_ms']:.2f}", f"{s:.3f}", f"{e:.1f}"] for n, s, e in zip(ns, sp, eff)])


def plot_weak_scaling(run_dir, out_dir):
    """
    @brief This function draws weak scaling, where the model width grows with N.
    @param run_dir folder holding scaling_N*.json
    @param out_dir output folder
    """
    res = {n: load(run_dir, f"scaling_N{n}") for n in WORLD}
    if not ok(res[1]):
        print("skipped 06_weak_scaling: no finished N=1 run in", run_dir)
        return
    ns = [n for n in WORLD if ok(res[n])]
    B = res[1]["config"]["batch"]
    thr = [n * B / res[n]["step_ms"] * 1e3 for n in ns]  # model-size-normalized samples/s
    ideal = [n * B / res[1]["step_ms"] * 1e3 for n in WORLD]
    eff = [res[1]["step_ms"] / res[n]["step_ms"] * 100 for n in ns]
    widths = {n: f"h={res[n]['config']['h'] if res[n] else 4096 * n}" for n in WORLD}
    meta = dict(res[1], config=dict(res[1]["config"], h="4096·N"))
    fig, (a1, a2) = figure("Weak scaling",
                           setup_text(meta), ncols=2, width=12.5)
    a1.plot(WORLD, ideal, linestyle="--", color=INK_2, linewidth=1.3)
    a1.text(8, ideal[-1], "  ideal", va="center", fontsize=9, color=INK_2)
    a1.plot(ns, thr, marker="o", color=BLUE)
    a1.text(8, thr[-1], "  measured", va="center", fontsize=9, color=BLUE, fontweight="bold")
    n_axis(a1, "Number of GPUs (N) and model width", widths)
    a1.set_xlim(0.75, 14)
    a1.set_ylim(0, max(ideal) * 1.1)
    a1.set_ylabel("Throughput × N (samples/s)")
    a1.set_title("Throughput (N × samples/s)")

    a2.axhline(100, linestyle="--", color=INK_2, linewidth=1.3)
    a2.plot(ns, eff, marker="o", color=BLUE)
    for n, e in zip(ns, eff):
        a2.annotate(f"{e:.0f}%", (n, e), textcoords="offset points", xytext=(0, 9 if n == 1 else -18),
                    ha="center", fontsize=10, color=INK, fontweight="bold")
    n_axis(a2, "Number of GPUs (N) and model width", widths)
    a2.set_ylim(0, 112)
    a2.set_ylabel("Weak-scaling efficiency (%)")
    a2.set_title("Weak-scaling efficiency")
    save(fig, out_dir, "06_weak_scaling", ["N", "h", "step_ms", "normalized_samples_per_s", "ideal", "efficiency_pct"],
         [[n, res[n]["config"]["h"], f"{res[n]['step_ms']:.2f}", f"{t:.1f}", f"{ideal[WORLD.index(n)]:.1f}",
           f"{e:.1f}"] for n, t, e in zip(ns, thr, eff)])


def plot_utilization(run_dir, prefix, out_dir, peak_tflops):
    """
    @brief This function draws achieved compute throughput and GPU busy time.
    @param run_dir folder of the run to show
    @param prefix result file prefix, for example headline_b1
    @param out_dir output folder
    @param peak_tflops peak FP32 TFLOPS of the GPU
    """
    res = {n: load(run_dir, f"{prefix}_N{n}") for n in WORLD}
    ns = [n for n in WORLD if ok(res[n])]
    if not ns:
        print("skipped 07_utilization: no finished runs in", run_dir)
        return
    tf = [res[n]["flops_per_gpu_step"] / (res[n]["step_ms"] / 1e3) / 1e12 for n in ns]
    pct = [t / peak_tflops * 100 for t in tf]
    busy = [res[n].get("gpu_util_pct") or 0 for n in ns]
    fig, (a1, a2) = figure("GPU utilization",
                           setup_text(res[ns[0]]), ncols=2, width=12.5)
    xs = list(range(len(ns)))
    a1.bar(xs, pct, 0.58, color=BLUE)
    a1.set_ylim(0, max(pct) * 1.3)
    bar_labels(a1, xs, pct, [f"{p:.1f}%\n({t:.2f} TFLOPS)" for p, t in zip(pct, tf)])
    a1.set_xticks(xs, [f"N={n}" for n in ns])
    a1.set_xlabel("Number of GPUs")
    a1.set_ylabel(f"% of peak ({peak_tflops:g} TFLOPS)")
    a1.set_title("Compute utilization")
    a2.bar(xs, busy, 0.58, color=ORANGE)
    a2.set_ylim(0, 100)
    bar_labels(a2, xs, busy, [f"{b:.0f}%" for b in busy])
    a2.set_xticks(xs, [f"N={n}" for n in ns])
    a2.set_xlabel("Number of GPUs")
    a2.set_ylabel("nvidia-smi GPU util (%)")
    a2.set_title("GPU busy time")
    save(fig, out_dir, "07_utilization", ["N", "achieved_TFLOPS", "pct_of_peak", "gpu_busy_pct"],
         [[n, f"{t:.3f}", f"{p:.2f}", f"{b:.1f}"] for n, t, p, b in zip(ns, tf, pct, busy)])


def read_allreduce(run_dir):
    """
    @brief This function reads the all-reduce measurements of one run.
    @param run_dir folder holding allreduce_N*.json, or None
    @return (dict N -> list of rows sorted by bytes, bytes of the message the model sends)
    """
    data, model_bytes = {}, None
    for n in (2, 4, 8):
        r = load(run_dir, f"allreduce_N{n}") if run_dir else None
        if r:
            data[n] = sorted(r["rows"], key=lambda row: row["bytes"])
            model_bytes = r["config"]["batch"] * r["config"]["d"] * 4
    return data, model_bytes


def size_axis(ax, model_bytes):
    """
    @brief This function sets a log message-size axis with readable units and marks the model's message.
    @param ax axes
    @param model_bytes size of the all-reduce the model sends
    """
    ax.set_xscale("log")
    ticks = [4, 1024, 65536, 2**20, 16 * 2**20]
    ax.set_xticks(ticks, ["4 B", "1 KB", "64 KB", "1 MB", "16 MB"])
    ax.minorticks_off()
    ax.set_xlabel("Message size")
    if model_bytes:
        ax.axvline(model_bytes, color=VIOLET, linestyle=":", linewidth=1.5)


def plot_allreduce_latency(run_dir, out_dir):
    """
    @brief This function compares MPI-only and end-to-end all-reduce latency, one panel per N.
    @param run_dir folder holding allreduce_N*.json measured across machines
    @param out_dir output folder
    """
    data, model_bytes = read_allreduce(run_dir)
    if not data:
        print("skipped 08_allreduce_latency: no allreduce_N*.json in", run_dir)
        return
    fig, axes = figure("All-reduce latency (GPU path)",
                       "1 Gb Ethernet",
                       ncols=len(data), width=4.6 * len(data), height=4.4)
    rows = []
    ymax = max(row["e2e_us"] for rs in data.values() for row in rs) * 2
    ymin = min(row["mpi_us"] for rs in data.values() for row in rs) / 2
    for ax, (n, rs) in zip(axes, sorted(data.items())):
        xs = [row["bytes"] for row in rs]
        mpi = [row["mpi_us"] for row in rs]
        e2e = [row["e2e_us"] for row in rs]
        ax.fill_between(xs, mpi, e2e, color=N_COLOR[n], alpha=0.15, linewidth=0)
        ax.plot(xs, e2e, marker="o", color=N_COLOR[n], label="end-to-end")
        ax.plot(xs, mpi, marker="o", markersize=5, color=N_COLOR[n], linestyle="--", alpha=0.8,
                label="MPI only")
        size_axis(ax, model_bytes)
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"N = {n} GPUs")
        rows += [[n, row["bytes"], f"{row['mpi_us']:.1f}", f"{row['e2e_us']:.1f}"] for row in rs]
    axes[0].set_ylabel("Latency (µs, log scale)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    if model_bytes:
        axes[-1].text(model_bytes / 1.3, ymin * 1.5, f"B×d×4\n({model_bytes / 2**20:.0f} MB)",
                      ha="right", va="bottom", fontsize=8.5, color=VIOLET)
    save(fig, out_dir, "08_allreduce_latency", ["N", "bytes", "mpi_only_us", "end_to_end_us"], rows)


def plot_allreduce_bandwidth(run_dir, out_dir):
    """
    @brief This function draws end-to-end all-reduce bandwidth on a linear scale against the network limit.
    @param run_dir folder holding allreduce_N*.json measured across machines
    @param out_dir output folder
    """
    data, model_bytes = read_allreduce(run_dir)
    if not data:
        print("skipped 09_allreduce_bandwidth: no allreduce_N*.json in", run_dir)
        return
    fig, (ax,) = figure("End-to-end all-reduce bandwidth",
                        "1 Gb Ethernet", width=8.6)
    rows = []
    for n, rs in sorted(data.items()):
        xs = [row["bytes"] for row in rs]
        bw = [row["e2e_algbw_gbs"] for row in rs]
        ax.plot(xs, bw, marker="o", color=N_COLOR[n])
        ax.text(xs[-1] * 1.4, bw[-1], f"N={n}", va="center", fontsize=10, color=N_COLOR[n], fontweight="bold")
        rows += [[n, row["bytes"], f"{row['e2e_us']:.1f}", f"{b:.5f}"] for row, b in zip(rs, bw)]
    ax.axhline(ETHERNET_GBPS, color=RED, linestyle="--", linewidth=1.3)
    ax.text(5, ETHERNET_GBPS + 0.003, "1 Gb Ethernet (0.125 GB/s)", fontsize=9, color=RED, va="bottom")
    size_axis(ax, model_bytes)
    ax.set_xlim(2.5, 16 * 2**20 * 4)
    ax.set_ylim(0, ETHERNET_GBPS * 1.15)
    ax.set_ylabel("Achieved bandwidth (GB/s)")
    if model_bytes:
        ax.text(model_bytes / 1.2, 0.004, f"B×d×4\n({model_bytes / 2**20:.0f} MB)", ha="right", va="bottom", fontsize=8.5, color=VIOLET)
    save(fig, out_dir, "09_allreduce_bandwidth", ["N", "bytes", "end_to_end_us", "bandwidth_GBps"], rows)


def plot_allreduce_cpu_vs_gpu(net_dir, local_dir, out_dir):
    """
    @brief This function compares end-to-end all-reduce bandwidth across machines and on one machine.
    @param net_dir folder with allreduce_N*.json measured across machines (GPU runs)
    @param local_dir folder with allreduce_N*.json measured on one machine (CPU runs)
    @param out_dir output folder
    """
    net, model_bytes = read_allreduce(net_dir)
    local, _ = read_allreduce(local_dir)
    if not net or not local:
        print("skipped 10_allreduce_cpu_vs_gpu: need both GPU and CPU all-reduce results")
        return
    ratios = []
    for n in net:
        if n in local:
            g = next((row for row in net[n] if row["bytes"] == model_bytes), None)
            c = next((row for row in local[n] if row["bytes"] == model_bytes), None)
            if g and c:
                ratios.append(g["e2e_us"] / c["e2e_us"])
    if ratios:
        print(f"model-size all-reduce is {min(ratios):.0f}-{max(ratios):.0f}x slower across machines than on one")
    fig, (ax,) = figure("End-to-end all-reduce bandwidth: CPU vs. GPU runs",
                        "CPU on one machine, GPU across machines", width=9.4, height=5)
    rows = []
    for name, data, ls, mk in (("one machine (CPU)", local, "--", "^"), ("across machines (GPU)", net, "-", "o")):
        for n, rs in sorted(data.items()):
            xs = [row["bytes"] for row in rs]
            bw = [row["e2e_algbw_gbs"] for row in rs]
            ax.plot(xs, bw, linestyle=ls, marker=mk, markersize=6, color=N_COLOR[n])
            rows += [[name, n, row["bytes"], f"{row['e2e_us']:.1f}", f"{b:.6f}"] for row, b in zip(rs, bw)]
    ax.axhline(ETHERNET_GBPS, color=RED, linestyle="--", linewidth=1.1, alpha=0.8)
    ax.text(5, ETHERNET_GBPS * 1.15, "1 Gb Ethernet", fontsize=9, color=RED)
    size_axis(ax, model_bytes)
    ax.set_yscale("log")
    ax.set_ylabel("Achieved bandwidth (GB/s, log scale)")
    ax.legend(handles=[plt.Line2D([], [], color=N_COLOR[n], label=f"N={n}") for n in (2, 4, 8)] +
                      [plt.Line2D([], [], color=INK_2, linestyle="--", marker="^", label="CPU"),
                       plt.Line2D([], [], color=INK_2, marker="o", label="GPU")],
              loc="upper left", bbox_to_anchor=(1.01, 1), handlelength=3.5)
    save(fig, out_dir, "10_allreduce_cpu_vs_gpu", ["where", "N", "bytes", "end_to_end_us", "bandwidth_GBps"], rows)


def main():
    """
    @brief This function reads the arguments and draws every graph.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(REPO, "results"),
                    help="folder holding the tp_bench_* run folders (default: needle-tp/results)")
    ap.add_argument("--out", default=None, help="output folder (default: <results>/plots)")
    ap.add_argument("--small", default="tp_bench_full_small", help="main run: 1 block, h = 8192")
    ap.add_argument("--large1", default="tp_bench_large_block_1", help="1 block, h = 16384")
    ap.add_argument("--large4", default="tp_bench_large_block_4", help="large model: 4 blocks, h = 16384")
    ap.add_argument("--local", default="tp_bench_cpu_full", help="CPU run with all ranks on one machine")
    ap.add_argument("--peak-tflops", type=float, default=10.0, help="GPU peak FP32 TFLOPS (RTX 2080: 10)")
    a = ap.parse_args()
    missing = [d for d in (a.small, a.large1, a.large4) if not os.path.isdir(os.path.join(a.results, d))]
    if missing:
        raise SystemExit(f"Run folders not found in {os.path.abspath(a.results)}: {', '.join(missing)}. "
                         "Pass --results with the folder that holds them.")
    out = a.out or os.path.join(a.results, "plots")
    os.makedirs(out, exist_ok=True)
    p = lambda name: os.path.join(a.results, name)
    local = p(a.local) if os.path.isdir(p(a.local)) else None
    style()

    plot_memory_large(p(a.large4), "headline_b4", out)
    plot_memory_runs([("1 block, h=8192", p(a.small), "headline_b1"),
                      ("1 block, h=16384", p(a.large1), "headline_b1"),
                      ("4 blocks, h=16384", p(a.large4), "headline_b4")], out)
    plot_correctness(p(a.small), out)
    plot_breakdown(p(a.small), "headline_b1", out)
    plot_strong_scaling(p(a.small), "headline_b1", out)
    plot_weak_scaling(p(a.small), out)
    plot_utilization(p(a.small), "headline_b1", out, a.peak_tflops)
    plot_allreduce_latency(p(a.small), out)
    plot_allreduce_bandwidth(p(a.small), out)
    plot_allreduce_cpu_vs_gpu(p(a.small), local, out)
    print("graphs in", out)


if __name__ == "__main__":
    main()
