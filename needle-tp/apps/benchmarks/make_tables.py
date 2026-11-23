"""Turn results/tp_bench/*.json (written by bench_tp.py) into the report tables.

    python benchmarks/make_tables.py                      # reads results/tp_bench, writes tables.md + bench.json
    python benchmarks/make_tables.py --dir results/tp_bench --peak-tflops 10

Cells whose run has not been done yet show "—". Columns that are pure arithmetic (params per rank, predicted
memory, FLOPs per GPU) are filled from the formulas even before any run.
"""
import argparse
import json
import os

GB = 1e9
WORLD = (1, 2, 4, 8)


def load(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def f(x, nd=1, dash="—"):
    return dash if x is None else f"{x:,.{nd}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def predicted(d, h, blocks, B, n):
    params = blocks * (2 * d * h // n + h // n + d)
    return params, 16 * params, 2 * 4 * blocks * (3 * B * h // n + 3 * B * d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/tp_bench")
    ap.add_argument("--peak-tflops", type=float, default=10.0)
    ap.add_argument("--d", type=int, default=4096)
    ap.add_argument("--h", type=int, default=16384)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--gpu-mem-gb", type=float, default=8 * 2**30 / GB, help="8 GB card = 8 GiB = 8.59e9 B")
    a = ap.parse_args()
    D, H, K, B = a.d, a.h, a.blocks, a.batch

    head = {n: load(a.dir, f"headline_b{K}_N{n}.json") for n in WORLD}
    head1 = {n: load(a.dir, f"headline_b1_N{n}.json") for n in WORLD}
    scal = {n: load(a.dir, f"scaling_N{n}.json") for n in WORLD}
    corr = {n: load(a.dir, f"correctness_N{n}.json") for n in WORLD}
    allr = {n: load(a.dir, f"allreduce_N{n}.json") for n in (2, 4, 8)}
    ok = lambda r: r is not None and r.get("status") == "ok"

    meta = next((r for r in list(head.values()) + list(head1.values()) + list(scal.values()) if r), None)
    md = ["# Tensor-parallel benchmark tables", ""]
    if meta:
        md.append(f"Hardware: {meta['gpu']} ({meta['device']}); config d={D}, h={H}, blocks={K}, batch={B}, "
                  f"Adam, fp32. Machines in the N-rank runs: " +
                  ", ".join(f"N={n}: {r['machines']}" for n, r in sorted(head.items()) if r))
        md.append("")

    # ---------------------------------------------------------------- Memory
    rows = []
    for n in WORLD:
        p, st, ac = predicted(D, H, K, B, n)
        r = head[n]
        if r is None:
            meas, fits = "—", "—"
        elif r["status"] == "OOM":
            meas, fits = f"OOM (reached {r['peak_total_bytes'] / GB:.2f} GB)", "No"
        else:
            tot, mod = r["peak_total_bytes"] / GB, r["peak_model_bytes"] / GB
            meas = f"{mod:.2f} (total {tot:.2f} incl. CUDA context)"
            cap = (r.get("gpu_total_bytes") or a.gpu_mem_gb * GB) / GB
            fits = "n/a (cpu run)" if r["device"] == "cpu" else ("Yes" if tot <= cap else "No")
        rows.append([n, f"{(r['params_per_rank'] if r and 'params_per_rank' in r else p) / 1e6:,.1f} M",
                     f"{(st + ac) / GB:.2f}", meas, fits])
    md += ["## Memory", "", table(["N", "Params per rank", "Predicted per rank (GB)",
                                    "Measured peak per rank (GB)", "Fits 8 GB GPU?"], rows),
           "", "Predicted = 16 B/param (weights, grads, Adam m and v) + activation estimate (Needle keeps `.grad` on "
           "every graph node, so saved activations count twice). GB = 1e9 bytes; an 8 GB card has 8.59e9 B. "
           "Measured = NVML peak of this process minus the CUDA-context baseline." +
           (f" Memory source: {meta['mem_mode']} (`nvml-device` = whole GPU, so other users' memory would be included)."
            if meta and meta["device"] == "cuda" else ""), ""]

    # ---------------------------------------------------------------- Correctness
    def cget(n, key):
        r = corr[n]
        return r["results"][key] if r else None

    def crow(label, c):
        if c is None:
            return [label, "—", "—", "—", "—"]
        f_ = lambda k: f"{c[k]:.1e}" if k in c else "n/a"
        return [label, f_("dloss"), f_("dgrad"), f_("ddx"), "PASS" if c["pass"] else "FAIL"]

    rows = []
    for n in WORLD:
        c = cget(n, "one_block")
        rows.append(["One TPMLP block", n] + crow("", c)[1:])
    for label, key in (("Two stacked blocks", "two_blocks"), ("5 Adam steps (worst step)", "adam5")):
        cs = [cget(n, key) for n in WORLD]
        if all(c is None for c in cs):
            rows.append([label, "1 / 2 / 4 / 8", "—", "—", "—", "—"])
            continue
        have = [c for c in cs if c is not None]
        worst = {k: max(c[k] for c in have if k in c) for k in ("dloss", "dgrad", "ddx") if any(k in c for c in have)}
        worst["pass"] = all(c["pass"] for c in have)
        got = "/".join(str(n) for n, c in zip(WORLD, cs) if c is not None)
        rows.append([label, got] + crow("", worst)[1:])
    md += ["## Correctness", "", table(["Test", "N", "max abs Δ loss", "max abs Δ gradient", "max abs Δ dX",
                                        "Pass (≤ 1e-5)"], rows)]
    cm = next((r for r in corr.values() if r), None)
    if cm:
        cfg = cm["config"]
        rel = [cget(n, k)["dgrad_rel"] for n in WORLD for k in ("one_block", "two_blocks", "adam5") if corr[n]]
        md += ["", f"Run on {cm['device']} ({cm['gpu']}), d={cfg['d']}, h={cfg['h']}, B={cfg['batch']}, loss = "
               f"mean squared error. Worst gradient error relative to the largest gradient entry: {max(rel):.1e}. "
               "Worst case over all ranks. Δ = TP result vs. the dense model built from the same weights."]
    md.append("")

    # ---------------------------------------------------------------- Throughput
    base_b1 = head1[1] if ok(head1[1]) else None
    if ok(head[1]):
        base_ms, base_note = head[1]["step_ms"], "measured"
    else:
        base_ms, base_note = None, None
    rows = []
    for n in WORLD:
        r = head[n]
        if r is not None and r["status"] == "OOM":
            rows.append([n, r["machines"], B, "OOM", "—", "—", "—"])
        elif ok(r):
            sp = base_ms / r["step_ms"] if base_ms else None
            rows.append([n, r["machines"], B, f(r["step_ms"]), f(B / r["step_ms"] * 1e3), f(sp, 2),
                         f(sp / n * 100, 0) + " %" if sp else "—"])
        else:
            rows.append([n, n if n > 1 else 1, B, "—", "—", "—", "—"])
    md += ["## Throughput", "", table(["N", "Machines", "Batch", "Median step (ms)", "Samples/s",
                                       "Speedup vs. N=1", "Parallel efficiency"], rows)]
    if base_ms is not None:
        md += ["", f"Speedup baseline: {f(base_ms)} ms, measured at N=1."]
    else:
        md += ["", "Speedup and efficiency are unavailable: no successful measured N=1 baseline."]
    rows = []
    for n in WORLD:
        r = head1[n]
        if ok(r):
            sp = base_b1["step_ms"] / r["step_ms"] if base_b1 else None
            rows.append([n, r["machines"], B, f(r["step_ms"]), f(B / r["step_ms"] * 1e3), f(sp, 2),
                         f(sp / n * 100, 0) + " %" if sp else "—"])
        else:
            rows.append([n, "—", B, "—", "—", "—", "—"])
    md += [
    "",
    "Supplement: one-block results. Successful-run timings are measured; "
    "speedup requires a successful one-block N=1 baseline.",
    "",
    table(
        ["N", "Machines", "Batch", "Median step (ms)", "Samples/s",
         "Speedup vs. N=1", "Parallel efficiency"],
        rows,
    ),
    "",
    ]

    # ---------------------------------------------------------------- Communication
    rows = []
    for n in WORLD:
        r = head[n]
        if n == 1:
            src = head[1] if ok(head[1]) else base_b1
            mult = 1 if ok(head[1]) else K
            if src:
                tot = src["step_ms"] * mult
                rows.append([1, f(src["compute_ms"] * mult) + ("" if ok(head[1]) else " *"), 0, 0, 0,
                             f(tot) + ("" if ok(head[1]) else " *"), 0, 0, "n/a"])
                continue
        if ok(r):
            rows.append([n, f(r["compute_ms"]), f(r["d2h_ms"], 2), f(r["allreduce_ms"], 2), f(r["h2d_ms"], 2),
                         f(r["step_ms"]), f(r["comm_ms"] / r["step_ms"] * 100) + " %", f(r["calls_per_step"], 0),
                         f"{r['bytes_per_call'] / 1e6:.2f} MB (= {r['bytes_per_call'] / 4 / r['config']['d']:.0f} × {r['config']['d']} × 4)"])
        else:
            rows.append([n, "—", "—", "—", "—", "—", "—", "—", "B × 4096 × 4"])
    md += ["## Communication time", "", table(["N", "Compute (ms)", "GPU→host (ms)", "All-reduce (ms)",
                                              "Host→GPU (ms)", "Step total (ms)", "Comm share (%)",
                                              "All-reduces per step", "Bytes per all-reduce"], rows),
           "", "Medians over timed steps, slowest rank. Compute = step − comm (queued kernels are charged to compute "
           "because all_reduce is preceded by a device sync). All-reduces per step are measured: block 0's input "
           "does not require a gradient, so autograd detaches its `f` and only 2·blocks − 1 fire "
           f"({2 * K - 1} for {K} blocks, not {2 * K}). `* ` = N=1 row estimated from the 1-block run.", ""]

    # ---------------------------------------------------------------- Scaling
    s1 = scal[1]["step_ms"] if ok(scal[1]) else None
    rows = []
    for n in WORLD:
        r = scal[n]
        h = 4096 * n
        if ok(r) and s1:
            raw = B / r["step_ms"] * 1e3
            norm = raw * n
            ideal = n * (B / s1 * 1e3)
            rows.append([n, h, f"{norm:,.1f}", f"{raw:,.1f}", f"{ideal:,.1f}", f"{norm / ideal * 100:.0f} %"])
        else:
            rows.append([n, h, "—", "—", "—", "100 %" if n == 1 and s1 else "—"])
    md += ["## Scaling (weak: h = 4096·N, per-rank work constant)", "",
           table(["N", "h", "Throughput (samples/s × N, model-size-normalized)", "Raw samples/s",
                  "Ideal (N × throughput₁)", "Efficiency"], rows),
           "", "Raw samples/s is flat under perfect weak scaling, because the model grows N-fold while each rank's work "
           "stays constant. The normalized column multiplies by N (model FLOPs per sample grow ∝ h), which makes "
           "`Ideal = N × throughput₁` correct, and efficiency = raw₍N₎ / raw₍1₎ = step₁ / step_N.", ""]

    # ---------------------------------------------------------------- All-reduce latency
    labels = None
    for n in (2, 4, 8):
        if allr[n]:
            labels = [(r["label"], r["floats"]) for r in allr[n]["rows"]]
    labels = labels or [("4 B", 1), ("1 KB", 256), ("64 KB", 16384), ("1 MB", 262144), ("16 MB", 4194304),
                        ("B × d × 4", B * D)]

    def lat_table(kind, title):
        rows = []
        for label, nf in labels:
            lat, bw = [], []
            for n in (2, 4, 8):
                row = next((r for r in allr[n]["rows"] if r["floats"] == nf), None) if allr[n] else None
                lat.append(f(row[f"{kind}_us"], 1) if row else "—")
                bw.append(f(row[f"{kind}_algbw_gbs"], 3) if row else "—")
            rows.append([label.replace("x", "×"), f"{nf:,}"] + lat + bw)
        return [f"### {title}", "", table(["Message size", "Floats", "N=2 latency (µs)", "N=4 latency (µs)",
                                           "N=8 latency (µs)", "N=2 algo BW (GB/s)", "N=4 algo BW (GB/s)",
                                           "N=8 algo BW (GB/s)"], rows), ""]

    md += ["## All-reduce latency", ""]
    md += lat_table("mpi", "Host-side MPI Allreduce only (network + MPI)")
    md += lat_table("e2e", "End-to-end `distributed.all_reduce` on a device array (GPU→host + MPI + host→GPU)")
    md += ["Algo BW = message bytes / latency. Median over 30–200 iterations after 10 warm-ups, slowest rank. "
           "The last row is the message the model actually sends.", ""]

    # ---------------------------------------------------------------- Utilization
    rows = []
    for n in WORLD:
        r = head[n]
        fl = 12 * B * D * H * K / n
        if ok(r):
            fl = r["flops_per_gpu_step"]  # 1-block run for N=1 (the 4-block model does not fit)
            t = r["step_ms"] / 1e3
            tf = fl / t / 1e12
            rows.append([n, f"{fl / 1e9:,.0f} GFLOP",
                         f"{t:.3f}", f"{tf:.3f}", f"{tf / a.peak_tflops * 100:.1f} %", f(r.get("gpu_util_pct"), 0)])
        else:
            rows.append([n, f"{fl / 1e9:,.0f} GFLOP", "—", "—", "—", "—"])
    md += ["## Utilization", "", table(["N", "FLOPs per GPU per step", "Step time (s)", "Achieved TFLOPS",
                                        f"% of peak ({a.peak_tflops:g} TFLOPS)", "GPU util (%)"], rows),
           "", "FLOPs = 12·B·d·h per block (fwd + bwd ≈ 3 × fwd), ÷ N, × blocks. Achieved TFLOPS counts the whole "
           "step including communication. GPU util is NVML's utilization.gpu (fraction of time a kernel was running, "
"not SM occupancy), averaged over the timed steps. All rows use the selected headline configuration.", ""]

    text = "\n".join(md)
    with open(os.path.join(a.dir, "tables.md"), "w") as fh:
        fh.write(text)

    # bench.json in the format the report notebook reads
    bench = []
    for r in list(head.values()) + list(head1.values()) + list(scal.values()):
        if ok(r):
            bench.append({"world_size": r["world_size"], "hidden": r["config"]["h"],
                          "blocks": r["config"]["blocks"], "params_per_rank": r["params_per_rank"],
                          "peak_mem_mb": r["peak_model_bytes"] / 2**20, "step_ms": r["step_ms"],
                          "allreduce_ms": r["allreduce_ms"], "host_copy_ms": r["d2h_ms"] + r["h2d_ms"]})
    with open(os.path.join(a.dir, "bench.json"), "w") as fh:
        json.dump(bench, fh, indent=1)
    print(text)


if __name__ == "__main__":
    main()
