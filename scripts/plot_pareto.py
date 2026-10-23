#!/usr/bin/env python3
"""Pareto curves for conventional beam search vs directional search (DBS).

Reads the joined CSV(s) of sbatch/bench_adaptive_termination.sbatch
(adaptive_<dataset>.csv, from scripts/adaptive_summary.py) and draws, per
dataset and search k, one figure whose panels each put recall on x:

  QPS on device / QPS on host   higher is better, from run_query
  nodes expanded per query      lower is better, from distortion_test --pops
  vectors read per query        lower is better; one per pop for DBS, one per
                                appended neighbor for conventional search, so
                                this is the panel that compares their work

Recall is on a logit axis by default (--linear-recall for a plain one): the
methods separate above 0.9, where a linear axis turns every curve into a cliff.

Each method (and, for DBS, each k_ranks) gets ONE curve: the Pareto frontier
of all its runs, beam-width points and Adaptive Beam Search points (beam width
x gamma) together, so the curve is the best that method does at each recall
whichever stop rule gets it there. Conventional search runs the beam-width
rule only.

Also writes pareto_<dataset>.csv (the points on every curve, the figures'
table view) and prints, per placement and target recall, each method's best
QPS, nodes expanded and vectors read, with the configuration that gets there.

    python3 scripts/plot_pareto.py adaptive_termination/adaptive_deep10M.csv --out adaptive_termination
    python3 scripts/plot_pareto.py adaptive_termination/adaptive_*.csv --out adaptive_termination --format png
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

# Same names and hues as plot_pops.py (Fig. 10), so a method keeps its color
# across figures. Conventional search is the reference, not a fourth hue: with
# every curve on one panel only three categorical hues stay distinguishable,
# so it is drawn in neutral ink and told apart by its dashed line and marker.
METHOD_ORDER = ["conventional", "lsh", "rabitq", "pq"]
METHOD_LABEL = {"conventional": "Conventional", "lsh": "CPH", "pq": "PQ", "rabitq": "RaBitQ"}
COLOR = {"conventional": "#4a4a48", "lsh": "#2a78d6", "pq": "#eb6834", "rabitq": "#1baf7a"}
MARKER = {"conventional": "D", "lsh": "o", "pq": "s", "rabitq": "^"}
# A second k_ranks of the same method changes line style, not hue.
K_DASH = [(0, ()), (0, (4, 1.6)), (0, (1.2, 1.6))]

INK = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"

PANELS = [  # (key, y label, higher is better)
    ("qps_device", "QPS (device)", True),
    ("qps_host", "QPS (host, over PCIe)", True),
    ("mean_pops", "nodes expanded / query", False),
    ("mean_vectors_fetched", "vectors read / query", False),
]


def fnum(x):
    return None if x in (None, "") else float(x)


def read_rows(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            for r in csv.DictReader(f):
                for c in ("recall", "qps", "recall_pops", "mean_pops", "mean_vectors_fetched",
                          "gamma"):
                    r[c] = fnum(r.get(c))
                r["k_ranks"] = int(r["k_ranks"])
                r["topk"] = int(r["topk"])
                r["beam_width"] = int(r["beam_width"])
                rows.append(r)
    return rows


def series_key(r):
    return (r["method"], r["k_ranks"])


def series_label(key, multi_k):
    m, k = key
    if m == "conventional" or not multi_k.get(m):
        return METHOD_LABEL[m]
    return f"{METHOD_LABEL[m]} r={k}"


def config(r):
    if r["stop"] == "beam":
        return f"bw={r['beam_width']}"
    return f"L={r['beam_width']} g={r['gamma']:g}"


def frontier(points, higher_better):
    """Pareto front of (recall, value, row): no other point has both higher
    recall and a better value. Returned in ascending recall."""
    pts = sorted((p for p in points if p[0] is not None and p[1] is not None),
                 key=lambda p: (-p[0], -p[1] if higher_better else p[1]))
    front, best = [], None
    for rec, val, row in pts:
        if best is None or (val > best if higher_better else val < best):
            front.append((rec, val, row))
            best = val
    return front[::-1]


def panel_points(rows, panel):
    """(recall, value, row) per run. Pops and vectors read do not depend on
    placement (the summary checks), so they come from one placement only."""
    if panel in ("qps_device", "qps_host"):
        place = panel.split("_")[1]
        return [(r["recall"], r["qps"], r) for r in rows if r["placement"] == place]
    places = {r["placement"] for r in rows}
    place = "device" if "device" in places else next(iter(places), None)
    return [(r["recall_pops"], r[panel], r) for r in rows if r["placement"] == place]


def si(x):
    for div, suf in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if x >= div:
            return f"{x / div:.3g}{suf}"
    return f"{x:.3g}"


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=7, length=2)
    ax.set_yscale("log")


def plot_cell(ds, topk, rows, out_dir, fmt, min_recall, linear):
    keys = sorted({series_key(r) for r in rows},
                  key=lambda k: (METHOD_ORDER.index(k[0]), k[1]))
    multi_k = defaultdict(bool)
    ks = defaultdict(list)
    for m, k in keys:
        ks[m].append(k)
    for m, lst in ks.items():
        multi_k[m] = len(lst) > 1

    panels = [p for p in PANELS if any(v is not None for _, v, _ in panel_points(rows, p[0]))]
    if not panels:
        return []
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.3),
                             squeeze=False, constrained_layout=True)
    fronts = []
    lo, hi = 1.0, 0.0
    for ax, (panel, ylabel, higher) in zip(axes[0], panels):
        style_axes(ax)
        for key in keys:
            sub = [r for r in rows if series_key(r) == key]
            front = frontier(panel_points(sub, panel), higher)
            if not front:
                continue
            m, k = key
            dash = K_DASH[ks[m].index(k) % len(K_DASH)]
            if m == "conventional":
                dash = (0, (5, 2))
            # A recall of exactly 1 has no place on a logit axis; pull it in.
            xs = [min(p[0], 1 - 1e-5) if not linear else 100 * p[0] for p in front]
            ys = [p[1] for p in front]
            # Markers carry identity beside the hue and dash; thin them so a
            # dense frontier stays a line.
            ax.plot(xs, ys, color=COLOR[m], linestyle=dash, linewidth=1.6,
                    marker=MARKER[m], markersize=4, markevery=0.07, markeredgecolor="white",
                    markeredgewidth=0.6, label=series_label(key, multi_k))
            # No direct labels: the curves start close together and converge at
            # high recall, so labels collide at both ends. Identity is carried
            # by the legend plus hue, dash and marker, and pareto_<ds>.txt/.csv
            # give the same numbers as text.
            lo = min(lo, front[0][0])
            hi = max(hi, front[-1][0])
            for rec, val, row in front:
                fronts.append({"dataset": ds, "topk": topk, "panel": panel, "method": m,
                               "k_ranks": k, "recall": rec, "value": val,
                               "stop": row["stop"], "beam_width": row["beam_width"],
                               "gamma": "" if row["gamma"] is None else row["gamma"]})
        ax.set_ylabel(ylabel, fontsize=8, color=INK)
        ax.set_xlabel(f"Recall {topk}@{topk}" + (" (%)" if linear else ""), fontsize=8, color=INK)
    left = min_recall if min_recall is not None else lo
    for ax in axes[0]:
        if linear:
            ax.set_xlim(left=100 * math.floor(left * 20) / 20, right=100.8)
        else:
            ax.set_xscale("logit")
            # Margins in logit space, so neither end sits on the frame.
            ax.set_xlim(left=max(1e-3, left * 0.9),
                        right=1 - max(1e-5, (1 - min(hi, 1 - 1e-5)) * 0.6))
            lo_x, hi_x = ax.get_xlim()
            ticks = [t for t in (0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995,
                                 0.999, 0.9999) if lo_x <= t <= hi_x]
            ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(f"{ds}, recall {topk}@{topk}: Pareto front per method "
                 "(DBS over beam width and γ, conventional over beam width)",
                 fontsize=9, color=INK)
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False,
               fontsize=7, labelcolor=INK)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"pareto_{ds}_top{topk}.{fmt}"
    fig.savefig(dst, dpi=200)
    plt.close(fig)
    print(f"wrote {dst}")
    return fronts


def best_at(front, target, higher):
    """Best value on a frontier among points that reach the target recall."""
    ok = [p for p in front if p[0] >= target]
    if not ok:
        return None
    return (max if higher else min)(ok, key=lambda p: p[1])


def table(ds, topk, rows, targets):
    keys = sorted({series_key(r) for r in rows},
                  key=lambda k: (METHOD_ORDER.index(k[0]), k[1]))
    multi_k = defaultdict(bool)
    for m in {k[0] for k in keys}:
        multi_k[m] = sum(1 for k in keys if k[0] == m) > 1
    for panel, title, higher in PANELS:
        fronts = {key: frontier(panel_points([r for r in rows if series_key(r) == key], panel),
                                higher) for key in keys}
        if not any(fronts.values()):
            continue
        what = {"qps_device": "highest QPS on device", "qps_host": "highest QPS on host",
                "mean_pops": "fewest nodes expanded", "mean_vectors_fetched": "fewest vectors read"}[panel]
        print(f"\n{ds}  {topk}@{topk}  {what} reaching each recall")
        cw = 24
        print(f"  {'recall':>6} | " + " | ".join(f"{series_label(k, multi_k):<{cw}}" for k in keys))
        for t in targets:
            cells = []
            for k in keys:
                b = best_at(fronts[k], t, higher)
                txt = f"{si(b[1])} ({config(b[2])})" if b else "—"
                cells.append(f"{txt:<{cw}}")
            print(f"  {t:>6.3f} | " + " | ".join(cells))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, nargs="+", help="adaptive_<dataset>.csv file(s)")
    ap.add_argument("--out", type=Path, required=True,
                    help="directory for figures/ and pareto_<dataset>.csv")
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[0.90, 0.95, 0.98, 0.99, 0.995])
    ap.add_argument("--min-recall", type=float, default=None,
                    help="left edge of the recall axis, as a fraction (default: fit the curves)")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    ap.add_argument("--linear-recall", action="store_true",
                    help="plain recall axis instead of logit")
    args = ap.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        print("no rows", file=sys.stderr)
        return 1
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    for ds, ds_rows in sorted(by_ds.items()):
        fronts = []
        for topk in sorted({r["topk"] for r in ds_rows}):
            cell = [r for r in ds_rows if r["topk"] == topk]
            fronts += plot_cell(ds, topk, cell, args.out / "figures", args.format,
                                args.min_recall, args.linear_recall)
            table(ds, topk, cell, args.targets)
        dst = args.out / f"pareto_{ds}.csv"
        with dst.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "topk", "panel", "method", "k_ranks",
                                              "recall", "value", "stop", "beam_width", "gamma"])
            w.writeheader()
            w.writerows(fronts)
        print(f"\nwrote {dst} ({len(fronts)} frontier points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
