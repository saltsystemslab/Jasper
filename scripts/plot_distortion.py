#!/usr/bin/env python3
"""Plot estimated-vs-true edge distances (paper Fig. 8) for every quantizer.

Reads the edge-score CSVs written by distortion_test --trajectory or --random
(see sbatch/distortion_sweep.sbatch), named <dataset>_<method>_k<k>.csv, and draws
one figure per dataset: rows = method (lsh, pq, rabitq), columns = k_ranks.
Each panel is a log-count 2D histogram of true ||q-v|| (x) against the
estimate (y) with the y = x line.

Every panel of a dataset shares its axis range, bins and color scale, so the
spread is comparable across methods and k — a wider band really is a worse
estimator, not a different zoom.

    python3 scripts/plot_distortion.py distortion_data/random/
    python3 scripts/plot_distortion.py distortion_data/random/ --datasets bigann10M openai --k 8
    python3 scripts/plot_distortion.py distortion_data/random/ --format png --out figs/

CSV units are the search's: squared L2 (plotted as its square root, so the
axes read ||q-v||; negative estimates clip to 0), or -<q,v> for ip (plotted as
<q,v>).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

METHOD_ORDER = ["lsh", "pq", "rabitq"]
METHOD_LABEL = {"lsh": "LSH", "pq": "PQ", "rabitq": "RaBitQ"}

# Single-hue sequential ramp (light -> dark blue): density is magnitude, so
# one hue; the lightest step recedes toward the white panel for "few".
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)

INK = "#2b2b2b"      # primary text
MUTED = "#6b6b6b"    # axes, ticks, reference line
GRID = "#e6e6e6"

NAME_RE = re.compile(r"^(?P<ds>.+)_(?P<method>lsh|pq|rabitq)_k(?P<k>\d+)\.csv$")


def load(path: Path) -> tuple[np.ndarray, np.ndarray, str, int]:
    """Return (true, est) on the plotted scale, the distance kind, and #clipped."""
    df = pd.read_csv(path, usecols=["distance", "exact", "est"])
    kind = str(df["distance"].iloc[0]) if len(df) else "l2"
    exact = df["exact"].to_numpy(np.float64)
    est = df["est"].to_numpy(np.float64)
    if kind == "l2":
        clipped = int((est < 0).sum())
        return np.sqrt(np.maximum(exact, 0)), np.sqrt(np.maximum(est, 0)), kind, clipped
    return -exact, -est, kind, 0  # back to <q,v>


def stats(true: np.ndarray, est: np.ndarray, kind: str) -> str:
    if kind == "l2":
        ok = true > 0
        rel = np.abs(est[ok] - true[ok]) / true[ok]
        return f"mean rel err {rel.mean():.3f}\np95 {np.percentile(rel, 95):.3f}"
    r = np.corrcoef(true, est)[0, 1]
    return f"Pearson r {r:.3f}"


def plot_dataset(ds: str, files: dict[tuple[str, int], Path], ks: list[int],
                 bins: int, out: Path, fmt: str) -> Path:
    methods = [m for m in METHOD_ORDER if any((m, k) in files for k in ks)]
    data = {key: load(p) for key, p in files.items() if key[0] in methods and key[1] in ks}
    if not data:
        raise SystemExit(f"{ds}: no CSVs for the requested methods/k")
    kind = next(iter(data.values()))[2]

    # Shared range from robust percentiles of every panel's true AND estimated
    # values, so a few wild estimates cannot shrink the bulk to a corner.
    allv = np.concatenate([np.concatenate([t, e]) for t, e, _, _ in data.values()])
    lo, hi = np.percentile(allv, [0.05, 99.95])
    pad = 0.03 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    edges = np.linspace(lo, hi, bins + 1)

    hists = {key: np.histogram2d(t, e, bins=[edges, edges])[0].T
             for key, (t, e, _, _) in data.items()}
    vmax = max(h.max() for h in hists.values())
    norm = LogNorm(vmin=1, vmax=max(vmax, 10))

    nr, nc = len(methods), len(ks)
    fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc + 0.9, 2.5 * nr + 0.5),
                             squeeze=False, sharex=True, sharey=True,
                             constrained_layout=True)
    mesh = None
    for i, m in enumerate(methods):
        for j, k in enumerate(ks):
            ax = axes[i][j]
            ax.set_facecolor("white")
            ax.set_axisbelow(True)  # grid stays behind the density
            ax.grid(True, color=GRID, linewidth=0.6)
            for s in ax.spines.values():
                s.set_color(MUTED)
                s.set_linewidth(0.6)
            ax.tick_params(colors=MUTED, labelsize=7, length=2)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")

            if (m, k) not in hists:
                ax.text(0.5, 0.5, "not run", transform=ax.transAxes,
                        ha="center", va="center", color=MUTED, fontsize=8)
                continue
            h = np.ma.masked_equal(hists[(m, k)], 0)  # empty bins show the surface
            mesh = ax.pcolormesh(edges, edges, h, cmap=CMAP, norm=norm,
                                 rasterized=True, zorder=1)
            ax.plot([lo, hi], [lo, hi], ls="--", lw=0.9, color=MUTED, zorder=2)

            t, e, _, clipped = data[(m, k)]
            note = stats(t, e, kind)
            if clipped:
                note += f"\n{clipped} est<0 clipped"
            ax.text(0.04, 0.96, note, transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

            if i == 0:
                ax.set_title(f"k = {k}", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{METHOD_LABEL[m]}\nestimated", fontsize=8, color=INK)
            if i == nr - 1:
                ax.set_xlabel("true", fontsize=8, color=INK)

    label = "||q - v||" if kind == "l2" else "<q, v>"
    fig.suptitle(f"{ds}: estimated vs true {label}", fontsize=10, color=INK)
    if mesh is not None:
        cb = fig.colorbar(mesh, ax=axes, shrink=0.6, pad=0.02)
        cb.set_label("count (log)", fontsize=8, color=INK)
        cb.ax.tick_params(labelsize=7, colors=MUTED)
        cb.outline.set_visible(False)

    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"distortion_{ds}.{fmt}"
    fig.savefig(dst, dpi=200)
    plt.close(fig)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path, help="directory of <dataset>_<method>_k<k>.csv")
    ap.add_argument("--datasets", nargs="*", help="subset of datasets (default: all found)")
    ap.add_argument("--methods", nargs="*", choices=METHOD_ORDER, help="subset of methods")
    ap.add_argument("--k", nargs="*", type=int, help="subset of k_ranks columns")
    ap.add_argument("--bins", type=int, default=120, help="histogram bins per axis")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: data_dir)")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    args = ap.parse_args()

    found: dict[str, dict[tuple[str, int], Path]] = defaultdict(dict)
    for p in sorted(args.data_dir.glob("*.csv")):
        mt = NAME_RE.match(p.name)
        if mt:
            found[mt["ds"]][(mt["method"], int(mt["k"]))] = p
    if not found:
        print(f"no <dataset>_<method>_k<k>.csv files in {args.data_dir}", file=sys.stderr)
        return 1

    for ds in args.datasets or sorted(found):
        if ds not in found:
            print(f"skip {ds}: no CSVs", file=sys.stderr)
            continue
        files = found[ds]
        if args.methods:
            files = {key: p for key, p in files.items() if key[0] in args.methods}
        ks = sorted(args.k or {k for _, k in files})
        dst = plot_dataset(ds, files, ks, args.bins, args.out or args.data_dir, args.format)
        print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
