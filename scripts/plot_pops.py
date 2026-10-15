#!/usr/bin/env python3
"""Recall vs nodes popped per query — Fig. 10 with the cost taken out.

Reads distortion_test --pops CSVs (<dataset>_<method>_k<k>.csv, one row per
beam_width:limit) and draws one panel per dataset: x = recall@k, y = mean
nodes popped per query (log). Same encoding as Fig. 10 — color = method,
line style + marker = k — so the two figures can be read side by side.

Lower is better here, where Fig. 10 has higher-is-better throughput. Each pop
costs one exact vector read plus one round of edge estimates, so if a method
sits LOWER here but also lower in Fig. 10, it ranks edges better but pays more
per pop (and the reverse).

    python3 scripts/plot_pops.py distortion_data/pops/
    python3 scripts/plot_pops.py distortion_data/pops/ --datasets deep10M sra50 --format png
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_ORDER = ["lsh", "pq", "rabitq"]
METHOD_LABEL = {"lsh": "CPH", "pq": "PQ", "rabitq": "RaBitQ"}  # Fig. 10's names
COLOR = {"lsh": "#2a78d6", "pq": "#eb6834", "rabitq": "#1baf7a"}
K_STYLE = {4: (":", "o"), 8: ("--", "s"), 16: ("-", "^")}

INK = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"

NAME_RE = re.compile(r"^(?P<ds>.+)_(?P<method>lsh|pq|rabitq)_k(?P<k>\d+)\.csv$")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path, help="directory of --pops CSVs")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--min-recall", type=float, default=50.0,
                    help="left edge of the recall axis, in %% (default 50)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    args = ap.parse_args()

    frames = []
    for p in sorted(args.data_dir.glob("*.csv")):
        mt = NAME_RE.match(p.name)
        if mt:
            frames.append(pd.read_csv(p).assign(dataset=mt["ds"]))
    if not frames:
        print(f"no <dataset>_<method>_k<k>.csv files in {args.data_dir}", file=sys.stderr)
        return 1
    df = pd.concat(frames, ignore_index=True)
    datasets = args.datasets or sorted(df.dataset.unique())

    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 3.4),
                             squeeze=False, constrained_layout=True)
    for ax, ds in zip(axes[0], datasets):
        ax.set_axisbelow(True)
        ax.grid(True, which="major", color=GRID, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
            ax.spines[s].set_linewidth(0.6)
        ax.tick_params(colors=MUTED, labelsize=7, length=2)
        ax.set_yscale("log")

        sub = df[df.dataset == ds]
        for m in METHOD_ORDER:
            for k, (ls, mk) in K_STYLE.items():
                d = sub[(sub.method == m) & (sub.k_ranks == k)].sort_values("mean_pops")
                if d.empty:
                    continue
                ax.plot(100 * d.recall, d.mean_pops, ls=ls, marker=mk, color=COLOR[m],
                        markersize=4.5, markeredgecolor="white", markeredgewidth=0.6,
                        linewidth=1.6, label=f"{METHOD_LABEL[m]} r={k}")
        ax.set_xlim(left=args.min_recall, right=100.5)
        ax.set_title(ds, fontsize=9, color=INK)
        ax.set_xlabel("Recall 10@10 (%)", fontsize=8, color=INK)
    axes[0][0].set_ylabel("nodes popped / query (mean)", fontsize=8, color=INK)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False,
               fontsize=7, labelcolor=INK)

    out = args.out or args.data_dir
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"recall_vs_pops.{args.format}"
    fig.savefig(dst, dpi=200)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
