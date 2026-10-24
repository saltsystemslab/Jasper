#!/usr/bin/env python3
"""Graph build variants compared on recall per node expanded.

Reads sbatch/bench_graph_build.sbatch's raw directory:

  <ds>_<variant>_<method>_top<k>.pops.csv   distortion_test --pops
  <ds>_<variant>_build.log                  create_diskann_index output

and prints, per (method, k): each variant's best recall (the ceiling at the
widest width run), and the nodes expanded per query to reach each target
recall, interpolated log-linearly between the widths run so a coarse grid is
not charged for overshooting. A variant's ratio is against the first variant
listed (the baseline); below 1 means fewer expansions for the same recall.
Then each variant's build time and average degree from its build log.

    python3 scripts/compare_builds.py graph_build/raw --dataset sra50 \\
        --variants L64 L192 L192_2pass
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path


def load(path: Path):
    """(recall, mean_pops) for the beam-width rule, by width."""
    pts = []
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["stop"] != "beam" or r.get("rerank", "0") not in ("", "0"):
                continue
            pts.append((float(r["recall"]), float(r["mean_pops"])))
    return sorted(pts)


def pops_at(pts, target):
    """Fewest pops reaching target recall, log-linear between runs; None past the top."""
    front, best = [], math.inf      # recall ascending, pops strictly falling from the top
    for rec, pops in sorted(pts, key=lambda p: -p[0]):
        if pops < best:
            front.append((rec, pops))
            best = pops
    front.reverse()
    if not front or target > front[-1][0]:
        return None
    if target <= front[0][0]:
        return front[0][1]
    for (r0, p0), (r1, p1) in zip(front, front[1:]):
        if r0 <= target <= r1:
            w = (target - r0) / (r1 - r0) if r1 > r0 else 0.0
            return math.exp((1 - w) * math.log(p0) + w * math.log(p1))
    return None


def build_info(path: Path) -> str:
    if not path.exists():
        return "no build log (reused index)"
    text = path.read_text(errors="replace")
    t = re.search(r"build \(shard[^)]*\): ([0-9.]+) s", text)
    d = re.search(r"Average degree: ([0-9.]+)", text)
    return (f"build {float(t[1]):.0f} s" if t else "build time ?") + \
           (f", avg degree {float(d[1]):.1f}" if d else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir", type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variants", nargs="+", required=True, help="first one is the baseline")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.8, 0.9, 0.95, 0.98, 0.99])
    args = ap.parse_args()

    name_re = re.compile(rf"^{re.escape(args.dataset)}_(?P<v>L\d+(?:_2pass)?)_"
                         r"(?P<m>conventional|lsh|pq|rabitq)_top(?P<k>\d+)\.pops\.csv$")
    cells: dict = {}
    for p in sorted(args.raw_dir.iterdir()):
        m = name_re.match(p.name)
        if m and m["v"] in args.variants:
            cells.setdefault((m["m"], int(m["k"])), {})[m["v"]] = load(p)
    if not cells:
        print(f"no {args.dataset}_<variant>_<method>_top<k>.pops.csv in {args.raw_dir}",
              file=sys.stderr)
        return 1

    base = args.variants[0]
    print(f"{args.dataset}: nodes expanded per query to reach each recall "
          f"(x = vs {base}, below 1 is better)")
    for (method, k), by_v in sorted(cells.items(), key=lambda c: (c[0][1], c[0][0])):
        print(f"\n  {method}  recall {k}@{k}")
        hdr = f"    {'variant':<12} {'ceiling':>8} | " + " ".join(f"{t:>14g}" for t in args.targets)
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        ref = by_v.get(base)
        for v in args.variants:
            pts = by_v.get(v)
            if not pts:
                print(f"    {v:<12} {'—':>8} |")
                continue
            cols = []
            for t in args.targets:
                p = pops_at(pts, t)
                r = pops_at(ref, t) if ref else None
                if p is None:
                    cols.append(f"{'—':>14}")
                elif r is None or v == base:
                    cols.append(f"{p:>14.0f}")
                else:
                    cols.append(f"{p:>7.0f} ({p / r:4.2f}x)")
            print(f"    {v:<12} {max(r for r, _ in pts):>8.4f} | " + " ".join(cols))

    print()
    for v in args.variants:
        print(f"  {v:<12} {build_info(args.raw_dir / f'{args.dataset}_{v}_build.log')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
