#!/usr/bin/env python3
"""Tables from sbatch/profile_search.sbatch's raw directory.

  <ds>_<method>_top<k>_<placement>.phase.log   run_query, profile build
  <ds>_<method>_top<k>.visit.csv               distortion_test --visit_stats

Phases: for each profiled search, the cycles per pop the kernel spent in each
phase, as a share of the loop, plus the total and the measured pop rate (QPS x
mean pops). "list" adds up the phases that work on the candidate list (pop,
unique, sort, dedup, clip) and so grow with its width; "fetch" is the phase
that issues the per-pop reads (exact for directional search, expand +
estimate for conventional). A width that ran more than once keeps its last
run, so the first pass's cold start drops out.

Visits: per search point, the share of pops a GPU-resident cache of the top
0.1 ... 20% most popped nodes would serve, and the pops before a query first
expands a true neighbor.

    python3 scripts/profile_summary.py profile_search/raw --out profile_search
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

PHASES = ["pop", "exact", "frontier", "expand", "estimate",
          "unique", "sort", "dedup", "clip"]
LIST_PHASES = ["pop", "unique", "sort", "dedup", "clip"]

LOG_RE = re.compile(r"^(?P<ds>.+?)_(?P<m>conventional|lsh|rabitq|pq)_top(?P<k>\d+)_"
                    r"(?P<place>device|host)\.phase\.log$")
QPS_RE = re.compile(r"beam_width=(\d+) limit=\d+(?: gamma=\S+)?(?: rerank=\d+)? "
                    r"duration=\S+ throughput=(\d+) QPS")
RECALL_RE = re.compile(r"Recall@\d+: ([0-9.]+)")
VISIT_RE = re.compile(r"^(?P<ds>.+?)_(?P<m>conventional|lsh|rabitq|pq)_top(?P<k>\d+)\.visit\.csv$")


def read_phase_log(path: Path) -> dict[int, dict]:
    """Width -> its last profiled search."""
    out, cur = {}, None
    for line in path.read_text(errors="replace").splitlines():
        if m := QPS_RE.search(line):
            cur = {"beam_width": int(m[1]), "qps": float(m[2])}
        elif (m := RECALL_RE.search(line)) and cur is not None:
            cur["recall"] = float(m[1])
        elif line.startswith("phase_profile:") and cur is not None:
            for kv in line.split()[1:]:
                key, val = kv.split("=")
                cur[key] = float(val)
            out[cur["beam_width"]] = cur
            cur = None
    return out


def phases(raw: Path, out_dir: Path) -> list[str]:
    rows = []
    for p in sorted(raw.iterdir()):
        m = LOG_RE.match(p.name)
        if not m:
            continue
        for bw, r in sorted(read_phase_log(p).items()):
            q = r.get("queries", 0) or 1
            r.update(dataset=m["ds"], method=m["m"], k=int(m["k"]), placement=m["place"],
                     mean_pops=r.get("pops", 0) / q)
            r["mpops_per_s"] = r["qps"] * r["mean_pops"] / 1e6
            rows.append(r)
    if not rows:
        return [f"no profiled searches in {raw}"]

    fields = ["dataset", "method", "k", "placement", "beam_width", "qps", "recall",
              "queries", "pops", "mean_pops", "mpops_per_s", "cycles_per_pop"] + PHASES + ["rerank"]
    with (out_dir / "phases.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    lines = [f"wrote {out_dir / 'phases.csv'} ({len(rows)} searches)",
             "cycles per pop by phase, as % of the loop; list = pop+unique+sort+dedup+clip"]
    hdr = (f"  {'method':<12} {'width':>5} {'recall':>6} {'Mpops/s':>8} {'cyc/pop':>8} |"
           + "".join(f" {p[:6]:>6}" for p in PHASES) + f" | {'list':>5}")
    key = lambda r: (r["dataset"], r["k"], r["placement"])
    for grp in sorted({key(r) for r in rows}):
        lines += ["", f"{grp[0]}  {grp[1]}@{grp[1]}  on {grp[2]}", hdr, "  " + "-" * (len(hdr) - 2)]
        for r in sorted((r for r in rows if key(r) == grp),
                        key=lambda r: (r["method"], r["beam_width"])):
            tot = r.get("cycles_per_pop", 0) or 1
            share = {p: 100 * r.get(p, 0) / tot for p in PHASES}
            lines.append(
                f"  {r['method']:<12} {r['beam_width']:>5} {r.get('recall', 0):>6.4f} "
                f"{r['mpops_per_s']:>8.1f} {tot:>8.0f} |"
                + "".join(f" {share[p]:>5.1f}%" for p in PHASES)
                + f" | {sum(share[p] for p in LIST_PHASES):>4.1f}%")
    return lines


def visits(raw: Path) -> list[str]:
    lines = []
    for p in sorted(raw.iterdir()):
        m = VISIT_RE.match(p.name)
        if not m:
            continue
        with p.open() as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        hits = [c for c in rows[0] if c.startswith("hit_")]
        lines += ["", f"{m['ds']}  {m['m']}  {m['k']}@{m['k']}  "
                      f"(n_nodes {int(float(rows[0]['n_nodes'])):,}; share of pops served by a "
                      f"cache of the top x% most popped nodes; descent = pops before the first "
                      f"true neighbor)"]
        hdr = (f"  {'width':>5} {'recall':>6} {'pops':>7} {'distinct':>8} |"
               + "".join(f" {float(c[4:]) * 100:>5g}%" for c in hits)
               + f" | {'desc':>6} {'p50':>5} {'p95':>5} {'share':>5} {'never':>5}")
        lines += [hdr, "  " + "-" * (len(hdr) - 2)]
        for r in rows:
            lines.append(
                f"  {r['beam_width']:>5} {float(r['recall']):>6.4f} {float(r['mean_pops']):>7.1f} "
                f"{100 * float(r['distinct_frac']):>7.2f}% |"
                + "".join(f" {100 * float(r[c]):>5.1f}%" for c in hits)
                + f" | {float(r['descent_mean']):>6.1f} {r['descent_p50']:>5} {r['descent_p95']:>5} "
                f"{100 * float(r['descent_share']):>4.0f}% {100 * float(r['never_frac']):>4.1f}%")
    return lines or [f"no --visit_stats CSVs in {raw}"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not args.raw_dir.is_dir():
        print(f"no such directory: {args.raw_dir}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    for name, lines in (("phases.txt", phases(args.raw_dir, args.out)),
                        ("visits.txt", visits(args.raw_dir))):
        text = "\n".join(lines) + "\n"
        (args.out / name).write_text(text)
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
