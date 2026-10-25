#!/usr/bin/env python3
"""Tables from sbatch/bench_list_opts.sbatch's raw directory.

  <ds>_<method>_top<k>_m<mask>.check.log      distortion_test --check_list_opts
  <ds>_<method>_top<k>_<placement>.qps.log    run_query --list_opts 0,<masks>
  <ds>_<method>_top<k>_<placement>.phase.log  the same, profile build

check.txt   per mask: points whose every query made the same pops, exact
            distances, re-rank count and top-k as the reference (list_opts 0),
            and any that differ.
qps.txt     per point: the reference's QPS (median over passes) and each
            mask's speedup over it; recall must match across masks.
phases.txt  per point and mask: cycles per pop, the share in list work (pop,
            filter, unique, sort, dedup, clip), for the prefilter the new
            entries it keeps per pop and the share of pops it skips, and the
            new entries per pop the duplicate check (6b) finds already listed
            or in the top-k.

    python3 scripts/list_opts_summary.py list_opts/raw --out list_opts
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CHECK_NAME = re.compile(r"^(?P<ds>.+?)_(?P<m>lsh|rabitq|pq)_top(?P<k>\d+)_m(?P<mask>\d+)\.check\.log$")
RUN_NAME = re.compile(r"^(?P<ds>.+?)_(?P<m>lsh|rabitq|pq)_top(?P<k>\d+)_(?P<place>device|host)"
                      r"\.(?P<kind>qps|phase)\.log$")
CHECK_RE = re.compile(r"list_check: (?P<point>beam_width=\d+ limit=\d+(?: gamma=\S+)?(?: rerank=\d+)?) "
                      r"base=\d+ variant=\d+ queries=(?P<q>\d+) differ=(?P<d>\d+) "
                      r"pops_differ=(?P<pd>\d+) topk_differ=(?P<td>\d+) first=(?P<first>\S+) "
                      r".*result=(?P<res>\w+)")
QPS_RE = re.compile(r"beam_width=(\d+) limit=\d+(?: gamma=\S+)?(?: rerank=\d+)? "
                    r"list_opts=(\d+) duration=\S+ throughput=(\d+) QPS")
RECALL_RE = re.compile(r"Recall@\d+: ([0-9.]+)")
LIST_PHASES = ["pop", "filter", "unique", "sort", "dedup", "clip"]


def check(raw: Path) -> list[str]:
    lines = ["Exactness: each mask against list_opts 0, query by query "
             "(pops, exact distances, re-rank count, top-k)"]
    found = False
    for p in sorted(raw.iterdir()):
        m = CHECK_NAME.match(p.name)
        if not m:
            continue
        found = True
        rows = [c.groupdict() for c in map(CHECK_RE.search, p.read_text(errors="replace").splitlines()) if c]
        same = sum(r["res"] == "identical" for r in rows)
        tag = f"  {m['ds']:<8} {m['m']:<7} {m['k']:>3}@{m['k']:<4} mask {m['mask']:>2}: "
        if not rows:
            lines.append(tag + "no list_check lines (run failed? see the log)")
            continue
        lines.append(tag + f"{same}/{len(rows)} points identical")
        for r in rows:
            if r["res"] != "identical":
                lines.append(f"      {r['point']}: {r['d']}/{r['q']} queries differ "
                             f"(pops {r['pd']}, top-k {r['td']}; first query:pop {r['first']})")
    return lines if found else [f"no *.check.log in {raw}"]


def read_runs(path: Path) -> dict:
    """(width, mask) -> list of {qps, recall, profile fields} in run order."""
    runs, cur = defaultdict(list), None
    for line in path.read_text(errors="replace").splitlines():
        if m := QPS_RE.search(line):
            cur = {"qps": float(m[3])}
            runs[(int(m[1]), int(m[2]))].append(cur)
        elif cur is not None and (m := RECALL_RE.search(line)):
            cur["recall"] = float(m[1])
        elif cur is not None and line.startswith("phase_profile:"):
            for kv in line.split()[1:]:
                key, val = kv.split("=")
                cur[key] = float(val)
    return runs


def tables(raw: Path, kind: str) -> list[str]:
    out = []
    for p in sorted(raw.iterdir()):
        m = RUN_NAME.match(p.name)
        if not m or m["kind"] != kind:
            continue
        runs = read_runs(p)
        if not runs:
            out += ["", f"{p.name}: no runs with list_opts= in the log"]
            continue
        widths = sorted({w for w, _ in runs})
        masks = sorted({k for _, k in runs})
        out += ["", f"{m['ds']}  {m['m']}  {m['k']}@{m['k']}  on {m['place']}"]
        if kind == "qps":
            hdr = f"  {'width':>5} {'recall':>6} {'QPS m0':>9} |" + "".join(f" {'m' + str(k):>7}" for k in masks if k)
            out += [hdr, "  " + "-" * (len(hdr) - 2)]
            for w in widths:
                med = {k: statistics.median(r["qps"] for r in runs[(w, k)]) for k in masks if runs.get((w, k))}
                recalls = {round(r.get("recall", -1), 4) for k in masks for r in runs.get((w, k), [])}
                base = med.get(0)
                cells = "".join(f" {med[k] / base:>6.2f}x" if base and k in med else f" {'-':>7}"
                                for k in masks if k)
                flag = "" if len(recalls) <= 1 else f"   RECALL DIFFERS {sorted(recalls)}"
                rec = next(iter(recalls)) if recalls else float("nan")
                out.append(f"  {w:>5} {rec:>6.4f} {base or 0:>9.0f} |{cells}{flag}")
        else:
            hdr = f"  {'width':>5} {'mask':>4} {'cyc/pop':>8} {'list':>6} {'kept/pop':>8} {'skipped':>7} " \
                  f"{'dups/pop':>8} |" \
                  + "".join(f" {ph[:6]:>6}" for ph in LIST_PHASES)
            out += [hdr, "  " + "-" * (len(hdr) - 2)]
            for w in widths:
                for k in masks:
                    rs = runs.get((w, k))
                    if not rs or "cycles_per_pop" not in rs[-1]:
                        continue
                    r = rs[-1]                       # last pass: warm
                    tot = r["cycles_per_pop"] or 1
                    lst = sum(r.get(ph, 0) for ph in LIST_PHASES)
                    if k & 1:                        # prefilter on
                        kept = f"{r.get('kept_per_pop', 0):8.2f}"
                        skipped = f"{100 * r.get('skipped_frac', 0):6.1f}%"
                    else:
                        kept, skipped = f"{'-':>8}", f"{'-':>7}"
                    out.append(
                        f"  {w:>5} {k:>4} {tot:>8.0f} {100 * lst / tot:>5.1f}% {kept} {skipped} "
                        f"{r.get('matched_per_pop', 0):8.2f} |"
                        + "".join(f" {r.get(ph, 0):>6.0f}" for ph in LIST_PHASES))
    return out or [f"no *.{kind}.log in {raw}"]


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
    sections = {
        "check.txt": check(args.raw_dir),
        "qps.txt": ["QPS: list_opts 0 (median of passes) and each mask's speedup over it"]
                   + tables(args.raw_dir, "qps"),
        "phases.txt": ["Profile build: cycles per pop (list = pop+filter+unique+sort+dedup+clip), "
                       "and per-phase cycles per pop for the list phases"]
                      + tables(args.raw_dir, "phase"),
    }
    for name, lines in sections.items():
        text = "\n".join(lines) + "\n"
        (args.out / name).write_text(text)
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
