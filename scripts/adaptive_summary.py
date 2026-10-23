#!/usr/bin/env python3
"""Beam-width vs adaptive termination, compared at matched recall.

Joins the two halves of sbatch/bench_adaptive_termination.sbatch for every
(dataset, method, k_ranks, k, placement) cell found in <raw_dir>, k being the
search k of recall k@k:

  <ds>_<method>_k<k_ranks>_top<k>_<placement>.qps.log
                                run_query output: per point, one
                                "beam_width=.. limit=.. [gamma=..] .. throughput=.. QPS"
                                line followed by "Recall@k: ..". A point listed
                                more than once (REPEATS) is reduced to its
                                median QPS.
  <ds>_<method>_k<k_ranks>_top<k>_<placement>.pops.csv
                                distortion_test --pops over the same points,
                                with the stop,gamma columns.

placement is device or host (graph in pinned host memory, --on_host).

Writes --out, one row per (cell, point):

  dataset,method,k_ranks,topk,placement,stop,beam_width,limit,gamma,recall,qps,qps_runs,
  recall_pops,mean_pops,p50_pops,p95_pops,capped_frac

and prints, per cell and target recall, the best configuration of each rule
that reaches the target: the highest QPS (recall from the timed run) and,
separately, the fewest mean pops (recall from the pops run). They can be
different configurations, since a smaller candidate list is cheaper per pop.
"Reaches" means recall >= target, with no interpolation between points, so
both rules are judged on configurations that were actually run.

Placement changes where the graph is read from, not the search, so pops and
recall should agree between a cell's device and host runs; the report gives
the largest drift over matching points and flags it past rebuild noise.

    python3 scripts/adaptive_summary.py adaptive_termination/raw \\
        --dataset deep10M --out adaptive_termination/adaptive_deep10M.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

NAME_RE = re.compile(r"^(?P<ds>.+)_(?P<method>conventional|lsh|pq|rabitq)_k(?P<k>\d+)"
                     r"_top(?P<topk>\d+)"
                     r"_(?P<place>device|host)"
                     r"\.(?P<kind>qps\.log|pops\.csv)$")
QPS_RE = re.compile(r"beam_width=(\d+) limit=(\d+)(?: gamma=(\S+))? duration=\S+ "
                    r"throughput=(\d+) QPS")
RECALL_RE = re.compile(r"Recall@\d+: ([0-9.]+)")

FIELDS = ["dataset", "method", "k_ranks", "topk", "placement", "stop", "beam_width", "limit", "gamma",
          "recall", "qps", "qps_runs", "recall_pops", "mean_pops", "p50_pops",
          "p95_pops", "capped_frac", "mean_vectors_fetched", "mean_unique_pops",
          "dup_topk_frac"]


def point_key(bw, limit, gamma):
    """gamma None = beam-width rule. Rounded so log and CSV spellings agree."""
    return (int(bw), int(limit), None if gamma in (None, "") else round(float(gamma), 6))


def read_qps(path: Path) -> dict:
    """key -> {"qps": [..], "recall": [..]}, in first-seen order."""
    out: dict = {}
    pending = None
    for line in path.read_text(errors="replace").splitlines():
        m = QPS_RE.search(line)
        if m:
            pending = point_key(m[1], m[2], m[3])
            out.setdefault(pending, {"qps": [], "recall": []})["qps"].append(float(m[4]))
            continue
        m = RECALL_RE.search(line)
        if m and pending is not None:
            out[pending]["recall"].append(float(m[1]))
            pending = None
    return out


def read_pops(path: Path) -> dict:
    out: dict = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            out.setdefault(point_key(r["beam_width"], r["limit"], r.get("gamma")), r)
    return out


def fnum(x):
    return None if x in (None, "") else float(x)


def cell_rows(ds, method, k, topk, place, qps: dict, pops: dict) -> list[dict]:
    rows = []
    for key in list(qps) + [k_ for k_ in pops if k_ not in qps]:
        bw, limit, gamma = key
        q, p = qps.get(key), pops.get(key)
        rows.append({
            "dataset": ds, "method": method, "k_ranks": k, "topk": topk,
            "placement": place,
            "stop": "beam" if gamma is None else "adaptive",
            "beam_width": bw, "limit": limit, "gamma": "" if gamma is None else gamma,
            "recall": statistics.fmean(q["recall"]) if q and q["recall"] else None,
            "qps": statistics.median(q["qps"]) if q else None,
            "qps_runs": len(q["qps"]) if q else 0,
            "recall_pops": fnum(p["recall"]) if p else None,
            "mean_pops": fnum(p["mean_pops"]) if p else None,
            "p50_pops": fnum(p["p50_pops"]) if p else None,
            "p95_pops": fnum(p["p95_pops"]) if p else None,
            "capped_frac": fnum(p["capped_frac"]) if p else None,
            # One per pop for DBS, one per appended neighbor for conventional:
            # the work measure that compares across the two.
            "mean_vectors_fetched": fnum(p.get("mean_vectors_fetched")) if p else None,
            # Duplicate columns; absent from --pops CSVs written before them.
            "mean_unique_pops": fnum(p.get("mean_unique_pops")) if p else None,
            "dup_topk_frac": fnum(p.get("dup_topk_frac")) if p else None,
        })
    return rows


def config(r: dict) -> str:
    if r["stop"] == "beam":
        return f"bw={r['beam_width']}"
    return f"L={r['beam_width']} g={r['gamma']:g}"


def best(rows, stop, target, recall_col, value_col, pick):
    ok = [r for r in rows if r["stop"] == stop and r[recall_col] is not None
          and r[value_col] is not None and r[recall_col] >= target]
    return pick(ok, key=lambda r: r[value_col]) if ok else None


def report(ds, method, k, topk, place, rows, targets):
    if method == "conventional":
        # Beam-width rule only: no rule comparison to make; plot_pareto.py
        # compares it with the DBS methods.
        top = max((r for r in rows if r["recall"] is not None),
                  key=lambda r: r["recall"], default=None)
        print(f"\n{ds}  conventional  recall {topk}@{topk}  on {place}: {len(rows)} beam points"
              + (f", max recall {top['recall']:.4f} ({config(top)})" if top else ""))
        return
    print(f"\n{ds}  {method}  k_ranks={k}  recall {topk}@{topk}  on {place}")
    hdr = (f"  {'recall':>6} | {'beam QPS':>9} {'':<12} {'adapt QPS':>9} {'':<14} {'QPS x':>6} |"
           f" {'beam pops':>9} {'':<12} {'adapt pops':>10} {'':<14} {'pops':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for t in targets:
        bq = best(rows, "beam", t, "recall", "qps", max)
        aq = best(rows, "adaptive", t, "recall", "qps", max)
        bp = best(rows, "beam", t, "recall_pops", "mean_pops", min)
        ap = best(rows, "adaptive", t, "recall_pops", "mean_pops", min)

        def cell(r, col, fmt, width, cw):
            return (f"{format(r[col], fmt):>{width}} {('(' + config(r) + ')'):<{cw}}" if r
                    else f"{'—':>{width}} {'':<{cw}}")
        speed = f"{aq['qps'] / bq['qps']:.2f}x" if aq and bq else "—"
        saved = f"{100 * (ap['mean_pops'] / bp['mean_pops'] - 1):+.1f}%" if ap and bp else "—"
        print(f"  {t:>6.3f} | {cell(bq, 'qps', '.0f', 9, 12)} {cell(aq, 'qps', '.0f', 9, 14)} "
              f"{speed:>6} | {cell(bp, 'mean_pops', '.1f', 9, 12)} "
              f"{cell(ap, 'mean_pops', '.1f', 10, 14)} {saved:>7}")

    # Where each curve tops out. A capacity whose best recall sits below the
    # beam curve's is saturating: its list overflows before gamma stops it.
    def top(sub):
        sub = [r for r in sub if r["recall"] is not None]
        return max(sub, key=lambda r: r["recall"]) if sub else None
    parts = []
    b = top([r for r in rows if r["stop"] == "beam"])
    if b:
        parts.append(f"beam {b['recall']:.4f} ({config(b)})")
    for cap in sorted({r["beam_width"] for r in rows if r["stop"] == "adaptive"}):
        a = top([r for r in rows if r["stop"] == "adaptive" and r["beam_width"] == cap])
        if a:
            parts.append(f"L={cap} {a['recall']:.4f} (g={a['gamma']:g})")
    if parts:
        # One entry per capacity, so wrap between entries, never inside one.
        line = "  max recall: "
        for i, part in enumerate(parts):
            part += "," if i < len(parts) - 1 else ""
            if len(line) + len(part) > 118 and line.strip() != "max recall:":
                print(line.rstrip())
                line = " " * 14
            line += part + " "
        print(line.rstrip())
    capped = [r for r in rows if (r["capped_frac"] or 0) > 0]
    if capped:
        worst = max(capped, key=lambda r: r["capped_frac"])
        print(f"  WARNING: {len(capped)} point(s) hit LIMIT on some queries "
              f"(worst {100 * worst['capped_frac']:.2f}% at {config(worst)}); raise LIMIT")
    duplicates(rows)


def duplicates(rows) -> None:
    """A node reached from two parents carries two estimates that dedup cannot
    pair; the kernel keeps one list entry per node. Report what still gets
    through: re-expansions of nodes clipped from the list and the top-k and
    then re-added (a wasted pop), and repeated top-k ids (a bug if nonzero)."""
    d = [r for r in rows if r["mean_unique_pops"] is not None and r["mean_pops"]]
    if not d:
        return
    reexp = max(d, key=lambda r: 1 - r["mean_unique_pops"] / r["mean_pops"])
    share = 1 - reexp["mean_unique_pops"] / reexp["mean_pops"]
    dup = max(d, key=lambda r: r["dup_topk_frac"])
    if share <= 0 and dup["dup_topk_frac"] <= 0:
        print("  duplicates: none (every pop a distinct node, no repeated top-k id)")
        return
    parts = []
    for stop in ("beam", "adaptive"):
        sub = [r for r in d if r["stop"] == stop]
        if sub:
            avg = statistics.fmean(1 - r["mean_unique_pops"] / r["mean_pops"] for r in sub)
            parts.append(f"{stop} {100 * avg:.1f}%")
    print(f"  re-expanded pops: {', '.join(parts)} on average, worst {100 * share:.1f}% "
          f"at {config(reexp)}")
    if dup["dup_topk_frac"] > 0:
        print(f"  WARNING: top-k repeats an id in {100 * dup['dup_topk_frac']:.1f}% of queries "
              f"at {config(dup)}; the kernel's repeat check is not holding")


def placement_check(ds, method, k, topk, by_place: dict) -> None:
    """Pops and recall are properties of the search, so device and host should
    agree point for point. Codes are rebuilt per run, so allow a little drift."""
    if not {"device", "host"} <= by_place.keys():
        return
    dev = {(r["stop"], r["beam_width"], r["gamma"]): r for r in by_place["device"]}
    worst_pops = worst_recall = 0.0
    n = 0
    for r in by_place["host"]:
        d = dev.get((r["stop"], r["beam_width"], r["gamma"]))
        if not d or d["mean_pops"] is None or r["mean_pops"] is None:
            continue
        n += 1
        if d["mean_pops"] > 0:
            worst_pops = max(worst_pops, abs(r["mean_pops"] / d["mean_pops"] - 1))
        worst_recall = max(worst_recall, abs(r["recall_pops"] - d["recall_pops"]))
    if n:
        flag = "  <- larger than code-rebuild noise; check the host run" \
            if worst_pops > 0.01 or worst_recall > 0.002 else ""
        print(f"\n{ds}  {method}  k_ranks={k}  {topk}@{topk}: host vs device over {n} points: "
              f"max |d pops| {100 * worst_pops:.2f}%, max |d recall| {worst_recall:.4f}{flag}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir", type=Path)
    ap.add_argument("--dataset", help="only cells of this dataset")
    ap.add_argument("--out", type=Path, required=True, help="joined CSV to write")
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[0.90, 0.95, 0.98, 0.99, 0.995])
    args = ap.parse_args()

    cells: dict = {}
    for p in sorted(args.raw_dir.iterdir()):
        m = NAME_RE.match(p.name)
        if not m or (args.dataset and m["ds"] != args.dataset):
            continue
        cells.setdefault((m["ds"], m["method"], int(m["k"]), int(m["topk"]), m["place"]),
                         {})[m["kind"]] = p
    if not cells:
        print(f"no <dataset>_<method>_k<k_ranks>_top<k>_<placement>.{{qps.log,pops.csv}} "
              f"in {args.raw_dir}",
              file=sys.stderr)
        return 1

    all_rows = []
    by_cell: dict = {}
    for (ds, method, k, topk, place), files in sorted(cells.items()):
        qps = read_qps(files["qps.log"]) if "qps.log" in files else {}
        pops = read_pops(files["pops.csv"]) if "pops.csv" in files else {}
        rows = cell_rows(ds, method, k, topk, place, qps, pops)
        if not rows:
            print(f"\n{ds}  {method}  k_ranks={k}  {topk}@{topk}  on {place}: no points parsed",
                  file=sys.stderr)
            continue
        report(ds, method, k, topk, place, rows, args.targets)
        by_cell.setdefault((ds, method, k, topk), {})[place] = rows
        all_rows += rows
    for (ds, method, k, topk), by_place in by_cell.items():
        placement_check(ds, method, k, topk, by_place)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in all_rows:
            w.writerow({c: ("" if r[c] is None else r[c]) for c in FIELDS})
    print(f"\nwrote {args.out} ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
