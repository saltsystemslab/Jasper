#!/usr/bin/env python3
"""Turn one `ncu --csv --page raw` profile of distortion_test --pops into
PCIe bytes per query, per pop and per vector fetched.

sbatch/profile_pcie.sbatch runs `distortion_test --pops` under ncu, which
launches the search kernel once per beam:limit pair. So the ncu CSV has one
row per pair, in the same order as the rows of the --pops CSV, and the two
describe the SAME launches: the bytes and the counts come from one search.
This script pairs them and appends one row per beam width to --out:

    dataset,method,k_ranks,placement,beam_width,limit,n_queries,recall,
    kernel_ms,pops_per_query,vectors_per_query,pcie_bytes_per_query,
    pcie_bytes_per_pop,pcie_bytes_per_vector,dram_bytes_per_query

pops_per_query     nodes the search expanded (popped) per query
vectors_per_query  full vectors read per query: one per pop for directional
                   search, one per appended neighbor (+ the medoid) for
                   conventional search
pcie_bytes_*       ncu's pcie__read_bytes (protocol overhead included)
                   divided per query / per pop / per vector
dram_bytes_*       device-memory bytes, for device-resident runs

Columns whose metric was not collected are left blank. (The L2-sector
host-memory metric reads 0 for zero-copy loads on this GPU, so it is not used.)
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import struct
import sys

PCIE = "pcie__read_bytes.sum"
DRAM = "dram__bytes_read.sum"
TIME = "gpu__time_duration.sum"

FIELDS = ["dataset", "method", "k_ranks", "placement", "beam_width", "limit",
          "n_queries", "recall", "kernel_ms", "pops_per_query", "vectors_per_query",
          "pcie_bytes_per_query", "pcie_bytes_per_pop", "pcie_bytes_per_vector",
          "dram_bytes_per_query"]


def read_ncu_rows(path: str) -> list[dict[str, str]]:
    """Rows of an ncu raw CSV, skipping ==PROF== chatter and the units row."""
    with open(path, errors="replace") as f:
        lines = f.read().splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if start is None:
        return []
    body = [l for l in lines[start:] if l.startswith('"')]
    rows = list(csv.DictReader(io.StringIO("\n".join(body))))
    return [r for r in rows if r.get("ID", "").strip()]   # drop the units row


def num(row: dict[str, str], key: str) -> float | None:
    v = row.get(key)
    if v is None:
        return None
    try:
        return float(v.replace(",", "").strip())
    except ValueError:
        return None


def n_queries(fbin: str) -> int:
    with open(fbin, "rb") as f:
        return struct.unpack("<I", f.read(4))[0]


def fmt(v: float | None, digits: int = 1) -> str:
    return "" if v is None else f"{v:.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ncu", required=True, help="ncu --csv --page raw output")
    ap.add_argument("--pops", required=True, help="distortion_test --pops CSV of the same run")
    ap.add_argument("--queries", required=True, help="query .fbin (for the count)")
    ap.add_argument("--out", required=True, help="summary CSV to append to")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--k_ranks", default="")
    ap.add_argument("--placement", required=True)
    args = ap.parse_args()

    rows = read_ncu_rows(args.ncu)
    if not rows:
        print(f"  no profiled kernels in {args.ncu}", file=sys.stderr)
        return 1
    with open(args.pops) as f:
        pops = list(csv.DictReader(f))
    if len(rows) != len(pops):
        print(f"  WARNING: {len(rows)} profiled launches but {len(pops)} --pops rows; "
              f"pairing in order", file=sys.stderr)
    nq = n_queries(args.queries)

    new = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, r in enumerate(rows):
            p = pops[i] if i < len(pops) else {}
            npop = float(p["mean_pops"]) if p.get("mean_pops") else None
            nvec = float(p["mean_vectors_fetched"]) if p.get("mean_vectors_fetched") else None
            pcie, dram, t = num(r, PCIE), num(r, DRAM), num(r, TIME)
            pq = pcie / nq if pcie is not None else None
            w.writerow({
                "dataset": args.dataset, "method": args.method, "k_ranks": args.k_ranks,
                "placement": args.placement,
                "beam_width": p.get("beam_width", ""), "limit": p.get("limit", ""),
                "n_queries": nq, "recall": p.get("recall", ""),
                "kernel_ms": fmt(t / 1e6 if t is not None else None, 3),
                "pops_per_query": fmt(npop), "vectors_per_query": fmt(nvec),
                "pcie_bytes_per_query": fmt(pq),
                "pcie_bytes_per_pop": fmt(pq / npop if pq is not None and npop else None),
                "pcie_bytes_per_vector": fmt(pq / nvec if pq is not None and nvec else None),
                "dram_bytes_per_query": fmt(dram / nq if dram is not None else None),
            })
            print(f"  bw={p.get('beam_width', '?'):>5} recall={p.get('recall', '-'):>8}"
                  f"  pops/q={fmt(npop):>8}  vectors/q={fmt(nvec):>10}"
                  f"  PCIe B/q={fmt(pq):>12}  B/vector={fmt(pq / nvec if pq is not None and nvec else None):>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
