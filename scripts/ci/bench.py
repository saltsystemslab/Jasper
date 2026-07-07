#!/usr/bin/env python3
"""neu-ci performance / regression benchmark for jasperpy.

Builds `bench_all` and runs it across bigann10M, deep10M, gist and openai,
timing insertions (construct), queries (QPS + recall@10), deletions
(mark_deleted / consolidate, with correctness checks) and live appends. It
writes machine-readable results and a chart, and — if a committed baseline
exists — fails (exit 1) when any gated metric regresses beyond its threshold.

Outputs (consumed by neu-ci's `outputs:` -> injected into the README):
    bench/results.csv       one row per dataset
    bench/qps_recall.png    QPS and recall@10 per dataset (current vs baseline)

Regression baseline:
    bench/baseline.csv      committed reference; update deliberately with
                            `python scripts/ci/bench.py --update-baseline`

Env (set by neu-ci's cluster.setup, with sensible defaults):
    ANN_DATA   root of datasets   (default /projects/SaltSystemsLab/ann_data)
    ANN_GT     root of gt files   (default /projects/SaltSystemsLab/ann_gt)
    THRUST_DIR, CONDA_PREFIX      passed to cmake when building bench_all
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO / "bench"
DATA = os.environ.get("ANN_DATA", "/projects/SaltSystemsLab/ann_data")
GT = os.environ.get("ANN_GT", "/projects/SaltSystemsLab/ann_gt")

# Per-dataset params (dtype/dim confirmed from the .bin headers on the cluster).
DATASETS = [
    dict(name="bigann10M", base=f"{DATA}/bigann/bigann10M",
         q=f"{DATA}/bigann/bigann10kquery", gt=f"{GT}/GT_10M/bigann-10M",
         dtype="uint8", dim=128),
    dict(name="deep10M", base=f"{DATA}/deep/deep10M",
         q=f"{DATA}/deep/deep10kquery", gt=f"{GT}/GT_10M/deep-10M",
         dtype="float", dim=96),
    dict(name="gist", base=f"{DATA}/gist/gist_base.bin",
         q=f"{DATA}/gist/gist_query.bin", gt=f"{GT}/gist-gt",
         dtype="float", dim=960),
    dict(name="openai", base=f"{DATA}/openai/openai_base.bin",
         q=f"{DATA}/openai/openai_query.bin", gt=f"{GT}/openai-gt",
         dtype="float", dim=1536),
]

# Metric columns kept in the CSV, in order.
FIELDS = ["dataset", "insert_vps", "query_qps", "recall", "delete_ms",
          "live_recall", "viol", "consolidate_ms", "append_vps"]

# Regression gates. rel = relative drop fraction; abs = absolute drop.
# Higher-is-better metrics unless noted. GPU timing has a few % of noise, so
# throughput thresholds are looser than the correctness ones.
GATES = {
    "recall":      dict(kind="abs_drop", limit=0.01, label="recall@10"),
    "live_recall": dict(kind="abs_drop", limit=0.02, label="post-delete live recall"),
    "query_qps":   dict(kind="rel_drop", limit=0.15, label="query QPS"),
    "insert_vps":  dict(kind="rel_drop", limit=0.20, label="insert throughput"),
    "append_vps":  dict(kind="rel_drop", limit=0.20, label="append throughput"),
    # viol is checked separately: any deleted id returned is a hard failure.
}

# ---- parsing --------------------------------------------------------------
_RE = {
    "insert": re.compile(r"\[INSERT\]\s+construct\s+\d+\s+vectors:\s+([\d.]+)\s+ms\s+\(([\d.]+)\s+vec/s\)"),
    "query":  re.compile(r"\[QUERY\s*\]\s+bw=\d+:\s+([\d.]+)\s+ms,\s+([\d.]+)\s+QPS,\s+recall@\d+=([\d.]+)"),
    "delete": re.compile(r"\[DELETE\]\s+mark_deleted\s+\d+\s+ids:\s+([\d.]+)\s+ms\s+\|\s+query\s+[\d.]+\s+QPS\s+live_recall=([\d.]+)\s+viol=(\d+)"),
    "consol": re.compile(r"\[DELETE\]\s+consolidate:\s+([\d.]+)\s+ms"),
    "append": re.compile(r"\[APPEND\]\s+append\s+\d+\s+vectors:\s+([\d.]+)\s+ms\s+\(([\d.]+)\s+vec/s\)"),
}


def parse_bench_output(name: str, text: str) -> dict:
    """Extract metrics from one bench_all run. Missing lines -> None."""
    row: dict = {"dataset": name}
    m = _RE["insert"].search(text)
    row["insert_vps"] = float(m.group(2)) if m else None
    m = _RE["query"].search(text)
    row["query_qps"] = float(m.group(2)) if m else None
    row["recall"] = float(m.group(3)) if m else None
    m = _RE["delete"].search(text)
    row["delete_ms"] = float(m.group(1)) if m else None
    row["live_recall"] = float(m.group(2)) if m else None
    row["viol"] = int(m.group(3)) if m else None
    m = _RE["consol"].search(text)
    row["consolidate_ms"] = float(m.group(1)) if m else None
    m = _RE["append"].search(text)
    row["append_vps"] = float(m.group(2)) if m else None
    return row


# ---- csv ------------------------------------------------------------------
def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in FIELDS})


def read_csv(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            out[r["dataset"]] = r
    return out


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---- regression -----------------------------------------------------------
def check_regressions(rows: list[dict], baseline: dict[str, dict]) -> list[str]:
    """Return a list of human-readable regression messages (empty == pass)."""
    problems: list[str] = []
    for r in rows:
        ds = r["dataset"]
        # Hard correctness gate: deleted ids must never be returned.
        if _num(r.get("viol")) not in (None, 0.0):
            problems.append(f"{ds}: deletion returned {r['viol']} deleted ids (viol>0)")
        base = baseline.get(ds)
        if not base:
            continue
        for metric, g in GATES.items():
            cur, ref = _num(r.get(metric)), _num(base.get(metric))
            if cur is None or ref is None:
                continue
            if g["kind"] == "abs_drop" and (ref - cur) > g["limit"]:
                problems.append(
                    f"{ds}: {g['label']} {cur:.4f} < baseline {ref:.4f} "
                    f"(drop {ref - cur:.4f} > {g['limit']})")
            elif g["kind"] == "rel_drop" and ref > 0 and (ref - cur) / ref > g["limit"]:
                problems.append(
                    f"{ds}: {g['label']} {cur:.0f} < baseline {ref:.0f} "
                    f"(-{100 * (ref - cur) / ref:.1f}% > {100 * g['limit']:.0f}%)")
    return problems


# ---- plot -----------------------------------------------------------------
def make_plot(rows: list[dict], baseline: dict[str, dict], path: Path) -> bool:
    """Render QPS + recall@10 per dataset (current vs baseline). Returns False
    if matplotlib isn't available (the run still succeeds; CSV is the source)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); skipping chart", flush=True)
        return False
    import numpy as np

    names = [r["dataset"] for r in rows]
    x = np.arange(len(names))
    qps = [(_num(r.get("query_qps")) or 0) for r in rows]
    rec = [(_num(r.get("recall")) or 0) for r in rows]
    b_qps = [(_num(baseline.get(n, {}).get("query_qps")) or 0) for n in names]
    b_rec = [(_num(baseline.get(n, {}).get("recall")) or 0) for n in names]
    has_base = any(b_qps) or any(b_rec)
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    cur_c, base_c = "#2563eb", "#cbd5e1"
    ax1.bar(x - (w/2 if has_base else 0), qps, w if has_base else w*1.6,
            label="current", color=cur_c)
    if has_base:
        ax1.bar(x + w/2, b_qps, w, label="baseline", color=base_c)
    ax1.set_title("Query throughput"); ax1.set_ylabel("QPS")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x - (w/2 if has_base else 0), rec, w if has_base else w*1.6,
            label="current", color=cur_c)
    if has_base:
        ax2.bar(x + w/2, b_rec, w, label="baseline", color=base_c)
    ax2.set_title("Recall@10"); ax2.set_ylabel("recall")
    ax2.set_ylim(0, 1.0)
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"[plot] wrote {path}", flush=True)
    return True


# ---- build + run ----------------------------------------------------------
def build_bench_all() -> Path:
    build = REPO / "build_bench"
    cmake = ["cmake", "-B", str(build), "-S", str(REPO),
             "-DJASPER_BUILD_CMD=ON", "-DJASPER_BUILD_TESTS=OFF",
             "-DJASPER_BUILD_FFI=OFF"]
    if os.environ.get("THRUST_DIR"):
        cmake.append(f"-DThrust_DIR={os.environ['THRUST_DIR']}")
    if os.environ.get("CONDA_PREFIX"):
        cmake.append(f"-DCMAKE_PREFIX_PATH={os.environ['CONDA_PREFIX']}")
    print("[build]", " ".join(cmake), flush=True)
    subprocess.run(cmake, check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "bench_all",
                    "-j", str(os.cpu_count() or 8)], check=True)
    exe = build / "bin" / "bench_all"
    if not exe.exists():
        raise SystemExit(f"bench_all not built at {exe}")
    return exe


def run_one(exe: Path, ds: dict, k: int, beam: int, df: float, na: int) -> dict:
    cmd = [str(exe), "-b", ds["base"], "-q", ds["q"], "-t", ds["dtype"],
           "--dim", str(ds["dim"]), "-k", str(k), "--beam_width", str(beam),
           "-f", str(df), "-a", str(na)]
    if ds.get("gt"):
        cmd += ["-g", ds["gt"]]
    print(f"\n[run] {ds['name']}: {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr, file=sys.stderr)
        # Record an empty row so the dataset still shows up (as a failure).
        return {"dataset": ds["name"], "error": p.stderr.strip()[:200]}
    return parse_bench_output(ds["name"], p.stdout + p.stderr)


# ---- selftest -------------------------------------------------------------
_SAMPLE = """\
##### bench_all: n_base=10000000 dim=128 df=0.05 append=100000 #####
[INSERT]  construct 10000000 vectors: 41230.5 ms  (242548 vec/s)
[QUERY ]  bw=64: 210.35 ms, 47540 QPS, recall@10=0.9873
[DELETE]  mark_deleted 500000 ids: 88.2 ms | query 46110 QPS live_recall=0.9861 viol=0
[DELETE]  consolidate: 512.7 ms
[DELETE]  compact: disabled (mark + consolidate only)
[APPEND]  append 100000 vectors: 640.1 ms  (156225 vec/s)
##### done #####
"""


def selftest() -> int:
    row = parse_bench_output("bigann10M", _SAMPLE)
    assert row["insert_vps"] == 242548.0, row
    assert row["query_qps"] == 47540.0, row
    assert row["recall"] == 0.9873, row
    assert row["live_recall"] == 0.9861 and row["viol"] == 0, row
    assert row["append_vps"] == 156225.0, row
    print("parse OK:", row)
    # regression logic
    base = {"bigann10M": dict(row, query_qps="60000", recall="0.9990")}
    probs = check_regressions([row], base)
    assert any("query QPS" in p for p in probs), probs
    assert any("recall@10" in p for p in probs), probs
    print("regression detection OK:", *probs, sep="\n  ")
    # viol hard gate
    bad = dict(row, viol=3)
    assert any("viol>0" in p for p in check_regressions([bad], {})), "viol gate"
    print("viol gate OK")
    # csv round-trip
    out = BENCH_DIR / "results.csv"
    write_csv([row], out)
    assert read_csv(out)["bigann10M"]["recall"] == "0.9873"
    print(f"csv OK -> {out}")
    make_plot([row], {}, BENCH_DIR / "qps_recall.png")  # best-effort
    print("selftest PASSED")
    return 0


# ---- main -----------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--beam_width", type=int, default=64)
    ap.add_argument("--delete_fraction", type=float, default=0.05)
    ap.add_argument("--n_append", type=int, default=100000)
    ap.add_argument("--only", help="comma-separated dataset names to run")
    ap.add_argument("--update-baseline", action="store_true",
                    help="write results.csv to baseline.csv and exit 0")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    datasets = DATASETS
    if args.only:
        want = set(args.only.split(","))
        datasets = [d for d in DATASETS if d["name"] in want]

    exe = build_bench_all()
    rows = [run_one(exe, d, args.k, args.beam_width, args.delete_fraction,
                    args.n_append) for d in datasets]

    results = BENCH_DIR / "results.csv"
    write_csv(rows, results)
    print(f"\n[results] wrote {results}")

    baseline_path = BENCH_DIR / "baseline.csv"
    baseline = read_csv(baseline_path)
    make_plot(rows, baseline, BENCH_DIR / "qps_recall.png")

    if args.update_baseline:
        write_csv(rows, baseline_path)
        print(f"[baseline] updated {baseline_path} — commit it to set the reference")
        return 0

    hard_errors = [r for r in rows if r.get("error")]
    for r in hard_errors:
        print(f"[error] {r['dataset']}: {r['error']}", file=sys.stderr)

    problems = check_regressions(rows, baseline)
    if not baseline:
        print("[regression] no baseline.csv yet — recording only. "
              "Review results.csv, then commit it via --update-baseline.")
    if problems:
        print("\n[REGRESSION] gated metrics regressed:")
        for p in problems:
            print(f"  - {p}")
    if hard_errors or problems:
        return 1
    print("\n[OK] no regressions" + (" (vs baseline)" if baseline else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
