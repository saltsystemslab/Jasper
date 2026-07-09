#!/usr/bin/env python3
"""neu-ci throughput/recall tradeoff figure for bigann10M.

Builds a bigann10M index once, then sweeps the search beam width with
`run_query` to trace the classic ANNS Pareto curve — query throughput (QPS)
vs. recall@10. Writes:

    bench/tradeoff_bigann10m.png    QPS-vs-recall curve (annotated by beam width)
    bench/tradeoff_bigann10m.csv    the (beam_width, limit, recall, qps) points

Consumed by neu-ci's `outputs:` -> injected into the README (relative path).

Env (set by neu-ci's cluster.setup, with defaults):
    ANN_DATA, ANN_GT, THRUST_DIR, CONDA_PREFIX
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO / "bench"
DATA = os.environ.get("ANN_DATA", "/projects/SaltSystemsLab/ann_data")
GT = os.environ.get("ANN_GT", "/projects/SaltSystemsLab/ann_gt")

# bigann10M params (uint8, dim 128; graph degree 64, L2) — matches bench.py.
BASE = f"{DATA}/bigann/bigann10M"
QUERIES = f"{DATA}/bigann/bigann10kquery"
GROUNDTRUTH = f"{GT}/GT_10M/bigann-10M"
DTYPE, DIM, NEIGHBORS, DISTANCE, K = "uint8", 128, 64, "l2", 10
# beam_width:limit pairs, increasing search effort -> higher recall, lower QPS.
BEAM_LIMITS = ["16:32", "32:64", "64:128", "128:256", "256:512"]

# run_query prints, per sweep point:
#   beam_width=64 limit=128 duration=210.35ms throughput=47540 QPS
#   Recall@10: 0.9873
_BEAM = re.compile(r"beam_width=(\d+)\s+limit=(\d+)\s+duration=([\d.]+)ms\s+"
                   r"throughput=(\d+)\s+QPS")
_RECALL = re.compile(r"Recall@\d+:\s+([\d.]+)")


def parse_sweep(text: str) -> list[dict]:
    """Pair each beam_width/throughput line with the recall line that follows."""
    points: list[dict] = []
    pending = None
    for line in text.splitlines():
        m = _BEAM.search(line)
        if m:
            pending = {"beam_width": int(m.group(1)), "limit": int(m.group(2)),
                       "qps": float(m.group(4)), "recall": None}
            points.append(pending)
            continue
        r = _RECALL.search(line)
        if r and pending is not None:
            pending["recall"] = float(r.group(1))
            pending = None
    return points


def write_csv(points: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["beam_width", "limit", "recall", "qps"])
        w.writeheader()
        for p in points:
            w.writerow({k: p.get(k) for k in ["beam_width", "limit", "recall", "qps"]})


def make_plot(points: list[dict], path: Path) -> bool:
    pts = [p for p in points if p.get("recall") is not None]
    if not pts:
        print("[plot] no (recall,qps) points; skipping", flush=True)
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); skipping chart", flush=True)
        return False
    pts.sort(key=lambda p: p["recall"])
    xs = [p["recall"] for p in pts]
    ys = [p["qps"] for p in pts]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(xs, ys, "-o", color="#2563eb", linewidth=2, markersize=6)
    for p in pts:  # annotate each point with its beam width
        ax.annotate(f"bw={p['beam_width']}", (p["recall"], p["qps"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8,
                    color="#475569")
    ax.set_xlabel("recall@10")
    ax.set_ylabel("throughput (QPS)")
    ax.set_yscale("log")
    ax.set_title("bigann10M — throughput vs. recall")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"[plot] wrote {path}", flush=True)
    return True


def _cmake_flags() -> list[str]:
    flags = []
    if os.environ.get("THRUST_DIR"):
        flags.append(f"-DThrust_DIR={os.environ['THRUST_DIR']}")
    if os.environ.get("CONDA_PREFIX"):
        flags.append(f"-DCMAKE_PREFIX_PATH={os.environ['CONDA_PREFIX']}")
    return flags


def build_tools() -> Path:
    build = REPO / "build_tradeoff"
    subprocess.run(["cmake", "-B", str(build), "-S", str(REPO),
                    "-DJASPER_BUILD_CMD=ON", "-DJASPER_BUILD_TESTS=OFF",
                    "-DJASPER_BUILD_FFI=OFF", *_cmake_flags()], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target",
                    "create_index", "run_query", "-j",
                    str(os.cpu_count() or 8)], check=True)
    for exe in ("create_index", "run_query"):
        if not (build / "bin" / exe).exists():
            raise SystemExit(f"{exe} not built at {build/'bin'/exe}")
    return build / "bin"


def run_sweep(bindir: Path) -> list[dict]:
    tmp = Path(tempfile.mkdtemp(prefix="tradeoff_", dir=str(REPO)))
    idx = tmp / "bigann10m.idx"
    try:
        print("[build index]", flush=True)
        subprocess.run([str(bindir / "create_index"), "-f", BASE, "-t", DTYPE,
                        "-d", DISTANCE, "-n", str(NEIGHBORS), "--dim", str(DIM),
                        "-i", str(idx), "-w", "10GB"], check=True)
        print("[sweep] beam_limits:", " ".join(BEAM_LIMITS), flush=True)
        p = subprocess.run([str(bindir / "run_query"), "-i", str(idx),
                            "-q", QUERIES, "-g", GROUNDTRUTH, "-t", DTYPE,
                            "-d", DISTANCE, "-n", str(NEIGHBORS), "--dim", str(DIM),
                            "-k", str(K), "--beam_limits", *BEAM_LIMITS],
                           capture_output=True, text=True)
        print(p.stdout)
        if p.stderr:
            print(p.stderr, file=sys.stderr)
        if p.returncode != 0:
            raise SystemExit(f"run_query failed (exit {p.returncode})")
        return parse_sweep(p.stdout + p.stderr)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


_SAMPLE = """\
=== Benchmark (k=10) ===
  beam_width=16 limit=32 duration=95.20ms throughput=105042 QPS
  Recall@10: 0.8123
  beam_width=32 limit=64 duration=140.10ms throughput=71377 QPS
  Recall@10: 0.9012
  beam_width=64 limit=128 duration=210.35ms throughput=47540 QPS
  Recall@10: 0.9569
  beam_width=128 limit=256 duration=360.80ms throughput=27716 QPS
  Recall@10: 0.9811
  beam_width=256 limit=512 duration=690.40ms throughput=14484 QPS
  Recall@10: 0.9925
"""


def selftest() -> int:
    pts = parse_sweep(_SAMPLE)
    assert len(pts) == 5, pts
    assert pts[0] == {"beam_width": 16, "limit": 32, "qps": 105042.0, "recall": 0.8123}, pts[0]
    assert pts[-1]["recall"] == 0.9925 and pts[-1]["qps"] == 14484.0, pts[-1]
    # recall increases, qps decreases as beam grows (monotone tradeoff)
    assert all(pts[i]["recall"] < pts[i+1]["recall"] for i in range(len(pts)-1))
    assert all(pts[i]["qps"] > pts[i+1]["qps"] for i in range(len(pts)-1))
    out = BENCH_DIR / "tradeoff_bigann10m.csv"
    write_csv(pts, out)
    assert out.exists()
    make_plot(pts, BENCH_DIR / "tradeoff_bigann10m.png")  # best-effort
    print("selftest PASSED:", len(pts), "points ->", out)
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    bindir = build_tools()
    points = run_sweep(bindir)
    if not points:
        print("[tradeoff] no sweep points parsed", file=sys.stderr)
        return 1
    write_csv(points, BENCH_DIR / "tradeoff_bigann10m.csv")
    make_plot(points, BENCH_DIR / "tradeoff_bigann10m.png")
    graded = [p for p in points if p.get("recall") is not None]
    print(f"[tradeoff] {len(graded)} points: " +
          ", ".join(f"r={p['recall']:.3f}/{p['qps']:.0f}QPS" for p in graded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
