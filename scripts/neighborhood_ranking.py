#!/usr/bin/env python3
"""Within-neighborhood ranking accuracy of the edge estimators, per byte.

Fig. 8's scatter flatters every estimator: most of ||q-v||² is known exactly
(||q-u||² and mag² for LSH/RaBitQ, ||q||² and ||v||² for PQ), so the shared
exact part lines the cloud up along y = x regardless of how good the estimated
part is. What the beam search actually consumes is the ORDER the estimator
puts a popped node's out-edges in — that decides which neighbors enter the
beam. This scores exactly that, per neighborhood (one `sample` of the CSV =
one popped node u against one query q):

  recall@m   |top-m by estimate  ∩  top-m by exact| / m      (m = 1, 4, 8)
  spearman   rank correlation of estimate vs exact over all of u's edges

and plots each against bytes per edge, one line per quantizer through
k = 4/8/16 — accuracy per byte. A summary CSV with the same numbers (and
standard errors) is written next to the figure.

Input: the edge-score CSVs from distortion_test --trajectory or --random
(sbatch/distortion_sweep.sbatch), named <dataset>_<method>_k<k>.csv.

    python3 scripts/neighborhood_ranking.py distortion_data/random/
    python3 scripts/neighborhood_ranking.py distortion_data/trajectory/ --min-step-frac 0.5
    python3 scripts/neighborhood_ranking.py distortion_data/random/ --m 1 4 8 16 --format png
    python3 scripts/neighborhood_ranking.py distortion_data/random/ --dim mydata=384

Bytes/edge mirror the structs (edge_lsh_list, edge_pq_list, edge_rq_list) and
count only per-edge storage; PQ's 4-byte per-node ||v||² is left out.
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

METHOD_ORDER = ["lsh", "pq", "rabitq"]
METHOD_LABEL = {"lsh": "LSH", "pq": "PQ", "rabitq": "RaBitQ"}
# Categorical slots 1-3, fixed per method (validated: CVD ΔE >= 9.2). Aqua is
# under 3:1 on white, so every method also gets its own marker.
COLOR = {"lsh": "#2a78d6", "pq": "#eb6834", "rabitq": "#1baf7a"}
MARKER = {"lsh": "o", "pq": "s", "rabitq": "^"}

INK = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"

DEFAULT_DIMS = {"bigann10M": 128, "deep10M": 96, "gist": 960, "openai": 1536,
                "sra50": 768, "sra500": 768}

NAME_RE = re.compile(r"^(?P<ds>.+)_(?P<method>lsh|pq|rabitq)_k(?P<k>\d+)\.csv$")


def bytes_per_edge(method: str, k: int, dim: int) -> int:
    if method == "lsh":
        return k * (1 if dim <= 128 else 2) + 2       # packed words + bf16 mag²
    if method == "pq":
        return k                                      # one uint8 code per subspace
    if method == "rabitq":
        return 4 * ((8 * k + 31) // 32) + 4           # code words + fp16 rescale + bf16 mag²
    raise ValueError(method)


def to_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pad each neighborhood into a row: [n_samples, max_deg], NaN past its edges."""
    s = df["sample"].to_numpy()
    _, inv = np.unique(s, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    inv = inv[order]
    counts = np.bincount(inv)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    col = np.arange(len(inv)) - starts[inv]
    shape = (len(counts), int(counts.max()))
    exact = np.full(shape, np.nan)
    est = np.full(shape, np.nan)
    exact[inv, col] = df["exact"].to_numpy(np.float64)[order]
    est[inv, col] = df["est"].to_numpy(np.float64)[order]
    return exact, est


def ranks(a: np.ndarray) -> np.ndarray:
    """Per-row 0-based ranks; NaN slots sort last."""
    filled = np.where(np.isnan(a), np.inf, a)
    r = np.empty_like(filled)
    idx = np.argsort(filled, axis=1, kind="stable")
    np.put_along_axis(r, idx, np.arange(a.shape[1], dtype=np.float64)[None, :], axis=1)
    return r


def score(path: Path, ms: list[int], min_step_frac: float) -> dict[str, tuple[float, float, int]]:
    """metric -> (mean, standard error, #neighborhoods)."""
    cols = pd.read_csv(path, nrows=0).columns
    has_steps = "step" in cols and "n_steps" in cols
    df = pd.read_csv(path, usecols=["sample", "exact", "est"]
                     + (["step", "n_steps"] if has_steps else []))
    if min_step_frac > 0:
        # Trajectory samples only (step >= 0): keep pops from the last part of
        # each search, where the candidates are close to q.
        if not has_steps or (df.step < 0).all():
            raise SystemExit(f"{path.name}: --min-step-frac needs trajectory CSVs")
        df = df[df.step >= min_step_frac * df.n_steps]
        if df.empty:
            raise SystemExit(f"{path.name}: no pops left after --min-step-frac")
    exact, est = to_matrix(df)
    valid = ~np.isnan(exact)
    n_valid = valid.sum(axis=1)
    re_, rs = ranks(exact), ranks(est)

    out: dict[str, tuple[float, float, int]] = {}
    for m in ms:
        ok = n_valid > m  # a neighborhood of <= m edges is trivially perfect
        hit = ((re_ < m) & (rs < m) & valid).sum(axis=1)[ok] / m
        out[f"recall@{m}"] = (hit.mean(), hit.std(ddof=1) / np.sqrt(len(hit)), len(hit))

    # Spearman = Pearson of the ranks over each row's valid edges.
    ok = n_valid >= 3
    x = np.where(valid, re_, np.nan)[ok]
    y = np.where(valid, rs, np.nan)[ok]
    x = x - np.nanmean(x, axis=1, keepdims=True)
    y = y - np.nanmean(y, axis=1, keepdims=True)
    rho = np.nansum(x * y, axis=1) / np.sqrt(np.nansum(x * x, axis=1) * np.nansum(y * y, axis=1))
    out["spearman"] = (rho.mean(), rho.std(ddof=1) / np.sqrt(len(rho)), len(rho))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path, help="directory of <dataset>_<method>_k<k>.csv")
    ap.add_argument("--datasets", nargs="*", help="subset of datasets (default: all found)")
    ap.add_argument("--m", nargs="*", type=int, default=[1, 4, 8], help="recall@m cutoffs")
    ap.add_argument("--dim", nargs="*", default=[], metavar="DS=DIM",
                    help="dimension for datasets not in the built-in table")
    ap.add_argument("--min-step-frac", type=float, default=0.0, metavar="F",
                    help="trajectory CSVs only: keep pops with step >= F * n_steps "
                         "(e.g. 0.5 = second half of each search)")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: data_dir)")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    args = ap.parse_args()

    dims = dict(DEFAULT_DIMS)
    for kv in args.dim:
        ds, _, d = kv.partition("=")
        dims[ds] = int(d)

    found: dict[str, dict[tuple[str, int], Path]] = defaultdict(dict)
    for p in sorted(args.data_dir.glob("*.csv")):
        mt = NAME_RE.match(p.name)
        if mt:
            found[mt["ds"]][(mt["method"], int(mt["k"]))] = p
    datasets = [d for d in (args.datasets or sorted(found)) if d in found]
    if not datasets:
        print(f"no <dataset>_<method>_k<k>.csv files in {args.data_dir}", file=sys.stderr)
        return 1
    missing = [d for d in datasets if d not in dims]
    if missing:
        print(f"unknown dim for {missing}; pass --dim DS=DIM", file=sys.stderr)
        return 1

    rows = []
    for ds in datasets:
        for (method, k), path in sorted(found[ds].items()):
            print(f"  {ds} {method} k{k} ...", file=sys.stderr)
            for metric, (mean, se, n) in score(path, args.m, args.min_step_frac).items():
                rows.append(dict(dataset=ds, method=method, k_ranks=k,
                                 bytes_per_edge=bytes_per_edge(method, k, dims[ds]),
                                 metric=metric, mean=mean, stderr=se, n_neighborhoods=n))
    table = pd.DataFrame(rows)

    out = args.out or args.data_dir
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "neighborhood_ranking.csv", index=False)
    wide = table.pivot_table(index=["dataset", "method", "k_ranks", "bytes_per_edge"],
                             columns="metric", values="mean").round(4)
    print(wide.to_string())

    # ── Figure: rows = metric, columns = dataset ──
    metrics = [f"recall@{m}" for m in args.m] + ["spearman"]
    nr, nc = len(metrics), len(datasets)
    fig, axes = plt.subplots(nr, nc, figsize=(2.7 * nc + 0.4, 2.1 * nr + 0.6),
                             squeeze=False, sharey="row", constrained_layout=True)
    for j, ds in enumerate(datasets):
        for i, metric in enumerate(metrics):
            ax = axes[i][j]
            ax.set_axisbelow(True)
            ax.grid(True, color=GRID, linewidth=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(MUTED)
                ax.spines[s].set_linewidth(0.6)
            ax.tick_params(colors=MUTED, labelsize=7, length=2)

            sub = table[(table.dataset == ds) & (table.metric == metric)]
            for method in METHOD_ORDER:
                d = sub[sub.method == method].sort_values("bytes_per_edge")
                if d.empty:
                    continue
                ax.errorbar(d.bytes_per_edge, d["mean"], yerr=d.stderr,
                            color=COLOR[method], marker=MARKER[method], markersize=5,
                            markeredgecolor="white", markeredgewidth=0.8,
                            linewidth=2, capsize=0, label=METHOD_LABEL[method])
                # Name the k at the first point only, to keep the panel quiet.
                if i == 0:
                    for _, r in d.iterrows():
                        ax.annotate(f"k{r.k_ranks}", (r.bytes_per_edge, r["mean"]),
                                    textcoords="offset points", xytext=(4, -9),
                                    fontsize=5.5, color=MUTED)
            if i == 0:
                ax.set_title(ds, fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(metric, fontsize=8, color=INK)
            if i == nr - 1:
                ax.set_xlabel("bytes / edge", fontsize=8, color=INK)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(handles),
               frameon=False, fontsize=8, labelcolor=INK)
    dst = out / f"neighborhood_ranking.{args.format}"
    fig.savefig(dst, dpi=200)
    plt.close(fig)
    print(f"wrote {dst}\nwrote {out / 'neighborhood_ranking.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
