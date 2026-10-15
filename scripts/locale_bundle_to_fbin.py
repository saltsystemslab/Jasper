#!/usr/bin/env python3
"""Lay out a LOCALE benchmark bundle the way the jasper locale scripts read it.

The Hugging Face release (rsynk/locale-benchmark-sra500) ships

    <bundle>/manifest.json
    <bundle>/base/embeddings-00000.fbin ...   one .fbin per shard, each with
                                              its own (n, d) header
    <bundle>/mut<rate>/query.fbin             queries (chunks), one .fbin
    <bundle>/mut<rate>/gt.bin                 uint32 nq, uint32 k, uint32 ids,
                                              float32 dists — exact top-100

while sbatch/bench_locale_ksweep.sbatch reads $DATA/<dataset>/

    base.fbin              ONE file: create_diskann_index takes a single input
    query_mut<rate>.fbin
    gt_mut<rate>.ibin      read by run_query / distortion_test, same layout as gt.bin

So the only real work is concatenating the shards in filename order (which is
the global row order the ground truth indexes into) under one header. Queries
and ground truth are symlinked, not copied.

    python3 scripts/locale_bundle_to_fbin.py \\
        --bundle /scratch/wang.zikun/sra500/embeddings/locale \\
        --out    /scratch/wang.zikun/locale/sra500

Rerunnable: a base.fbin that already has the expected size is left alone, and
a partial one (from an interrupted run) is rebuilt from scratch.

Note from the bundle's README: mut0.05/ and mut0.10/ query vectors come from an
earlier mutation draw than the queries_mut*.parquet sequences. Their gt.bin was
computed against those same vectors, so ANN recall is consistent; only the
link back to the parquet sequences differs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

HDR = struct.Struct("<II")   # uint32 n, uint32 d


def header(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        return HDR.unpack(f.read(8))


def check_fbin(path: Path, dim: int | None = None) -> tuple[int, int]:
    n, d = header(path)
    want = 8 + n * d * 4
    have = path.stat().st_size
    if have != want:
        raise SystemExit(f"{path}: size {have} != 8 + {n}*{d}*4 = {want} (truncated download?)")
    if dim is not None and d != dim:
        raise SystemExit(f"{path}: dim {d}, expected {dim}")
    return n, d


def build_base(bundle: Path, out: Path) -> tuple[int, int]:
    shards = sorted((bundle / "base").glob("embeddings-*.fbin"))
    if not shards:
        raise SystemExit(f"no base shards under {bundle / 'base'}")

    manifest = {}
    if (bundle / "manifest.json").exists():
        manifest = json.loads((bundle / "manifest.json").read_text())

    # Validate every shard before writing anything.
    n_total, dim = 0, None
    for s in shards:
        n, d = check_fbin(s, dim)
        dim = d
        n_total += n
    print(f"  {len(shards)} shards, {n_total:,} x {dim} float32")
    for key, got in (("n", n_total), ("dim", dim), ("num_shards", len(shards))):
        if key in manifest and int(manifest[key]) != got:
            raise SystemExit(f"manifest.json says {key}={manifest[key]}, shards give {got}")

    dst = out / "base.fbin"
    want = 8 + n_total * dim * 4
    if dst.exists() and dst.stat().st_size == want and header(dst) == (n_total, dim):
        print(f"  {dst} already complete ({want / 2**30:.1f} GiB), keeping it")
        return n_total, dim

    free = shutil.disk_usage(out).free + (dst.stat().st_size if dst.exists() else 0)
    if free < want:
        raise SystemExit(f"need {want / 2**30:.1f} GiB at {out}, only {free / 2**30:.1f} GiB free")

    tmp = out / "base.fbin.partial"
    t0 = time.time()
    done = 0
    with open(tmp, "wb") as fo:
        fo.write(HDR.pack(n_total, dim))
        for i, s in enumerate(shards):
            with open(s, "rb") as fi:
                fi.seek(8)                                   # skip the shard header
                shutil.copyfileobj(fi, fo, length=64 << 20)
            done += s.stat().st_size - 8
            if i % 25 == 0 or i == len(shards) - 1:
                rate = done / max(time.time() - t0, 1e-9) / 2**30
                print(f"  [{i + 1}/{len(shards)}] {done / 2**30:.1f} GiB  ({rate:.2f} GiB/s)", flush=True)
    if tmp.stat().st_size != want:
        raise SystemExit(f"{tmp}: wrote {tmp.stat().st_size} bytes, expected {want}")
    os.replace(tmp, dst)
    print(f"  wrote {dst} in {time.time() - t0:.0f} s")
    return n_total, dim


def link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True,
                    help="the embeddings/locale directory of the release")
    ap.add_argument("--out", type=Path, required=True,
                    help="output directory, i.e. $DATA/<dataset> for the ksweep")
    ap.add_argument("--muts", nargs="*", default=["0.00", "0.05", "0.10"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("base:")
    n_base, dim = build_base(args.bundle, args.out)

    print("queries / ground truth:")
    for m in args.muts:
        q, g = args.bundle / f"mut{m}" / "query.fbin", args.bundle / f"mut{m}" / "gt.bin"
        if not q.exists() or not g.exists():
            print(f"  mut{m}: missing {q if not q.exists() else g}, skipped")
            continue
        nq, _ = check_fbin(q, dim)
        gq, gk = header(g)
        if gq != nq:
            raise SystemExit(f"{g}: {gq} rows but {q} has {nq} queries")
        if g.stat().st_size != 8 + 2 * gq * gk * 4:
            raise SystemExit(f"{g}: size does not match uint32 ids + float32 dists for {gq}x{gk}")
        link(q, args.out / f"query_mut{m}.fbin")
        link(g, args.out / f"gt_mut{m}.ibin")
        print(f"  mut{m}: {nq} queries, gt k={gk}")

    print(f"done: {args.out}  ({n_base:,} x {dim})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
