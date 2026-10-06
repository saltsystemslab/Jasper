#!/usr/bin/env python3
"""Tile a query set and its ground truth up to a target batch size.

Small query sets (the locale sets are ~500) leave the GPU almost empty: the
beam-search kernel runs one block per query, so 500 queries is 500 blocks and
the reported QPS measures launch overhead more than search. Repeating the batch
until it saturates the device fixes that without changing what is being
measured.

Recall is unchanged BY CONSTRUCTION, because the output is a whole number of
copies of the input: every original query appears the same number of times, so
the per-query hit rate — and therefore the average — is identical to the
untiled run. (A partial final block would silently reweight the queries in it,
which is why the count is rounded up to a whole repeat rather than cut to the
target exactly.)

Formats, both little-endian with an [int32 count][int32 width] header:
  .fbin query  count=n_queries width=dim,  then n*dim float32
  .ibin gt     count=n_queries width=k,    then n*k uint32 ids,
               optionally followed by n*k float32 distances

The distance block is optional because jasper's read_groundtruth reads it but
get_recall never uses it; locale's .ibin files may or may not carry one. Both
layouts round-trip here unchanged.
"""

import argparse
import os
import struct
import sys


def read_header(path):
    with open(path, "rb") as f:
        head = f.read(8)
    if len(head) < 8:
        sys.exit(f"{path}: too short for an [int32 count][int32 width] header")
    count, width = struct.unpack("<ii", head)
    if count <= 0 or width <= 0:
        sys.exit(f"{path}: nonsensical header (count={count}, width={width})")
    return count, width


def tile_fbin(src, dst, reps):
    with open(src, "rb") as f:
        raw = f.read()
    n, dim = struct.unpack("<ii", raw[:8])
    body = raw[8:]
    want = n * dim * 4
    if len(body) != want:
        sys.exit(f"{src}: expected {want} bytes of float32 payload, found {len(body)}")
    with open(dst, "wb") as f:
        f.write(struct.pack("<ii", n * reps, dim))
        f.write(body * reps)
    return n * reps, dim


def tile_gt(src, dst, reps):
    with open(src, "rb") as f:
        raw = f.read()
    n, k = struct.unpack("<ii", raw[:8])
    body = raw[8:]
    ids_bytes = n * k * 4
    if len(body) == ids_bytes:
        ids, dists = body, b""
    elif len(body) == 2 * ids_bytes:
        ids, dists = body[:ids_bytes], body[ids_bytes:]
    else:
        sys.exit(
            f"{src}: {len(body)} payload bytes is neither ids only ({ids_bytes}) "
            f"nor ids+distances ({2 * ids_bytes}) for n={n}, k={k}"
        )
    with open(dst, "wb") as f:
        f.write(struct.pack("<ii", n * reps, k))
        # Tile each block whole, so row i of the output still lines up with
        # query i of the tiled query file.
        f.write(ids * reps)
        if dists:
            f.write(dists * reps)
    return n * reps, k, bool(dists)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", required=True, help="source .fbin")
    ap.add_argument("--out-query", required=True)
    ap.add_argument("--gt", help="source .ibin (optional; omit for latency-only runs)")
    ap.add_argument("--out-gt")
    ap.add_argument("--target", type=int, default=10000,
                    help="minimum output query count (rounded UP to a whole "
                         "number of repeats; default 10000)")
    args = ap.parse_args()

    if args.gt and not args.out_gt:
        ap.error("--gt requires --out-gt")

    # Writing the output over the input silently destroys a source dataset, and
    # it is easier to do than it looks: a case-insensitive filesystem makes
    # q.fbin and Q.fbin the same file.
    def same_file(a, b):
        if not a or not b:
            return False
        if os.path.exists(a) and os.path.exists(b):
            return os.path.samefile(a, b)
        return os.path.realpath(a) == os.path.realpath(b)

    for src, dst, what in ((args.query, args.out_query, "query"),
                           (args.gt, args.out_gt, "gt")):
        if same_file(src, dst):
            sys.exit(f"refusing to overwrite the source {what} file: "
                     f"{src} and {dst} are the same path")

    n, dim = read_header(args.query)
    reps = max(1, -(-args.target // n))       # ceil
    out_n = n * reps

    if args.gt:
        gt_n, gt_k = read_header(args.gt)
        if gt_n != n:
            sys.exit(f"query/gt count mismatch: {n} queries but {gt_n} gt rows "
                     f"({args.query} vs {args.gt})")

    for path in (args.out_query, args.out_gt):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    tile_fbin(args.query, args.out_query, reps)
    msg = f"tiled {n} -> {out_n} queries (x{reps}, dim {dim})"
    if args.gt:
        _, k, had_dists = tile_gt(args.gt, args.out_gt, reps)
        msg += f"; gt k={k}" + ("" if had_dists else " (ids only, no distance block)")
    print(msg)


if __name__ == "__main__":
    main()
