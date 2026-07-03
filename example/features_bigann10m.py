#!/usr/bin/env python3
"""End-to-end feature test for the Jasper Python API on bigann10M.

Exercises: build, search (stable ids), reserve_ids, append (live batch insert),
mark_deleted + n_live/n_tombstoned, search-excludes-deleted, save/load
(tombstones + ids persisted), consolidate, compact (ids stable across
compaction), and get_vector by stable id.

Run (on a GPU node, with libjasper_ffi.so built into python/jasper/lib):
    PYTHONPATH=python python example/features_bigann10m.py \
        --base   $DATA/bigann/bigann10M \
        --queries $DATA/bigann/bigann10kquery \
        --gt      $GT/GT_10M/bigann-10M
"""
import argparse
import os
import tempfile

import torch
import jasper


def section(name):
    print(f"\n=========== {name} ===========", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--gt", default="")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n_neighbors", type=int, default=64)
    ap.add_argument("--distance", default="l2")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--beam_width", type=int, default=64)
    ap.add_argument("--n_delete", type=int, default=500_000)
    ap.add_argument("--n_append", type=int, default=50_000)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a CUDA device"
    failures = []

    def check(cond, msg):
        status = "OK" if cond else "FAIL"
        print(f"  [{status}] {msg}", flush=True)
        if not cond:
            failures.append(msg)

    # ---- load data ----
    section("load data")
    base = jasper.read_bin(args.base, dtype="u8")        # [n, dim] float16 (pinned)
    queries = jasper.read_bin(args.queries, dtype="u8").cuda()
    n_base = base.size(0)
    print(f"  base={tuple(base.shape)} queries={tuple(queries.shape)}", flush=True)
    gt = None
    if args.gt:
        gt_ids, _ = jasper.read_groundtruth(args.gt, k=args.k)
        gt = gt_ids  # [n_queries, k] int32, ids in construction (== slot) space
        print(f"  gt={tuple(gt.shape)}", flush=True)

    # ---- build ----
    section("build")
    g = jasper.Graph.build(base.cuda(), n_neighbors=args.n_neighbors,
                           distance=args.distance)
    print(f"  {g!r}", flush=True)
    check(g.n_vectors == n_base, f"n_vectors == {n_base}")
    check(g.n_live == n_base and g.n_tombstoned == 0, "n_live==n_vectors, n_tombstoned==0")

    def recall(indices):
        if gt is None:
            return None
        nq = indices.size(0)
        hits = sum(len(set(indices[i].tolist()) & set(gt[i].tolist()))
                   for i in range(nq))
        return hits / (nq * args.k)

    # ---- baseline search (returns stable ids) ----
    section("search (baseline)")
    idx, dist = g.search(queries, k=args.k, beam_width=args.beam_width)
    r = recall(idx)
    print(f"  recall@{args.k} = {r}", flush=True)
    check(idx.shape == (queries.size(0), args.k), "search output shape")
    if gt is not None:
        check(r > 0.90, f"baseline recall > 0.90 (got {r:.4f})")

    # ---- reserve_ids ----
    section("reserve_ids")
    nxt = g.n_vectors  # identity ids at construction -> next id == n_vectors
    rid = g.reserve_ids(1000)
    check(rid.numel() == 1000 and rid[0].item() == nxt and rid[-1].item() == nxt + 999,
          f"reserve_ids -> contiguous [{nxt}, {nxt+999}]")

    # ---- append (live batch insert) ----
    section("append")
    # Append query vectors (held out from the base) as brand-new points. The
    # query file has a fixed count, so cap the batch to it.
    n_app = min(args.n_append, queries.size(0))
    new_vecs = queries[:n_app].contiguous()
    before = g.n_vectors
    app_ids = g.append(new_vecs)
    check(app_ids.numel() == n_app, "append returned one id per vector")
    check(g.n_vectors == before + n_app, "n_vectors grew by n_append")
    check(int(app_ids.min()) >= before, "appended ids are fresh (>= old n_vectors)")
    # Each appended vector should find its own id (exact match).
    aidx, _ = g.search(new_vecs, k=args.k, beam_width=args.beam_width)
    self_hit = sum(1 for i in range(n_app)
                   if app_ids[i].item() in set(aidx[i].tolist()))
    frac = self_hit / n_app
    print(f"  appended-vector self-recall@{args.k} = {frac:.4f}", flush=True)
    check(frac > 0.98, "appended vectors are findable by their id")

    # ---- mark_deleted + search excludes deleted ----
    section("mark_deleted")
    torch.manual_seed(0)
    del_ids = torch.randperm(n_base, dtype=torch.int64)[: args.n_delete].to(torch.int32)
    del_set = set(del_ids.tolist())
    g.mark_deleted(del_ids)
    check(g.n_tombstoned == args.n_delete, f"n_tombstoned == {args.n_delete}")
    check(g.n_live == g.n_vectors - args.n_delete, "n_live == n_vectors - n_tombstoned")
    didx, _ = g.search(queries, k=args.k, beam_width=args.beam_width)
    viol = sum(1 for i in range(didx.size(0)) for v in didx[i].tolist() if v in del_set)
    check(viol == 0, f"no deleted id appears in results (violations={viol})")

    # ---- save / load (tombstones + ids persisted) ----
    section("save / load")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "bigann10m.graph")
        g.save(path)
        g2 = jasper.Graph.load(path, dim=args.dim, n_neighbors=args.n_neighbors,
                               distance=args.distance)
        check(g2.n_vectors == g.n_vectors, "loaded n_vectors matches")
        check(g2.n_tombstoned == g.n_tombstoned, "loaded n_tombstoned matches (tombstones persisted)")
        l2idx, _ = g2.search(queries, k=args.k, beam_width=args.beam_width)
        viol2 = sum(1 for i in range(l2idx.size(0)) for v in l2idx[i].tolist() if v in del_set)
        check(viol2 == 0, "loaded graph still excludes deleted ids")
        g2.free()

    # ---- consolidate ----
    section("consolidate")
    g.consolidate()
    check(g.n_tombstoned == 0, "n_tombstoned == 0 after consolidate")
    cidx, _ = g.search(queries, k=args.k, beam_width=args.beam_width)
    viol3 = sum(1 for i in range(cidx.size(0)) for v in cidx[i].tolist() if v in del_set)
    check(viol3 == 0, "no deleted id after consolidate")

    # ---- compact (ids stable across compaction) ----
    section("compact")
    # pick a few live ids present in the current results and snapshot their vectors
    sample = [v for v in cidx[0].tolist() if v not in del_set and v < n_base][:5]
    before_vecs = {i: g.get_vector(i).clone() for i in sample}
    nv_before = g.n_vectors
    g.compact()
    print(f"  n_vectors {nv_before} -> {g.n_vectors}", flush=True)
    check(g.n_vectors <= nv_before, "compact did not grow the graph")
    stable = all(torch.equal(before_vecs[i], g.get_vector(i)) for i in sample)
    check(stable, "get_vector(id) identical before and after compact (ids stable)")
    fidx, _ = g.search(queries, k=args.k, beam_width=args.beam_width)
    viol4 = sum(1 for i in range(fidx.size(0)) for v in fidx[i].tolist() if v in del_set)
    check(viol4 == 0, "no deleted id after compact")
    rf = recall(fidx)
    if gt is not None:
        # recall vs GT is meaningful post-compact only because ids are stable
        print(f"  recall@{args.k} after compact (live GT) approx = {rf}", flush=True)

    g.free()
    section("summary")
    if failures:
        print(f"  FAILED ({len(failures)}): " + "; ".join(failures), flush=True)
        raise SystemExit(1)
    print("  ALL FEATURE CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
