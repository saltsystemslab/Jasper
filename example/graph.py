import argparse
import torch
import jasper
import time


def parse_args():
    p = argparse.ArgumentParser(description="Jasper graph build & search benchmark")

    # files
    p.add_argument("--vectors", required=True, help="Path to vectors bin file")
    p.add_argument("--queries", required=True, help="Path to query vectors bin file")
    p.add_argument("--groundtruth", required=True, help="Path to groundtruth file")
    p.add_argument("--dtype", default="f32", choices=["f32", "u8"], help="Vector data type (default: f32)")

    # graph build settings
    p.add_argument("--n-neighbors", type=int, default=64, help="Number of neighbors (default: 64)")
    p.add_argument("--distance", default="l2", choices=["l2", "ip"], help="Distance metric (default: l2)")
    p.add_argument("--alpha", type=float, default=1.2, help="Pruning alpha (default: 1.2)")
    p.add_argument("--workspace-budget", default="10GB", help="Workspace memory budget (default: 10GB)")

    # store graph?
    p.add_argument("--out_index", default="", help="Output index path")

    # search settings
    p.add_argument("-k", type=int, default=1, help="Top-k results (default: 1)")
    p.add_argument("--beam-widths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768],
                    help="Beam widths to sweep (default: 1 2 4 8 16 32 64 128 256)")

    return p.parse_args()


def main():
    args = parse_args()

    print("Loading vectors")
    vectors = jasper.read_bin(args.vectors, args.dtype)
    print(f"Loaded {len(vectors)} vectors.")

    print("Start build graph")
    start = time.perf_counter()
    g = jasper.Graph.build(
        vectors,
        n_neighbors=args.n_neighbors,
        distance=args.distance,
        alpha=args.alpha,
        workspace_budget=args.workspace_budget,
    )
    elapsed = time.perf_counter() - start
    print(g)
    print(f"Graph construction complete, time: {elapsed:.4f} seconds.")

    queries = jasper.read_bin(args.queries, args.dtype)
    gt_indices, gt_distances = jasper.read_groundtruth(args.groundtruth, args.k)

    if args.out_index != "":
        print(f"Storing graph to {args.out_index}")
        
    # warmup
    g.search(queries, k=args.k, beam_width=max(args.beam_widths), print_throughput=True)
    time.sleep(1)

    # sweep beam widths
    for bw in args.beam_widths:
        limit = max(args.limit, 2 * bw)
        indices, distances = g.search(queries, k=args.k, beam_width=bw,  print_throughput=True)
        jasper.get_recall(gt_indices, indices, args.k, len(queries))

    g.free()


if __name__ == "__main__":
    main()