import argparse
import jasper


def parse_args():
    p = argparse.ArgumentParser(description="Jasper search benchmark")

    # files
    p.add_argument("--graph", required=True, help="Path to graph index.")
    p.add_argument("--queries", required=True, help="Path to query vectors bin file")
    p.add_argument("--groundtruth", required=True, help="Path to groundtruth file")

    # graph settings
    p.add_argument("--dim", type=int, default=128, help="Vector dimension (default: 128)")
    p.add_argument("--dtype", default="f32", choices=["f32", "u8"], help="Vector data type (default: f32)")
    p.add_argument("--n-neighbors", type=int, default=64, help="Number of neighbors (default: 64)")
    p.add_argument("--distance", default="l2", choices=["l2", "ip"], help="Distance metric (default: l2)")
    p.add_argument("--on_host", action="store_true", help="Construct the graph on host memory")

    # search settings
    p.add_argument("-k", type=int, default=1, help="Top-k results (default: 1)")
    p.add_argument("--beam-widths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
                    help="Beam widths to sweep (default: 1 2 4 8 16 32 64 128 256)")

    return p.parse_args()


def main():
    args = parse_args()

    print("Start loading graph")
    g = jasper.Graph.load(
        args.graph, 
        dim=args.dim, 
        n_neighbors=args.n_neighbors, 
        data_type=args.dtype, 
        distance=args.distance,
        on_host=args.on_host)
    print(f"Graph loaded.")

    queries = jasper.read_bin(args.queries, args.dtype).to(device="cuda")
    gt_indices, _ = jasper.read_groundtruth(args.groundtruth, args.k)

    # warmup
    g.search(queries, k=args.k, beam_width=max(args.beam_widths), print_throughput=False)

    # sweep beam widths
    for bw in args.beam_widths:
        indices, _ = g.search(queries, k=args.k, beam_width=bw, print_throughput=True)
        jasper.get_recall(gt_indices, indices, args.k, len(queries))

    g.free()


if __name__ == "__main__":
    main()