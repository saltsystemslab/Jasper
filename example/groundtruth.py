import argparse
import jasper


def main():
    parser = argparse.ArgumentParser(
        description="Generate exact k-NN ground truth from a base/query bin pair."
    )
    parser.add_argument("vectors", help="Path to base vectors .bin file")
    parser.add_argument("queries", help="Path to query vectors .bin file")
    parser.add_argument("gt_path", help="Output path for the ground truth file")
    parser.add_argument("-k", type=int, default=100,
                        help="Number of nearest neighbors (default: 100)")
    parser.add_argument("-d", "--distance", choices=["l2", "ip"], default="l2",
                        help="Distance function (default: l2)")
    parser.add_argument("--query-batch-size", type=int, default=128,
                        help="Queries per device batch (default: 128)")
    parser.add_argument("--vector-batch-size", type=int, default=200_000,
                        help="Vectors per device batch (default: 200000)")
    parser.add_argument("--dtype", default="f32",
                        help="dtype passed to jasper.read_bin (default: f32)")
    args = parser.parse_args()

    print("reading vectors")
    vectors = jasper.read_bin(args.vectors, dtype=args.dtype).cpu()
    queries = jasper.read_bin(args.queries, dtype=args.dtype).cpu()
    print("finished reading vectors")

    gt_ids, gt_dists = jasper.generate_groundtruth(
        vectors, queries,
        k=args.k,
        distance=args.distance,
        query_batch_size=args.query_batch_size,
        vector_batch_size=args.vector_batch_size,
    )
    jasper.save_groundtruth(args.gt_path, gt_ids, gt_dists)
    print(f"wrote ground truth to {args.gt_path}")


if __name__ == "__main__":
    main()