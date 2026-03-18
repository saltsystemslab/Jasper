import torch
import jasper
import time

k = 1

print("Loading vectors")
vectors = jasper.read_bin("/root/data/bigann10M", "uint8")
print(f"Loaded {len(vectors)} vectors.")

print("Start build graph")
start = time.perf_counter()
g = jasper.Graph.build(
    vectors,
    n_neighbors=64,
    distance="l2",
    alpha=1.2,
    workspace_budget="10GB",
)
end = time.perf_counter()
elapsed_time = end - start
print(g)
print(f"Graph construction complete, time: {elapsed_time:.4f} seconds.")

queries = jasper.read_bin("/root/data/bigann10kquery", "uint8")

# gt
gt_indices, gt_distances =  jasper.read_groundtruth("/root/bigann-10M", k)

# warmup
indices, distances = g.search(queries, k=k, beam_width=16, limit=128, print_throughput=False)

indices, distances = g.search(queries, k=k, beam_width=4, limit=128, print_throughput=True)
jasper.get_recall(gt_indices, indices, k, len(queries))

indices, distances = g.search(queries, k=k, beam_width=8, limit=128, print_throughput=True)
jasper.get_recall(gt_indices, indices, k, len(queries))

indices, distances = g.search(queries, k=k, beam_width=16, limit=128, print_throughput=True)
jasper.get_recall(gt_indices, indices, k, len(queries))

indices, distances = g.search(queries, k=k, beam_width=32, limit=128, print_throughput=True)
jasper.get_recall(gt_indices, indices, k, len(queries))

indices, distances = g.search(queries, k=k, beam_width=64, limit=128, print_throughput=True)
jasper.get_recall(gt_indices, indices, k, len(queries))

indices, distances = g.search(queries, k=k, beam_width=128, limit=256, print_throughput=True)
jasper.get_recall(gt_indices, indices, k, len(queries))

indices, distances = g.search(queries, k=k, beam_width=256, limit=512, print_throughput=True)
jasper.get_recall(gt_indices, indices, k, len(queries))
    
    
     
    
g.free()