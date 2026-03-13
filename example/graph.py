import torch
import jasper
import time

print("Loading vectors")
vectors = jasper.read_bin("/projects/SaltSystemsLab/ann_data/bigann/bigann10M", "uint8")
# vectors = jasper.read_bin("/projects/SaltSystemsLab/ann_data/bigann/bigann10M", "uint8", max_vectors=100_000)
print(f"Loaded {len(vectors)} vectors.")

print("Start build graph")
start = time.perf_counter()
g = jasper.Graph.build(
    vectors,
    n_neighbors=64,
    distance="l2",
    alpha=1.2,
    max_batch_size=1_000_000,
)
end = time.perf_counter()
elapsed_time = end - start
print(g)
print(f"Graph construction complete, time: {elapsed_time:.4f} seconds.")

queries = jasper.read_bin("/projects/SaltSystemsLab/ann_data/bigann/bigann10kquery", "uint8")

# warmup
# indices, distances = g.search(queries, k=10, beam_width=128, limit=512)

indices, distances = g.search(queries, k=10, beam_width=16, limit=128, print_throughput=True)
indices, distances = g.search(queries, k=10, beam_width=16, limit=128, print_throughput=True)
# indices, distances = g.search(queries, k=10, beam_width=32, limit=128, print_throughput=True)
# indices, distances = g.search(queries, k=10, beam_width=64, limit=128, print_throughput=True)
# indices, distances = g.search(queries, k=10, beam_width=128, limit=256, print_throughput=True)
# indices, distances = g.search(queries, k=10, beam_width=256, limit=512, print_throughput=True)
    
    
     
    
g.free()