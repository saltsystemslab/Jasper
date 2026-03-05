import jasper
import torch
import time

g = jasper.Graph(
  "/projects/SaltSystemsLab/ann_index/deep10M.index", 
  dim=96, n_neighbors=64, data_type="f32", distance="l2")

print(g)

queries = torch.randn(10000, 96, device="cuda", dtype=torch.float32)

start = time.perf_counter()
indices, distances = g.search(queries, k=10, beam_width=64)
end = time.perf_counter()
elapsed_time = end - start
print(f"Beam search time: {elapsed_time:.4f} seconds, throughput={10000/elapsed_time}")

print(f"indices.shape={indices.shape}")    
print(f"distances.shape={distances.shape}")

print("indices=", indices.cpu().numpy())
print("distances=", distances.cpu().numpy())