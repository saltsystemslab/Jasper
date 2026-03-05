import torch
import jasper

vectors = torch.randn(100_000, 128, device="cuda", dtype=torch.float32)

g = jasper.Graph.build(
    vectors,
    n_neighbors=64,
    distance="l2",
    alpha=1.2,
    max_batch_size=10000,
)
print(g)  # Graph(n_vectors=100000, dim=128, R=64, dtype=f32, dist=l2)

queries = torch.randn(10, 128, device="cuda", dtype=torch.float32)
indices, distances = g.search(queries, k=10, beam_width=64, limit=512)

print(indices.shape)     # torch.Size([10, 10])
print(distances.shape)   # torch.Size([10, 10])
print(indices[0])        # nearest 10 neighbor IDs for first query
print(distances[0])      # corresponding distances

dists = torch.cdist(queries, vectors)  # [10, 100000]
gt_dists, gt_indices = dists.topk(10, largest=False)

recall = sum(
    len(set(indices[i].tolist()) & set(gt_indices[i].tolist()))
    for i in range(len(queries))
) / (len(queries) * 10)
print(f"Recall@10: {recall:.3f}")

g.free()